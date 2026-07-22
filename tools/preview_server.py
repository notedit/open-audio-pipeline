#!/usr/bin/env python
"""清洗结果预览服务 (零依赖, 标准库实现).

读取 run_*.py 的输出目录({shard}-clean.tar + {shard}-drops.jsonl),
提供网页: 统计概览 + 通过/丢弃样本列表 + 在线试听 + 完整标注查看.
被丢弃样本的音频不在输出 tar 里, 传 --source 指向原始输入分片即可试听.

用法:
  python tools/preview_server.py --output-dir /data/emilia-clean \
      --source "/data/Emilia/ZH/*.tar" --port 8791

code-server 下访问: https://<host>/proxy/8791/  (页面内全部用相对路径)
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import tarfile
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

logger = logging.getLogger("preview")

MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac"}
STAGE_ORDER = ["duration", "audiobox", "multi_speaker", "asr_wer", "align", "pause"]


class Index:
    """扫描输出目录, 建立 (shard, key) -> tar 内偏移 的索引, 支持零拷贝读取."""

    def __init__(self, output_dir: Path, source_glob: str | None):
        self.output_dir = output_dir
        self.source_glob = source_glob
        self.lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        passed, drops = [], []
        audio_loc: dict[tuple[str, str], tuple[Path, int, int, str]] = {}
        meta_loc: dict[tuple[str, str], tuple[Path, int, int]] = {}

        for tar_path in sorted(self.output_dir.glob("*-clean.tar")):
            shard = tar_path.name.removesuffix("-clean.tar")
            with tarfile.open(tar_path) as tf:
                metas: dict[str, dict] = {}
                for m in tf:
                    if not m.isfile():
                        continue
                    stem = Path(m.name).stem
                    ext = Path(m.name).suffix.lstrip(".").lower()
                    if ext == "json":
                        metas[stem] = json.loads(tf.extractfile(m).read())
                        meta_loc[(shard, stem)] = (tar_path, m.offset_data, m.size)
                    elif ext in MIME:
                        audio_loc[(shard, stem)] = (tar_path, m.offset_data, m.size, ext)
                for key, meta in metas.items():
                    passed.append({"key": key, "shard": shard, "meta": _slim(meta)})

        for jl in sorted(self.output_dir.glob("*-drops.jsonl")):
            shard = jl.name.removesuffix("-drops.jsonl")
            with open(jl, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        d["shard"] = shard
                        d["meta"] = _slim(d.get("meta") or {})
                        drops.append(d)

        sources: dict[str, Path] = {}
        if self.source_glob:
            for p in glob.glob(self.source_glob):
                p = Path(p)
                sources[p.name.removesuffix(".tar.gz").removesuffix(".tar")] = p

        with self.lock:
            self.passed, self.drops = passed, drops
            self.audio_loc, self.meta_loc, self.sources = audio_loc, meta_loc, sources
            self.cache_dir = self.output_dir / ".preview_cache"
            self._source_audio_index.cache_clear()
        logger.info("indexed: %d passed, %d drops, %d source shards", len(passed), len(drops), len(sources))

        # gz 源分片不能按偏移随机读, 后台把被丢弃样本的音频预提取到缓存目录
        drop_keys: dict[str, set[str]] = {}
        for d in drops:
            drop_keys.setdefault(d["shard"], set()).add(d["key"])
        gz_shards = {s: k for s, k in drop_keys.items()
                     if (p := sources.get(s)) is not None and p.name.endswith(".gz")}
        if gz_shards:
            threading.Thread(target=self._prefetch_gz_drops, args=(gz_shards,), daemon=True).start()

    def _prefetch_gz_drops(self, drop_keys: dict[str, set[str]]) -> None:
        for shard, keys in drop_keys.items():
            cache = self.cache_dir / shard
            missing = {k for k in keys
                       if not any((cache / f"{k}.{e}").exists() for e in MIME)}
            if not missing:
                continue
            cache.mkdir(parents=True, exist_ok=True)
            logger.info("prefetching %d dropped-sample audios from %s ...", len(missing), shard)
            with tarfile.open(self.sources[shard]) as tf:
                for m in tf:
                    stem, ext = Path(m.name).stem, Path(m.name).suffix.lstrip(".").lower()
                    if m.isfile() and stem in missing and ext in MIME:
                        (cache / f"{stem}.{ext}").write_bytes(tf.extractfile(m).read())
                        missing.discard(stem)
                        if not missing:
                            break
            logger.info("prefetch done for %s (%d not found)", shard, len(missing))

    @lru_cache(maxsize=32)
    def _source_audio_index(self, shard: str) -> dict[str, tuple[int, int, str]]:
        """源 tar 的 key -> (offset, size, ext); gz 分片无法偏移读取, 返回空走慢路径."""
        path = self.sources.get(shard)
        if path is None or path.name.endswith(".gz"):
            return {}
        out = {}
        with tarfile.open(path) as tf:
            for m in tf:
                ext = Path(m.name).suffix.lstrip(".").lower()
                if m.isfile() and ext in MIME:
                    out[Path(m.name).stem] = (m.offset_data, m.size, ext)
        return out

    def read_passed_audio(self, shard: str, key: str) -> tuple[bytes, str] | None:
        loc = self.audio_loc.get((shard, key))
        if loc is None:
            return None
        path, off, size, ext = loc
        return _pread(path, off, size), ext

    def read_full_meta(self, shard: str, key: str) -> bytes | None:
        loc = self.meta_loc.get((shard, key))
        if loc is None:
            return None
        path, off, size = loc
        return _pread(path, off, size)

    def read_source_audio(self, shard: str, key: str) -> tuple[bytes, str] | None:
        for ext in MIME:
            p = self.cache_dir / shard / f"{key}.{ext}"
            if p.exists():
                return p.read_bytes(), ext
        idx = self._source_audio_index(shard)
        if key in idx:
            off, size, ext = idx[key]
            return _pread(self.sources[shard], off, size), ext
        path = self.sources.get(shard)
        if path is None:
            return None
        with tarfile.open(path) as tf:  # gz 慢路径
            for m in tf:
                if m.isfile() and Path(m.name).stem == key:
                    ext = Path(m.name).suffix.lstrip(".").lower()
                    if ext in MIME:
                        return tf.extractfile(m).read(), ext
        return None


def _slim(meta: dict) -> dict:
    """列表接口不带逐字对齐数组, 详情页再取全量."""
    meta = dict(meta)
    al = meta.pop("alignment", None)
    if al is not None:
        meta["alignment_len"] = len(al)
    return meta


def _pread(path: Path, offset: int, size: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(size)


def make_handler(index: Index):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(self.path.split("?", 1)[0]).strip("/")
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            try:
                if path == "":
                    self._send(PAGE.encode(), "text/html; charset=utf-8")
                elif path == "data.json":
                    if "reload" in query:
                        index.reload()
                    stats = {"passed": len(index.passed), "drops": len(index.drops)}
                    by_stage: dict[str, int] = {}
                    for d in index.drops:
                        by_stage[d["stage"]] = by_stage.get(d["stage"], 0) + 1
                    stats["by_stage"] = by_stage
                    stats["has_source"] = bool(index.sources)
                    stats["dir"] = str(index.output_dir)
                    body = json.dumps(
                        {"stats": stats, "passed": index.passed, "drops": index.drops},
                        ensure_ascii=False,
                    ).encode()
                    self._send(body, "application/json; charset=utf-8")
                elif path.startswith("audio/"):
                    _, kind, shard, key = path.split("/", 3)
                    r = index.read_passed_audio(shard, key) if kind == "p" else index.read_source_audio(shard, key)
                    if r is None:
                        self._send(b"not found", "text/plain", 404)
                    else:
                        data, ext = r
                        self._send(data, MIME[ext])
                elif path.startswith("meta/"):
                    _, shard, key = path.split("/", 2)
                    body = index.read_full_meta(shard, key)
                    if body is None:
                        self._send(b"{}", "application/json", 404)
                    else:
                        self._send(body, "application/json; charset=utf-8")
                else:
                    self._send(b"not found", "text/plain", 404)
            except BrokenPipeError:
                pass
            except Exception as e:
                logger.exception("request failed: %s", self.path)
                try:
                    self._send(str(e).encode(), "text/plain", 500)
                except Exception:
                    pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>清洗结果预览</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  --good: #0ca30c;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7;
    --muted:#898781; --grid:#2c2c2a; --border: rgba(255,255,255,0.10);
    --good:#0ca30c;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#9085e9;
  }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size:14px; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 20px; }
h1 { font-size: 18px; margin: 0 0 2px; }
.sub { color: var(--muted); font-size: 12px; margin-bottom: 16px; }
.sub a { color: var(--ink2); }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:8px;
  padding:10px 16px; min-width:110px; }
.tile .v { font-size:22px; font-weight:600; }
.tile .l { color:var(--ink2); font-size:12px; margin-top:2px; }
.bar { display:flex; height:14px; border-radius:4px; overflow:hidden; gap:2px;
  background:var(--page); margin: 4px 0 8px; }
.bar div { min-width:2px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--ink2); margin-bottom:16px; }
.legend span.dot { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
.tabs { display:flex; gap:2px; margin-bottom:0; }
.tab { padding:8px 18px; cursor:pointer; border:1px solid var(--border); border-bottom:none;
  border-radius:8px 8px 0 0; background:var(--page); color:var(--ink2); }
.tab.active { background:var(--surface); color:var(--ink); font-weight:600; }
.panel { background:var(--surface); border:1px solid var(--border); border-radius:0 8px 8px 8px; padding:12px; }
.controls { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; align-items:center; }
.controls select, .controls input { background:var(--page); color:var(--ink);
  border:1px solid var(--border); border-radius:6px; padding:5px 8px; font-size:13px; }
.controls input { width: 220px; }
.pager { margin-left:auto; color:var(--ink2); display:flex; gap:8px; align-items:center; }
.pager button { background:var(--page); color:var(--ink); border:1px solid var(--border);
  border-radius:6px; padding:4px 10px; cursor:pointer; }
.pager button:disabled { opacity:.4; cursor:default; }
table { width:100%; border-collapse:collapse; }
th { text-align:left; color:var(--muted); font-weight:500; font-size:12px;
  border-bottom:1px solid var(--grid); padding:6px 8px; white-space:nowrap; }
td { border-bottom:1px solid var(--grid); padding:6px 8px; vertical-align:top; }
tr:hover td { background: color-mix(in srgb, var(--surface) 92%, var(--ink)); }
td.num { font-variant-numeric: tabular-nums; white-space:nowrap; }
td.txt { max-width:420px; }
.t { color:var(--ink); }
.t2 { color:var(--ink2); font-size:12px; margin-top:2px; }
.key { font-size:12px; color:var(--ink2); word-break:break-all; max-width:180px; }
audio { width:220px; height:30px; }
.badge { display:inline-flex; align-items:center; gap:5px; font-size:12px; color:var(--ink2); white-space:nowrap; }
.badge i { width:9px; height:9px; border-radius:2px; display:inline-block; }
.reason { font-size:12px; color:var(--ink2); }
.detail { font-size:12px; }
.detail pre { background:var(--page); border:1px solid var(--border); border-radius:6px;
  padding:10px; overflow:auto; max-height:340px; margin:6px 0; }
button.mini { background:none; border:1px solid var(--border); color:var(--ink2);
  border-radius:6px; padding:2px 8px; cursor:pointer; font-size:12px; }
.bad { color:#d03b3b; }
.empty { color:var(--muted); padding:30px; text-align:center; }
</style>
</head>
<body>
<div class="wrap">
  <h1>清洗结果预览</h1>
  <div class="sub"><span id="dir"></span> · <a href="?" onclick="return doReload()">重新扫描</a></div>
  <div class="tiles" id="tiles"></div>
  <div class="bar" id="bar" title="样本去向"></div>
  <div class="legend" id="legend"></div>
  <div class="tabs">
    <div class="tab active" id="tab-pass" onclick="setTab('pass')">通过样本</div>
    <div class="tab" id="tab-drop" onclick="setTab('drop')">丢弃样本</div>
  </div>
  <div class="panel">
    <div class="controls">
      <select id="stageFilter" onchange="page=0;render()" style="display:none"></select>
      <select id="sortSel" onchange="page=0;render()"></select>
      <input id="search" placeholder="搜索 key / 文本" oninput="page=0;render()">
      <div class="pager">
        <button onclick="page--;render()" id="prev">上一页</button>
        <span id="pageinfo"></span>
        <button onclick="page++;render()" id="next">下一页</button>
      </div>
    </div>
    <div id="tablebox"></div>
  </div>
</div>
<script>
const STAGE_COLORS = { duration:'var(--s1)', audiobox:'var(--s2)', multi_speaker:'var(--s3)',
  asr_wer:'var(--s4)', align:'var(--s5)', pause:'var(--s6)' };
const STAGE_NAMES = { duration:'时长', audiobox:'美学评分', multi_speaker:'多说话人',
  asr_wer:'ASR一致性', align:'对齐失败', pause:'异常停顿' };
const PAGE_SIZE = 100;
let DATA = null, tab = 'pass', page = 0;

const SORTS = {
  pass: [['key','key'],['wer_desc','WER 高→低'],['pq_asc','PQ 低→高'],['dur_desc','时长 长→短']],
  drop: [['key','key'],['wer_desc','WER 高→低'],['pq_asc','PQ 低→高']],
};

function esc(s){ return (s??'').toString().replace(/[&<>"]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function fmt(v, d=2){ return (v===null||v===undefined) ? '—' : Number(v).toFixed(d); }

async function load(reload){
  const r = await fetch('data.json' + (reload ? '?reload=1' : ''));
  DATA = await r.json();
  document.getElementById('dir').textContent = DATA.stats.dir;
  renderStats(); setTab(tab);
}
function doReload(){ load(true); return false; }

function renderStats(){
  const s = DATA.stats, total = s.passed + s.drops;
  const rate = total ? (100*s.passed/total).toFixed(1) : '0';
  let tiles = [['输入样本', total], ['通过', s.passed], ['通过率', rate + '%'], ['丢弃', s.drops]];
  document.getElementById('tiles').innerHTML = tiles.map(
    ([l,v]) => `<div class="tile"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
  const bar = document.getElementById('bar'), leg = document.getElementById('legend');
  let segs = [['通过', s.passed, 'var(--good)']];
  for (const st of Object.keys(STAGE_COLORS))
    if (s.by_stage[st]) segs.push([STAGE_NAMES[st], s.by_stage[st], STAGE_COLORS[st]]);
  bar.innerHTML = segs.map(([n,v,c]) =>
    `<div style="flex:${v};background:${c}" title="${n}: ${v}"></div>`).join('');
  leg.innerHTML = segs.map(([n,v,c]) =>
    `<span><span class="dot" style="background:${c}"></span>${n} ${v}</span>`).join('');
}

function setTab(t){
  tab = t; page = 0;
  document.getElementById('tab-pass').classList.toggle('active', t==='pass');
  document.getElementById('tab-drop').classList.toggle('active', t==='drop');
  const sf = document.getElementById('stageFilter');
  sf.style.display = t==='drop' ? '' : 'none';
  if (t==='drop') {
    sf.innerHTML = '<option value="">全部 stage</option>' + Object.keys(STAGE_COLORS)
      .filter(st => DATA.stats.by_stage[st])
      .map(st => `<option value="${st}">${STAGE_NAMES[st]} (${DATA.stats.by_stage[st]})</option>`).join('');
  }
  document.getElementById('sortSel').innerHTML =
    SORTS[t].map(([v,l]) => `<option value="${v}">${l}</option>`).join('');
  render();
}

function getRows(){
  let rows = tab==='pass' ? DATA.passed : DATA.drops;
  const st = document.getElementById('stageFilter').value;
  if (tab==='drop' && st) rows = rows.filter(d => d.stage===st);
  const q = document.getElementById('search').value.trim().toLowerCase();
  if (q) rows = rows.filter(r => (r.key + ' ' + (r.meta.text||'') + ' ' + ((r.meta.asr||{}).text||'')).toLowerCase().includes(q));
  const sort = document.getElementById('sortSel').value;
  const wer = r => (r.meta.asr && r.meta.asr.wer != null) ? r.meta.asr.wer : -1;
  const pq  = r => r.meta.audiobox ? r.meta.audiobox.PQ : 99;
  rows = rows.slice();
  if (sort==='wer_desc') rows.sort((a,b)=>wer(b)-wer(a));
  else if (sort==='pq_asc') rows.sort((a,b)=>pq(a)-pq(b));
  else if (sort==='dur_desc') rows.sort((a,b)=>(b.meta.duration||0)-(a.meta.duration||0));
  else rows.sort((a,b)=>a.key<b.key?-1:1);
  return rows;
}

function audioCell(r){
  if (tab==='pass')
    return `<audio controls preload="none" src="audio/p/${r.shard}/${encodeURIComponent(r.key)}"></audio>`;
  if (DATA.stats.has_source)
    return `<audio controls preload="none" src="audio/s/${r.shard}/${encodeURIComponent(r.key)}"></audio>`;
  return '<span class="reason" title="启动时传 --source 可试听被丢弃样本">无音频</span>';
}

function textCell(r){
  const m = r.meta, asr = m.asr || {};
  let h = `<div class="t">${esc(m.text || (r.meta.text===''?'(空)':''))}</div>`;
  if (m.orig_text !== undefined && m.orig_text !== m.text)
    h += `<div class="t2">原始: ${esc(m.orig_text)}</div>`;
  else if (asr.text !== undefined && asr.text !== null && asr.text !== m.text)
    h += `<div class="t2">ASR: ${esc(asr.text)}</div>`;
  return h;
}

async function toggleDetail(btn, shard, key){
  const tr = btn.closest('tr'), next = tr.nextElementSibling;
  if (next && next.classList.contains('detailrow')) { next.remove(); return; }
  let meta;
  if (tab==='pass') { meta = await (await fetch(`meta/${shard}/${encodeURIComponent(key)}`)).json(); }
  else { meta = getRows().find(r => r.key===key && r.shard===shard); }
  const row = document.createElement('tr');
  row.className = 'detailrow';
  row.innerHTML = `<td colspan="9" class="detail"><pre>${esc(JSON.stringify(meta, null, 2))}</pre></td>`;
  tr.after(row);
}

function render(){
  const rows = getRows();
  const pages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  page = Math.min(Math.max(0, page), pages-1);
  document.getElementById('pageinfo').textContent = `${rows.length} 条 · ${page+1}/${pages}`;
  document.getElementById('prev').disabled = page===0;
  document.getElementById('next').disabled = page>=pages-1;
  const slice = rows.slice(page*PAGE_SIZE, (page+1)*PAGE_SIZE);
  if (!slice.length) { document.getElementById('tablebox').innerHTML = '<div class="empty">没有数据</div>'; return; }

  let head, body;
  if (tab==='pass') {
    head = '<th>试听</th><th>key</th><th>时长</th><th>PQ</th><th>CE</th><th>WER</th><th>说话人</th><th>最大停顿</th><th>文本</th><th></th>';
    body = slice.map(r => {
      const m=r.meta, ab=m.audiobox||{}, asr=m.asr||{}, di=m.diarization||{}, pa=m.pause||{};
      return `<tr><td>${audioCell(r)}</td><td class="key">${esc(r.key)}</td>
      <td class="num">${fmt(m.duration,1)}s</td><td class="num">${fmt(ab.PQ)}</td><td class="num">${fmt(ab.CE)}</td>
      <td class="num">${asr.wer!=null?fmt(asr.wer,3):'—'}</td><td class="num">${di.num_speakers??'—'}</td>
      <td class="num">${pa.max_gap!=null?fmt(pa.max_gap)+'s':'—'}</td>
      <td class="txt">${textCell(r)}</td>
      <td><button class="mini" onclick="toggleDetail(this,'${r.shard}','${esc(r.key)}')">详情</button></td></tr>`;
    }).join('');
  } else {
    head = '<th>试听</th><th>key</th><th>阶段</th><th>原因</th><th>时长</th><th>PQ</th><th>WER</th><th>文本</th><th></th>';
    body = slice.map(r => {
      const m=r.meta, ab=m.audiobox||{}, asr=m.asr||{};
      const c = STAGE_COLORS[r.stage] || 'var(--muted)';
      return `<tr><td>${audioCell(r)}</td><td class="key">${esc(r.key)}</td>
      <td><span class="badge"><i style="background:${c}"></i>${STAGE_NAMES[r.stage]||esc(r.stage)}</span></td>
      <td class="reason bad">${esc(r.reason)}</td>
      <td class="num">${fmt(m.duration,1)}s</td><td class="num">${fmt(ab.PQ)}</td>
      <td class="num">${asr.wer!=null?fmt(asr.wer,3):'—'}</td>
      <td class="txt">${textCell(r)}</td>
      <td><button class="mini" onclick="toggleDetail(this,'${r.shard}','${esc(r.key)}')">详情</button></td></tr>`;
    }).join('');
  }
  document.getElementById('tablebox').innerHTML =
    `<div style="overflow-x:auto"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

load(false);
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True, help="run_*.py 的输出目录")
    ap.add_argument("--source", default=None, help="原始输入分片 glob, 用于试听被丢弃样本")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    index = Index(Path(args.output_dir), args.source)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(index))
    logger.info("preview server on http://0.0.0.0:%d/  (code-server: /proxy/%d/)", args.port, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
