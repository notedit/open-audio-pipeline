#!/usr/bin/env python
"""对已清洗的 open-audio-data 重新做多说话人过滤 (只跑 MultiSpeakerFilter, 其余 stage 不重跑).

输入: {src}/{name}/pack_XXXXX/{shard}-clean.tar   (WebDataset: {key}.{flac|mp3|wav} + {key}.json)
输出: {work}/out/{name}/pack_XXXXX/{shard}-clean.tar  同名同布局; 通过样本的音频字节原样保留,
      json 中 diarization 字段替换为新结果 (多模型并集, 见 MultiSpeakerFilter).
      {work}/drops/{name}/{shard}-drops.jsonl         被拒样本的 key/原因/完整 meta
      {work}/done/{name}/{shard}.json                 完成标记 + 计数 (n_in/n_kept/bytes)

分片动态领取: 每个 worker 用 O_EXCL 创建 {work}/claims/{name}/{shard}.claim 抢分片,
worker 数量可随时增减, 不依赖文件列表顺序 (避免静态取模在文件集合变化时错位).
claim 里记录 host/pid, 同机上 pid 已死的 claim 会被回收重领.

用法 (每卡可起多个):
  CUDA_VISIBLE_DEVICES=0 python tools/refilter_diar.py --src /mnt/fsx/leolxliu/open-audio-data \
      --work /opt/dlami/nvme/leolxliu/open-audio-refilter --names emilia-yodas-zh emilia-ko ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from audio_pipeline.ops.diarization import MultiSpeakerFilter
from audio_pipeline.runner import _add_bytes, _batch_producer
from audio_pipeline.types import Sample

logger = logging.getLogger("refilter")
AUDIO_EXTS = {"flac", "mp3", "wav"}


def iter_clean_tar(path: Path):
    """读清洗输出 tar, 按 key 配对音频与 json; meta 即原 json (含 text/language)."""
    pending_json: dict[str, dict] = {}
    pending_audio: dict[str, tuple[bytes, str]] = {}
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            stem, _, ext = m.name.rpartition(".")
            data = tf.extractfile(m).read()
            if ext == "json":
                pending_json[stem] = json.loads(data)
            elif ext in AUDIO_EXTS:
                pending_audio[stem] = (data, ext)
            else:
                continue
            if stem in pending_json and stem in pending_audio:
                meta = pending_json.pop(stem)
                audio, aext = pending_audio.pop(stem)
                yield Sample(key=stem, audio_bytes=audio, ext=aext, text=meta.get("text"),
                             language=meta.get("language"), meta=meta)
    if pending_json or pending_audio:
        logger.warning("%s: %d unpaired json, %d unpaired audio", path.name, len(pending_json), len(pending_audio))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def try_claim(claim: Path) -> bool:
    """O_EXCL 抢占; 同机 pid 已死的旧 claim 通过原子 rename 回收 (多个 worker 同时回收时只有一个成功,
    不会出现 A 刚建的新 claim 被 B 当成旧的删掉、两人写同一个分片的情况)."""
    claim.parent.mkdir(parents=True, exist_ok=True)
    me = f"{socket.gethostname()} {os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                host, pid = claim.read_text().split()
            except Exception:
                return False
            if host != socket.gethostname() or _pid_alive(int(pid)):
                return False
            grave = claim.with_name(f"{claim.name}.stale.{os.getpid()}")
            try:
                os.rename(claim, grave)  # 原子: 并发回收者里只有一个能 rename 成功
            except FileNotFoundError:
                return False
            try:  # rename 之后再确认拿到的确实是那个死 pid 的 claim, 否则还回去
                if grave.read_text().split() != [host, pid]:
                    os.rename(grave, claim)
                    return False
            finally:
                grave.unlink(missing_ok=True)
            logger.info("reclaimed stale %s (pid %s dead)", claim.name, pid)
    return False


def list_shards(src: Path, names: list[str]) -> list[tuple[str, str, Path]]:
    """按 names 顺序、manifest 里的 pack/文件顺序列出全部分片."""
    out = []
    for name in names:
        manifest = json.loads((src / name / "manifest.json").read_text())
        for pack_id, pack in manifest["packs"].items():
            for f in pack["files"]:
                out.append((name, pack_id, src / name / pack_id / f))
    return out


def process_shard(name: str, pack_id: str, path: Path, work: Path, flt: MultiSpeakerFilter,
                  batch_size: int, pool: ThreadPoolExecutor) -> dict:
    out_dir = work / "out" / name / pack_id
    drops_dir = work / "drops" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    drops_dir.mkdir(parents=True, exist_ok=True)
    out_tar = out_dir / path.name
    tmp_tar = out_tar.with_suffix(f"{out_tar.suffix}.{os.getpid()}.tmp")
    stem = path.name.removesuffix(".tar")
    n_kept = 0
    reasons: dict[str, int] = {}
    t0 = time.time()
    with open(drops_dir / f"{stem}-drops.jsonl", "w", encoding="utf-8") as drops_f, \
            tarfile.open(tmp_tar, "w") as out_tf:
        q: Queue = Queue(maxsize=3)
        state = {"n_in": 0, "error": None}
        producer = threading.Thread(target=_batch_producer,
                                    args=(path, iter_clean_tar, batch_size, None, pool, q, state), daemon=True)
        producer.start()
        while (batch := q.get()) is not None:
            alive = []
            for s in batch:
                if s.rejected:  # 解码失败: 原清洗时能解码, 这里失败说明文件损坏, 记录后丢弃
                    drops_f.write(json.dumps({"key": s.key, "stage": "input", "reason": s.reject_reason,
                                              "meta": s.meta}, ensure_ascii=False) + "\n")
                    reasons["input"] = reasons.get("input", 0) + 1
                else:
                    alive.append(s)
            if alive:
                flt.process(alive)
            for s in alive:
                if s.rejected:
                    drops_f.write(json.dumps({"key": s.key, "stage": "multi_speaker", "reason": s.reject_reason,
                                              "meta": s.meta}, ensure_ascii=False) + "\n")
                    k = s.reject_reason.split("@", 1)[-1]
                    reasons[k] = reasons.get(k, 0) + 1
                else:
                    _add_bytes(out_tf, f"{s.key}.{s.ext}", s.audio_bytes)
                    _add_bytes(out_tf, f"{s.key}.json", json.dumps(s.meta, ensure_ascii=False).encode())
                    n_kept += 1
                s.free_wav()
        producer.join()
        if state["error"] is not None:
            raise state["error"]
        n_in = state["n_in"]
    tmp_tar.rename(out_tar)
    stats = {"name": name, "pack": pack_id, "file": path.name, "n_in": n_in, "n_kept": n_kept,
             "drops": reasons, "bytes": out_tar.stat().st_size, "src_bytes": path.stat().st_size,
             "sec": round(time.time() - t0, 1)}
    done = work / "done" / name / f"{stem}.json"
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(stats))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--names", nargs="+", required=True, help="按顺序处理的数据集目录名")
    ap.add_argument("--models", nargs="+",
                    default=["nvidia/diar_streaming_sortformer_4spk-v2", "nvidia/Nemotron-3-Diarization"])
    ap.add_argument("--batch-size", type=int, default=128, help="每批送入 diarization 的样本数")
    ap.add_argument("--diar-batch-size", type=int, default=32)
    ap.add_argument("--decode-threads", type=int, default=8)
    ap.add_argument("--max-shards", type=int, default=None, help="本 worker 最多处理的分片数 (测试用)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-free-gb", type=float, default=500, help="工作盘剩余低于此值时暂停领新分片")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    src, work = Path(args.src), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    flt = MultiSpeakerFilter(model_name=args.models, max_speakers=1, min_speaker_sec=0.0,
                             batch_size=args.diar_batch_size, device=args.device)
    pool = ThreadPoolExecutor(args.decode_threads)
    n_done = 0
    for name, pack_id, path in list_shards(src, args.names):
        stem = path.name.removesuffix(".tar")
        if (work / "done" / name / f"{stem}.json").exists():
            continue
        while (free := shutil.disk_usage(work).free / 2**30) < args.min_free_gb:
            logger.warning("work disk free %.0f GB < %.0f GB, waiting for uploader to drain", free, args.min_free_gb)
            time.sleep(300)
        if not try_claim(work / "claims" / name / f"{stem}.claim"):
            continue
        st = process_shard(name, pack_id, path, work, flt, args.batch_size, pool)
        logger.info("%s/%s %s: %d -> %d kept (%.2f%% dropped) in %.0fs, %.1f samples/s | %s",
                    name, pack_id, path.name, st["n_in"], st["n_kept"],
                    100 * (1 - st["n_kept"] / max(st["n_in"], 1)), st["sec"],
                    st["n_in"] / max(st["sec"], 1e-6), st["drops"])
        n_done += 1
        if args.max_shards and n_done >= args.max_shards:
            break
    logger.info("worker done: %d shards", n_done)


if __name__ == "__main__":
    main()
