#!/usr/bin/env python
"""多说话人判定双向交叉验证: Sortformer vs DiariZen.

A 组 = Sortformer 判多人而被丢弃的样本 (验证是否误杀);
B 组 = Sortformer 判单人而通过的样本 (验证是否漏放).
两组音频用 DiariZen(独立架构)复判, 输出一致率矩阵与分桶明细.

用法:
  python tools/diar_crosscheck.py --audio-dir <dir> --groups <groups.json> [--device cuda]
groups.json: {"a": {key: sortformer_diarization_meta}, "b": [keys]}
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from audio_pipeline.ops.diarization import DiarizenSpeakerFilter
from audio_pipeline.types import Sample


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-speaker-sec", type=float, default=0.5)
    args = ap.parse_args()

    groups = json.loads(Path(args.groups).read_text())
    op = DiarizenSpeakerFilter(device=args.device, min_speaker_sec=args.min_speaker_sec)

    samples = {}
    for p in glob.glob(f"{args.audio_dir}/*"):
        p = Path(p)
        samples[p.stem] = Sample(key=p.stem, audio_bytes=p.read_bytes(), ext=p.suffix.lstrip("."))

    def run(keys):
        batch = [samples[k] for k in keys if k in samples]
        op.process(batch)
        return {s.key: s.meta["diarization_diarizen"]["num_speakers"] for s in batch}

    # A 组: sortformer=多人
    a_keys = list(groups["a"])
    a_res = run(a_keys)
    def sec2(k):
        v = sorted(groups["a"][k]["speaker_seconds"].values(), reverse=True)
        return v[1] if len(v) > 1 else 0.0
    buckets = {"<1s": [], "1-2s": [], ">=2s": []}
    for k, n in a_res.items():
        b = "<1s" if sec2(k) < 1 else ("1-2s" if sec2(k) < 2 else ">=2s")
        buckets[b].append(n > 1)
    print(f"\nA 组 (Sortformer=多人, n={len(a_res)}): DiariZen 同意 {100*sum(n>1 for n in a_res.values())/len(a_res):.0f}%")
    for b, v in buckets.items():
        if v:
            print(f"  次要说话人 {b}: n={len(v)}, DiariZen 同意(也判多人) {100*sum(v)/len(v):.0f}%")

    # B 组: sortformer=单人
    b_res = run(groups["b"])
    fn = [k for k, n in b_res.items() if n > 1]
    print(f"\nB 组 (Sortformer=单人, n={len(b_res)}): DiariZen 也判单人 {100*(1-len(fn)/len(b_res))*1:.0f}%")
    if fn:
        print(f"  DiariZen 判多人的漏网嫌疑 ({len(fn)}): {fn[:10]}")

    out = {"a": a_res, "b": b_res}
    Path(args.audio_dir).with_name("crosscheck_result.json").write_text(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
