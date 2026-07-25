#!/usr/bin/env python
"""VoxBox (SparkAudio/voxbox) 清洗入口.

音频分片与元数据分离存放, 按分片名自动解析子集并加载对应 metadata jsonl:
  audios/{subset}/{subset}_NNNN.tar.gz  <->  metadata/{subset}.jsonl

用法:
  python pipelines/run_voxbox.py \
      --input "/path/voxbox/audios/aishell-3/*.tar.gz" \
      --metadata-dir /path/voxbox/metadata \
      --output /data/voxbox-clean --config configs/voxbox.yaml

多 worker 方式同 run_emilia.py (--worker-id/--num-workers)。
注意: emilia_zh / wenetspeech4tts 子集与独立处理的数据重叠, 默认建议跳过。
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from audio_pipeline.config import build_pipeline, load_config
from audio_pipeline.datasets.voxbox import iter_voxbox_tar, load_voxbox_metadata
from audio_pipeline.runner import run_shards, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="音频分片 glob")
    ap.add_argument("--metadata-dir", required=True, help="metadata/*.jsonl 所在目录")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/voxbox.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit-samples", type=int, default=None)
    ap.add_argument(
        "--skip-subsets",
        default="emilia_zh,wenetspeech4tts",
        help="跳过的子集(逗号分隔); 默认排除与独立处理重叠的两个子集",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    skip = {s.strip() for s in args.skip_subsets.split(",") if s.strip()}
    shards = [
        Path(p) for p in glob.glob(args.input)
        if Path(p).name.rsplit("_", 1)[0] not in skip
    ]
    if not shards:
        raise SystemExit(f"no shards match {args.input} (skip={skip})")

    meta_cache: dict[str, dict] = {}

    def iter_shard(path: Path):
        subset = path.name.rsplit("_", 1)[0]
        if subset not in meta_cache:
            meta_cache.clear()  # 每 worker 通常按子集顺序处理, 只留一份省内存
            meta_cache[subset] = load_voxbox_metadata(Path(args.metadata_dir) / f"{subset}.jsonl")
        return iter_voxbox_tar(path, meta_cache[subset])

    run_shards(
        shard_paths=shards,
        iter_shard=iter_shard,
        make_pipeline=lambda: build_pipeline(cfg, device=args.device),
        output_dir=Path(args.output),
        batch_size=cfg.get("batch_size", 64),
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
