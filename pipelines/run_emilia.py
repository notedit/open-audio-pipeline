#!/usr/bin/env python
"""Emilia-Dataset 清洗入口.

用法:
  python pipelines/run_emilia.py \
      --input "/path/to/Emilia/ZH/*.tar" \
      --output /path/to/output \
      --config configs/emilia.yaml \
      --device cuda:0

多卡并行(每卡一个进程):
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python pipelines/run_emilia.py ... \
        --worker-id $i --num-workers 4 &
  done
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from audio_pipeline.config import build_pipeline, load_config
from audio_pipeline.datasets import iter_emilia_tar
from audio_pipeline.runner import run_shards, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="输入 tar 分片 glob, 如 '/data/Emilia/ZH/*.tar'")
    ap.add_argument("--output", required=True, help="输出目录")
    ap.add_argument("--config", default="configs/emilia.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit-samples", type=int, default=None, help="每分片最多处理条数(调试用)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    shards = [Path(p) for p in glob.glob(args.input)]
    if not shards:
        raise SystemExit(f"no shards match {args.input}")

    run_shards(
        shard_paths=shards,
        iter_shard=iter_emilia_tar,
        make_pipeline=lambda: build_pipeline(cfg, device=args.device),
        output_dir=Path(args.output),
        batch_size=cfg.get("batch_size", 64),
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        limit_samples=args.limit_samples,
    )


if __name__ == "__main__":
    main()
