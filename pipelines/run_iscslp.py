#!/usr/bin/env python
"""ISCSLP2026-CoT-TTS 清洗入口 (仅 dialogue_segments; 输出禁止公开再分发).

用法:
  python pipelines/run_iscslp.py \
      --input "/path/movie-zh*/dialogue_segments/*.tar" \
      --output /data/iscslp-clean --config configs/iscslp.yaml
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from audio_pipeline.config import build_pipeline, load_config
from audio_pipeline.datasets.iscslp import iter_iscslp_tar, load_iscslp_metadata
from audio_pipeline.runner import run_shards, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/iscslp.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--limit-samples", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    shards = [Path(p) for p in glob.glob(args.input)]
    if not shards:
        raise SystemExit(f"no shards match {args.input}")

    cache: dict[str, dict] = {}

    def iter_shard(path: Path):
        subset = path.name.split(".")[0]
        if subset not in cache:
            cache.clear()
            meta_json = path.parent.parent / f"{subset}.json"
            cache[subset] = load_iscslp_metadata(meta_json)
        return iter_iscslp_tar(path, cache[subset])

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
