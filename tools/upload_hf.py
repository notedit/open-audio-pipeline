#!/usr/bin/env python
"""把清洗输出上传到 HuggingFace 数据集仓库, 按 ~100GB 一个包组织.

包 = repo 里的一个目录 (HF 单文件上限 50GB, 无法用单个 100GB 文件):
  {repo}/{name}/pack_00000/{shard}-clean.tar
  {repo}/{name}/manifest.json          # 包 -> 分片清单/大小/样本数

- 分片按文件名排序后贪心装包, 装包结果确定性可复现;
- 断点续传: 已上传的包记录在 <data-dir>/.hf_upload_state.json, 重跑跳过;
- 默认私有仓库, --public 公开.

用法:
  python tools/upload_hf.py --data /data/emilia-clean --name emilia \
      --repo leeoxiang/open-audio-data --pack-size-gb 100
"""

from __future__ import annotations

import argparse
import json
import logging
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger("upload")


def build_packs(data_dir: Path, pack_size_gb: float) -> list[list[Path]]:
    """按名字排序贪心装包, 单包总大小不超过 pack_size_gb."""
    shards = sorted(data_dir.glob("*-clean.tar"))
    limit = pack_size_gb * 2**30
    packs: list[list[Path]] = []
    cur: list[Path] = []
    cur_size = 0
    for p in shards:
        size = p.stat().st_size
        if cur and cur_size + size > limit:
            packs.append(cur)
            cur, cur_size = [], 0
        cur.append(p)
        cur_size += size
    if cur:
        packs.append(cur)
    return packs


def count_samples(tar_path: Path) -> int:
    with tarfile.open(tar_path) as tf:
        return sum(1 for m in tf if m.name.endswith(".json"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="run_*.py 输出目录")
    ap.add_argument("--name", required=True, help="repo 内的数据集目录名, 如 emilia / wenetspeech")
    ap.add_argument("--repo", default="leeoxiang/open-audio-data")
    ap.add_argument("--pack-size-gb", type=float, default=100.0)
    ap.add_argument("--public", action="store_true", help="公开仓库 (默认私有)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data_dir = Path(args.data)
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)

    state_path = data_dir / ".hf_upload_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    packs = build_packs(data_dir, args.pack_size_gb)
    total_gb = sum(p.stat().st_size for pack in packs for p in pack) / 2**30
    logger.info("%s: %d shards -> %d packs (%.1f GB total)", args.name,
                sum(len(p) for p in packs), len(packs), total_gb)

    manifest = {"name": args.name, "pack_size_gb": args.pack_size_gb, "packs": {}}
    for i, pack in enumerate(packs):
        pack_id = f"pack_{i:05d}"
        files = [p.name for p in pack]
        size_gb = sum(p.stat().st_size for p in pack) / 2**30
        manifest["packs"][pack_id] = {
            "files": files,
            "size_gb": round(size_gb, 2),
            "num_samples": sum(count_samples(p) for p in pack),
        }
        if state.get(pack_id) == files:
            logger.info("%s already uploaded, skip", pack_id)
            continue
        logger.info("uploading %s: %d shards, %.1f GB ...", pack_id, len(files), size_gb)
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=str(data_dir),
            path_in_repo=f"{args.name}/{pack_id}",
            allow_patterns=files,
            commit_message=f"add {args.name}/{pack_id} ({len(files)} shards, {size_gb:.1f} GB)",
        )
        state[pack_id] = files
        state_path.write_text(json.dumps(state, indent=2))

    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=json.dumps(manifest, indent=2, ensure_ascii=False).encode(),
        path_in_repo=f"{args.name}/manifest.json",
        commit_message=f"update {args.name} manifest",
    )
    logger.info("done: https://huggingface.co/datasets/%s/tree/main/%s", args.repo, args.name)


if __name__ == "__main__":
    main()
