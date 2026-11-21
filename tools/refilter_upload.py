#!/usr/bin/env python
"""把 refilter_diar.py 的输出按原 pack 结构覆盖上传到 HF, 与过滤 worker 并行运行.

- 某 pack 的全部分片都有 done 标记后, 上传 {work}/out/{name}/{pack} -> repo:{name}/{pack} (同名覆盖);
- 每传完一个 pack 更新 repo:{name}/manifest.json (num_samples/size_gb 为过滤后的值, 未传的 pack 沿用旧值),
  上传成功后删除本地暂存 tar (done 标记与 drops 保留), 控制 nvme 占用;
- --squash: 一个数据集的全部 pack 传完后执行 super_squash_history (不可逆地清掉 repo 全部历史),
  让被覆盖的旧 LFS 文件不再占存储额度; 默认关闭;
- 上传前校验: 本地 tar 齐全且字节数与 done 标记一致, 过滤前样本数之和等于源 manifest 的 num_samples;
- 上传失败按退避重试; 存储额度类错误 (403 / storage / quota) 直接停止并写明原因, 不删本地文件;
- 进度记录在 {work}/upload_state.json, 重启后跳过已上传的 pack.

用法:
  python tools/refilter_upload.py --src /mnt/fsx/leolxliu/open-audio-data \
      --work /opt/dlami/nvme/leolxliu/open-audio-refilter --names emilia-yodas-zh ...
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger("refilter-upload")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--repo", default="leeoxiang/open-audio-data")
    ap.add_argument("--poll-sec", type=int, default=120)
    ap.add_argument("--squash", action="store_true", help="数据集传完后压缩 repo 历史 (不可逆)")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--keep-local", action="store_true", help="上传后保留本地暂存 tar")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    src, work = Path(args.src), Path(args.work)
    api = HfApi()
    state_path = work / "upload_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"packs": {}, "squashed": []}

    def save():
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.rename(state_path)

    def call(desc: str, fn, *a, **kw):
        """带退避重试的 HF 调用; 存储额度错误不重试, 直接抛出终止."""
        for attempt in range(1, args.retries + 1):
            try:
                return fn(*a, **kw)
            except Exception as e:
                msg = str(e)
                if any(k in msg.lower() for k in ("storage", "quota", "exceeded")) or " 403 " in f" {msg} ":
                    logger.error("%s: storage/permission error, stopping uploader: %s", desc, msg[:500])
                    raise SystemExit(2)
                wait = min(1800, 60 * 2 ** (attempt - 1))
                logger.warning("%s failed (attempt %d/%d): %s; retry in %ds", desc, attempt, args.retries, msg[:300], wait)
                time.sleep(wait)
        raise RuntimeError(f"{desc}: giving up after {args.retries} attempts")

    def manifest_for(name: str) -> dict:
        m = json.loads((src / name / "manifest.json").read_text())
        for pack_id, pack in m["packs"].items():
            info = state["packs"].get(f"{name}/{pack_id}")
            if info:
                pack.update(num_samples=info["num_samples"], size_gb=info["size_gb"])
        m["refilter"] = {
            "multi_speaker": "union of nvidia/diar_streaming_sortformer_4spk-v2 and nvidia/Nemotron-3-Diarization, "
                             "min_speaker_sec=0 (any detected second speaker rejects)",
            "packs_refiltered": sum(1 for k in state["packs"] if k.startswith(name + "/")),
            "packs_total": len(m["packs"]),
        }
        return m

    while True:
        pending = False
        for name in args.names:
            if name in state["squashed"]:
                continue
            m = json.loads((src / name / "manifest.json").read_text())
            all_up = True
            for pack_id, pack in m["packs"].items():
                key = f"{name}/{pack_id}"
                if key in state["packs"]:
                    continue
                all_up = False
                stems = [f.removesuffix(".tar") for f in pack["files"]]
                done = [work / "done" / name / f"{s}.json" for s in stems]
                if not all(d.exists() for d in done):
                    pending = True
                    continue
                stats = [json.loads(d.read_text()) for d in done]
                out_dir = work / "out" / name / pack_id
                bad = [f for f, st in zip(pack["files"], stats)
                       if not (out_dir / f).exists() or (out_dir / f).stat().st_size != st["bytes"]]
                if bad:
                    logger.error("%s: local tar missing or size mismatch: %s; skipping this pack", key, bad[:5])
                    pending = True
                    continue
                if sum(s["n_in"] for s in stats) != pack["num_samples"]:
                    logger.error("%s: n_in %d != source manifest num_samples %d (truncated source?); skipping",
                                 key, sum(s["n_in"] for s in stats), pack["num_samples"])
                    pending = True
                    continue
                size_gb = sum(s["bytes"] for s in stats) / 2**30
                n_in, n_kept = sum(s["n_in"] for s in stats), sum(s["n_kept"] for s in stats)
                logger.info("uploading %s: %d shards, %.1f GB, %d -> %d samples", key, len(stems), size_gb, n_in, n_kept)
                t0 = time.time()
                call(f"upload {key}", api.upload_folder,
                    repo_id=args.repo, repo_type="dataset", folder_path=str(out_dir),
                    path_in_repo=key, allow_patterns=pack["files"],
                    commit_message=f"refilter {key}: multi-speaker union (v2+Nemotron-3), {n_in} -> {n_kept} samples",
                )
                state["packs"][key] = {"num_samples": n_kept, "num_samples_before": n_in,
                                       "size_gb": round(size_gb, 2), "at": time.strftime("%F %T")}
                save()
                call(f"manifest {name}", api.upload_file,
                    repo_id=args.repo, repo_type="dataset",
                    path_or_fileobj=json.dumps(manifest_for(name), indent=2, ensure_ascii=False).encode(),
                    path_in_repo=f"{name}/manifest.json", commit_message=f"update {name} manifest after {pack_id}",
                )
                logger.info("uploaded %s in %.0fs", key, time.time() - t0)
                if not args.keep_local:
                    for f in pack["files"]:
                        (out_dir / f).unlink(missing_ok=True)
            if all_up:
                if args.squash:
                    logger.info("%s fully uploaded; squashing repo history to release replaced LFS files", name)
                    call(f"squash after {name}", api.super_squash_history, repo_id=args.repo, repo_type="dataset",
                         commit_message=f"squash history after refiltering {name}")
                state["squashed"].append(name)
                save()
                logger.info("%s done", name)
        if not pending and all(n in state["squashed"] for n in args.names):
            logger.info("all datasets uploaded")
            return
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
