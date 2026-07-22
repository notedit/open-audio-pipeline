"""分片级运行框架: 断点续跑 + 丢弃日志 + 多进程并行.

粒度设计: 一个输入 tar 分片 -> 一个输出 tar(通过样本, WebDataset 格式,
{key}.{ext} + {key}.json) + 一个 drops.jsonl(被拒样本的原因与评分).
输出先写 .tmp 再原子重命名, 完成后落 done 标记; 重跑自动跳过已完成分片.

多进程并行: 启动 N 个进程, 各自带 --worker-id/--num-workers,
第 i 个进程处理 index % N == i 的分片, 配合 CUDA_VISIBLE_DEVICES 各占一卡.
"""

from __future__ import annotations

import json
import logging
import tarfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

from audio_pipeline.pipeline import Pipeline
from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


def write_output_tar(path: Path, samples: list[Sample]) -> None:
    """把通过的样本写成 WebDataset 布局的 tar: {key}.{ext} + {key}.json."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tarfile.open(tmp, "w") as tf:
        for s in samples:
            _add_bytes(tf, f"{s.key}.{s.ext}", s.audio_bytes)
            meta = dict(s.meta)
            meta["text"] = s.text
            meta["language"] = s.language
            _add_bytes(tf, f"{s.key}.json", json.dumps(meta, ensure_ascii=False).encode())
    tmp.rename(path)


def _add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    import io

    tf.addfile(info, io.BytesIO(data))


def _decode_batch(samples: list[Sample], pool: ThreadPoolExecutor) -> None:
    """线程池并行解码一批音频; 解码失败的样本标记拒绝而非中断."""

    def _one(s: Sample) -> None:
        try:
            s.wav_16k()
        except Exception as e:
            s.reject(f"decode_error:{type(e).__name__}")

    list(pool.map(_one, samples))


def _batch_producer(
    shard: Path,
    iter_shard: Callable[[Path], Iterator[Sample]],
    batch_size: int,
    limit_samples: int | None,
    pool: ThreadPoolExecutor,
    q: Queue,
    state: dict,
) -> None:
    """后台线程: 读 tar + 并行解码, 预取好的批次放入队列, 与 GPU 计算重叠."""
    try:
        batch: list[Sample] = []
        for sample in iter_shard(shard):
            state["n_in"] += 1
            batch.append(sample)
            if len(batch) >= batch_size:
                _decode_batch(batch, pool)
                q.put(batch)
                batch = []
            if limit_samples and state["n_in"] >= limit_samples:
                break
        if batch:
            _decode_batch(batch, pool)
            q.put(batch)
    except Exception as e:
        state["error"] = e
    finally:
        q.put(None)


def run_shards(
    shard_paths: list[Path],
    iter_shard: Callable[[Path], Iterator[Sample]],
    make_pipeline: Callable[[], Pipeline],
    output_dir: Path,
    batch_size: int = 64,
    worker_id: int = 0,
    num_workers: int = 1,
    limit_samples: int | None = None,
    decode_threads: int = 8,
    prefetch_batches: int = 2,
) -> None:
    """主循环: 遍历属于本 worker 的分片, 逐分片处理并写出."""
    output_dir.mkdir(parents=True, exist_ok=True)
    done_dir = output_dir / ".done"
    done_dir.mkdir(exist_ok=True)

    my_shards = [p for i, p in enumerate(sorted(shard_paths)) if i % num_workers == worker_id]
    logger.info("worker %d/%d: %d shards to process", worker_id, num_workers, len(my_shards))

    # pipeline(及其中的模型)整个 worker 生命周期只创建一次, 跨分片复用
    pipeline = make_pipeline()
    decode_pool = ThreadPoolExecutor(decode_threads)

    for shard in my_shards:
        stem = shard.name.removesuffix(".tar.gz").removesuffix(".tar")
        done_marker = done_dir / f"{stem}.done"
        if done_marker.exists():
            logger.info("skip %s (done)", shard.name)
            continue

        t0 = time.time()
        out_tar = output_dir / f"{stem}-clean.tar"
        drops_path = output_dir / f"{stem}-drops.jsonl"
        passed: list[Sample] = []

        with open(drops_path, "w", encoding="utf-8") as drops_f:

            def on_drop(s: Sample, stage: str) -> None:
                meta = dict(s.meta)
                meta["text"] = s.text
                meta["language"] = s.language
                drops_f.write(
                    json.dumps(
                        {"key": s.key, "stage": stage, "reason": s.reject_reason, "meta": meta},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            pipeline.on_drop = on_drop
            # 后台线程读 tar + 并行解码, 与本线程的 GPU 计算重叠
            q: Queue = Queue(maxsize=prefetch_batches)
            state = {"n_in": 0, "error": None}
            producer = threading.Thread(
                target=_batch_producer,
                args=(shard, iter_shard, batch_size, limit_samples, decode_pool, q, state),
                daemon=True,
            )
            producer.start()
            while (batch := q.get()) is not None:
                passed.extend(pipeline.run(batch))
            producer.join()
            if state["error"] is not None:
                raise state["error"]
            n_in = state["n_in"]

        write_output_tar(out_tar, passed)
        for s in passed:
            s.free_wav()
        done_marker.touch()
        logger.info(
            "%s: %d -> %d kept (%.1f%%) in %.0fs | %s",
            shard.name, n_in, len(passed), 100 * len(passed) / max(n_in, 1),
            time.time() - t0, pipeline.format_stats(),
        )
        logger.info("timing | %s", pipeline.format_timing())

    logger.info("worker %d done. totals: %s", worker_id, pipeline.format_stats())
    try:
        import torch

        if torch.cuda.is_available():
            logger.info(
                "gpu peak memory: allocated %.1f GB / reserved %.1f GB",
                torch.cuda.max_memory_allocated() / 2**30,
                torch.cuda.max_memory_reserved() / 2**30,
            )
    except Exception:
        pass


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # NeMo 日志非常吵
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
