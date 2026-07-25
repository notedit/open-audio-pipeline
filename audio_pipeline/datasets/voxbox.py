"""VoxBox (SparkAudio/voxbox) 加载.

结构: audios/{subset}/{subset}_NNNN.tar.gz 内平铺 {index}.wav;
文本与属性在独立的 metadata/{subset}.jsonl, 按 index 关联.
字段: index/text/language/gender/emotion/pitch/speed/duration/...
情感与韵律标签是该数据集的独特价值, 原样保留进 meta.
"""

from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)

# 透传到输出 meta 的属性字段
_KEEP_FIELDS = ("gender", "age", "emotion", "pitch", "pitch_std", "speed", "syllable_num", "split")


def load_voxbox_metadata(jsonl_path: str | Path) -> dict[str, dict]:
    """metadata/{subset}.jsonl -> {index: record}"""
    meta: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                meta[r["index"]] = r
    return meta


def iter_voxbox_tar(tar_path: str | Path, metadata: dict[str, dict]) -> Iterator[Sample]:
    """读取一个 voxbox 音频分片, 关联子集元数据, 逐条产出 Sample."""
    subset = Path(tar_path).name.rsplit("_", 1)[0]
    missing = 0
    mode = "r:gz" if str(tar_path).endswith(".gz") else "r"
    with tarfile.open(tar_path, mode) as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = Path(m.name)
            if name.suffix.lower().lstrip(".") not in ("wav", "flac", "mp3"):
                continue
            r = metadata.get(name.stem)
            if r is None:
                missing += 1
                continue
            extra = {k: r[k] for k in _KEEP_FIELDS if k in r}
            yield Sample(
                key=name.stem,
                audio_bytes=tf.extractfile(m).read(),
                ext=name.suffix.lower().lstrip("."),
                text=r.get("text"),
                language=(r.get("language") or "zh").lower(),
                meta={"duration": r.get("duration"), "source": "voxbox", "subset": subset, **extra},
            )
    if missing:
        logger.warning("%s: %d 条音频无元数据记录, 已跳过", tar_path, missing)
