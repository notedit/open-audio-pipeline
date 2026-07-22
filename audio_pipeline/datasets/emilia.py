"""Emilia-Dataset (amphion/Emilia-Dataset) 加载.

tar 内为 {id}.json + {id}.mp3 成对出现, json 字段:
id / wav / text / duration / speaker / language / dnsmos
"""

from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


def iter_emilia_tar(tar_path: str | Path) -> Iterator[Sample]:
    """按顺序读取一个 Emilia tar 分片, 逐条产出 Sample."""
    pending_json: dict[str, dict] = {}
    pending_audio: dict[str, tuple[bytes, str]] = {}
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name)
            stem, ext = name.stem, name.suffix.lower().lstrip(".")
            data = tf.extractfile(member).read()
            if ext == "json":
                pending_json[stem] = json.loads(data)
            elif ext in ("mp3", "wav", "flac"):
                pending_audio[stem] = (data, ext)
            else:
                continue
            if stem in pending_json and stem in pending_audio:
                info = pending_json.pop(stem)
                audio_bytes, aext = pending_audio.pop(stem)
                yield Sample(
                    key=info.get("id", stem),
                    audio_bytes=audio_bytes,
                    ext=aext,
                    text=info.get("text"),
                    language=info.get("language"),
                    meta={
                        "duration": info.get("duration"),
                        "speaker": info.get("speaker"),
                        "dnsmos": info.get("dnsmos"),
                        "source": "emilia",
                    },
                )
    if pending_json or pending_audio:
        logger.warning(
            "%s: %d unpaired json, %d unpaired audio entries skipped",
            tar_path, len(pending_json), len(pending_audio),
        )
