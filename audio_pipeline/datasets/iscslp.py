"""ISCSLP2026-CoT-TTS (HKUSTAudio) 加载 — 仅 dialogue_segments(句级).

结构: movie-*/dialogue_segments/*.tar 内平铺 {segment_id}.flac;
     movie-*/movie-*.json 为整目录元数据(list), 按 target_segment.segment_id 关联.
保留字段: 说话人/情感标签/表现力强度/响度(该数据集的韵律价值所在).
注意: continuous_segments 是同一音频的上下文切法, 与 dialogue 重复, 不处理.
许可: 非商用且禁止再分发 —— 本数据集输出仅限本地使用, 不进入 HF 公开上传.
"""

from __future__ import annotations

import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


def load_iscslp_metadata(json_path: str | Path) -> dict[str, dict]:
    """movie-*.json -> {segment_id: 展平记录}"""
    meta: dict[str, dict] = {}
    for r in json.load(open(json_path, encoding="utf-8")):
        t = r.get("target_segment") or {}
        sid = t.get("segment_id")
        if not sid:
            continue
        feats = t.get("features") or {}
        meta[sid] = {
            "text": t.get("text"),
            "speaker": f'{r.get("movie_id")}#{t.get("normalized_speaker") or t.get("speaker")}',
            "emotion": t.get("emotion_tag"),
            "expressive_intensity": _f(feats.get("expressive_intensity")),
            "loudness_dbfs": _f(str(feats.get("loudness", "")).replace("dBFS", "")),
            "duration": _f(str(feats.get("duration", "")).replace("s", "")),
            "movie_id": r.get("movie_id"),
        }
    return meta


def _f(v):
    try:
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def iter_iscslp_tar(tar_path: str | Path, metadata: dict[str, dict]) -> Iterator[Sample]:
    subset = Path(tar_path).name.split(".")[0]  # movie-zh1
    missing = 0
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".flac")):
                continue
            sid = Path(m.name).stem
            r = metadata.get(sid)
            if r is None:
                missing += 1
                continue
            yield Sample(
                key=sid,
                audio_bytes=tf.extractfile(m).read(),
                ext="flac",
                text=r.get("text"),
                language="zh" if "zh" in subset else "en",
                meta={
                    "duration": r.get("duration"),
                    "source": "iscslp-cot-tts",
                    "subset": subset,
                    "speaker": r.get("speaker"),
                    "emotion": r.get("emotion"),
                    "expressive_intensity": r.get("expressive_intensity"),
                    "loudness_dbfs": r.get("loudness_dbfs"),
                },
            )
    if missing:
        logger.warning("%s: %d 条音频无元数据, 已跳过", tar_path, missing)
