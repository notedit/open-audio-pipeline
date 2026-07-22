"""WenetSpeech4TTS (Wenetspeech4TTS/WenetSpeech4TTS) 加载.

每个 WenetSpeech4TTS_{subset}_{n}.tar.gz 内为 wavs/{utt}.wav + txts/{utt}.txt,
txt 首行: "<utt_id>\t<text>", 次行为词级时间戳(忽略, pipeline 自己重新对齐).
音频 16kHz 单声道 16bit PCM, 语言为中文.
"""

from __future__ import annotations

import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


def _parse_txt(data: bytes) -> str | None:
    line = data.decode("utf-8", errors="replace").splitlines()
    if not line:
        return None
    parts = line[0].split("\t", 1)
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def iter_wenetspeech4tts_tar(tar_path: str | Path) -> Iterator[Sample]:
    """读取一个 WenetSpeech4TTS tar.gz 分片, 逐条产出 Sample.

    gzip tar 无法随机访问, 顺序扫描并配对 wav/txt.
    """
    pending_text: dict[str, str | None] = {}
    pending_audio: dict[str, bytes] = {}
    mode = "r:gz" if str(tar_path).endswith(".gz") else "r"
    with tarfile.open(tar_path, mode) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name)
            stem, ext = name.stem, name.suffix.lower().lstrip(".")
            if ext == "txt":
                pending_text[stem] = _parse_txt(tf.extractfile(member).read())
            elif ext == "wav":
                pending_audio[stem] = tf.extractfile(member).read()
            else:
                continue
            if stem in pending_text and stem in pending_audio:
                yield Sample(
                    key=stem,
                    audio_bytes=pending_audio.pop(stem),
                    ext="wav",
                    text=pending_text.pop(stem),
                    language="zh",
                    meta={"source": "wenetspeech4tts"},
                )
    if pending_text or pending_audio:
        logger.warning(
            "%s: %d unpaired txt, %d unpaired wav entries skipped",
            tar_path, len(pending_text), len(pending_audio),
        )
