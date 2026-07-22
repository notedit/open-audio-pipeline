"""首尾静音规整: 句首保留 head_silence_sec, 句尾保留 tail_silence_sec.

复用强制对齐时间戳定位语音起止(首个/末个对齐单元), 静音超出目标则裁剪,
不足则补零. 裁剪后音频以 FLAC 无损重编码(原始 mp3/wav 均转 FLAC),
对齐时间戳与时长同步平移, 明细写入 meta["trim"].

必须放在 ForcedAlignStage 之后; 放在 AbnormalPauseFilter 之前时,
原本因首尾静音超标被丢弃的样本会被规整后保留.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)

# 两端偏差都小于该值时不动原音频, 避免无意义的重编码
_TOLERANCE_SEC = 0.05


class EdgeSilenceTrimStage:
    name = "trim"

    def __init__(
        self,
        head_silence_sec: float = 0.1,
        tail_silence_sec: float = 0.3,
        threads: int = 4,
    ):
        self.head = head_silence_sec
        self.tail = tail_silence_sec
        self.threads = threads

    def process(self, samples: list[Sample]) -> None:
        with ThreadPoolExecutor(self.threads) as pool:
            list(pool.map(self._trim_one, samples))

    def _trim_one(self, s: Sample) -> None:
        items = s.meta.get("alignment")
        if not items:
            return
        try:
            wav, sr = sf.read(io.BytesIO(s.audio_bytes), dtype="float32", always_2d=True)
        except Exception as e:
            s.reject(f"trim_decode_error:{type(e).__name__}")
            return
        dur = len(wav) / sr
        speech_start = items[0]["start"]
        speech_end = min(items[-1]["end"], dur)

        cut_start = max(0.0, speech_start - self.head)
        cut_end = min(dur, speech_end + self.tail)
        pad_head = max(0.0, self.head - (speech_start - cut_start))
        pad_tail = max(0.0, self.tail - (cut_end - speech_end))

        if (
            cut_start < _TOLERANCE_SEC
            and dur - cut_end < _TOLERANCE_SEC
            and pad_head < _TOLERANCE_SEC
            and pad_tail < _TOLERANCE_SEC
        ):
            return  # 首尾已符合目标, 保留原始编码

        seg = wav[int(cut_start * sr) : int(cut_end * sr)]
        parts = []
        if pad_head > 0:
            parts.append(np.zeros((int(pad_head * sr), seg.shape[1]), dtype=np.float32))
        parts.append(seg)
        if pad_tail > 0:
            parts.append(np.zeros((int(pad_tail * sr), seg.shape[1]), dtype=np.float32))
        new_wav = np.concatenate(parts) if len(parts) > 1 else seg

        buf = io.BytesIO()
        sf.write(buf, new_wav, sr, format="FLAC", subtype="PCM_16")
        s.audio_bytes = buf.getvalue()
        s.ext = "flac"
        s._wav16k = None  # 缓存失效, 需要时按新音频重解码

        # 新时间轴: t_new = t_old - cut_start + pad_head
        shift = cut_start - pad_head
        for it in items:
            it["start"] = round(max(0.0, it["start"] - shift), 3)
            it["end"] = round(max(0.0, it["end"] - shift), 3)
        new_dur = len(new_wav) / sr
        s._duration = new_dur
        s.meta["duration"] = round(new_dur, 3)
        s.meta["trim"] = {
            "orig_duration": round(dur, 3),
            "cut_head": round(cut_start, 3),
            "cut_tail": round(dur - cut_end, 3),
            "pad_head": round(pad_head, 3),
            "pad_tail": round(pad_tail, 3),
        }
