"""核心数据类型: 一条待处理的音频样本."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf


@dataclass
class Sample:
    """pipeline 中流转的一条样本.

    audio_bytes 保存原始编码字节 (mp3/wav/flac), 输出时原样写回, 避免重复转码.
    需要波形的算子通过 wav_16k() 拿到解码后的 16k 单声道 float32, 解码结果缓存.
    """

    key: str
    audio_bytes: bytes
    ext: str  # "mp3" / "wav" / "flac"
    text: str | None = None
    language: str | None = None  # "zh" / "en" / ...
    meta: dict = field(default_factory=dict)  # 各 stage 累积的标注
    reject_reason: str | None = None

    _wav16k: np.ndarray | None = field(default=None, repr=False, compare=False)
    _duration: float | None = field(default=None, repr=False, compare=False)

    @property
    def rejected(self) -> bool:
        return self.reject_reason is not None

    def reject(self, reason: str) -> None:
        if self.reject_reason is None:
            self.reject_reason = reason

    def wav_16k(self) -> np.ndarray:
        """解码为 16kHz 单声道 float32, 结果缓存."""
        if self._wav16k is None:
            wav, sr = sf.read(io.BytesIO(self.audio_bytes), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != 16000:
                import librosa

                wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            self._wav16k = np.ascontiguousarray(wav)
            self._duration = len(self._wav16k) / 16000.0
        return self._wav16k

    def duration(self) -> float:
        """时长(秒). 优先用 meta 里的时长字段, 否则解码计算."""
        if self._duration is None:
            d = self.meta.get("duration")
            if d is not None:
                self._duration = float(d)
            else:
                self.wav_16k()
        return self._duration

    def free_wav(self) -> None:
        """释放解码缓存, 控制内存."""
        self._wav16k = None
