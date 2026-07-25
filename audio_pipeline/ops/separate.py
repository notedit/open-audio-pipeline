"""人声分离 (demucs htdemucs, 提取 vocals 声部).

用于 BGM/配乐较重的原始源(影视/综艺), 放在降噪之前:
分离 -> 降噪 -> 后续过滤. 分离后的音频替换原始字节(FLAC),
meta["separated"]=True. 默认关闭, 按源特性在 config 开启.

注意: 对已较干净的语音, 分离会引入轻微伪影(实测 PQ -0.04),
只在 BGM 可闻的源上启用.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


class SeparateStage:
    name = "separate"

    def __init__(
        self,
        device: str = "cuda",
        model: str = "htdemucs",
        stem: str = "vocals",
        skip_if_clean: bool = True,
        bg_energy_ratio_thr: float = 0.12,
    ):
        self.device = device
        self.model = model
        self.stem = stem
        self.skip_if_clean = skip_if_clean
        self.bg_thr = bg_energy_ratio_thr
        self._sep = None
        self._vad = None

    def bg_energy_ratio(self, mono16k: np.ndarray) -> float | None:
        """VAD 间隙能量比: 非语音区 RMS / 语音区 RMS.
        干净语音间隙近静音(<0.1), 带 BGM 间隙音乐仍响(>0.15).
        间隙总长 <0.5s 无法判断, 返回 None."""
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self._vad is None:
            self._vad = load_silero_vad()
        ts = get_speech_timestamps(torch.from_numpy(mono16k), self._vad, sampling_rate=16000)
        mask = np.zeros(len(mono16k), dtype=bool)
        pad = int(0.15 * 16000)  # 掩码外扩 150ms, 剔除 VAD 边界渗漏的语音能量
        for t_ in ts:
            mask[max(0, t_["start"] - pad) : t_["end"] + pad] = True
        gap = mono16k[~mask]
        if len(gap) < 8000 or mask.sum() < 8000:
            return None
        sp_rms = float(np.sqrt((mono16k[mask] ** 2).mean()) + 1e-9)
        return float(np.sqrt((gap ** 2).mean())) / sp_rms

    def maybe_separate(self, mono: np.ndarray, sr: int) -> tuple[np.ndarray, bool]:
        """干净音频跳过分离; 返回 (波形, 是否执行了分离)."""
        if self.skip_if_clean:
            import librosa

            m16 = librosa.resample(mono, orig_sr=sr, target_sr=16000) if sr != 16000 else mono
            r = self.bg_energy_ratio(m16.astype(np.float32))
            # 无法判断(间隙不足)时保守跳过: 没有 BGM 证据不引入分离伪影
            if r is None or r < self.bg_thr:
                return mono, False
        return self.separate_wave(mono, sr), True

    def _load(self):
        if self._sep is None:
            import demucs.api

            logger.info("loading demucs %s ...", self.model)
            self._sep = demucs.api.Separator(model=self.model, device=self.device, progress=False)
        return self._sep

    def separate_wave(self, mono: np.ndarray, sr: int) -> np.ndarray:
        """输入任意采样率单声道, 返回同采样率的人声声部."""
        import librosa
        import torch

        sep = self._load()
        w44 = librosa.resample(mono, orig_sr=sr, target_sr=44100) if sr != 44100 else mono
        x = torch.from_numpy(np.stack([w44, w44])).float()
        _, stems = sep.separate_tensor(x, sr=44100)
        v = stems[self.stem].mean(0).cpu().numpy()
        if sr != 44100:
            v = librosa.resample(v, orig_sr=44100, target_sr=sr)
        return np.clip(v, -1, 1).astype(np.float32)

    def process(self, samples: list[Sample]) -> None:
        import soundfile as sf

        self._load()
        for s in samples:
            try:
                wav, sr = sf.read(io.BytesIO(s.audio_bytes), dtype="float32", always_2d=True)
                v, did = self.maybe_separate(wav.mean(axis=1), sr)
                if did:
                    buf = io.BytesIO()
                    sf.write(buf, v, sr, format="FLAC", subtype="PCM_16")
                    s.audio_bytes = buf.getvalue()
                    s.ext = "flac"
                    s.free_wav()
                s.meta["separated"] = did
            except Exception as e:
                logger.warning("separate failed for %s: %s", s.key, e)
                s.meta["separated"] = False
