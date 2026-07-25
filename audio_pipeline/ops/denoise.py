"""语音增强(降噪)与 VAD.

DenoiseStage: DeepFilterNet3 降噪, 增强后的音频替换原始字节(FLAC 重编码),
  meta["denoised"]=true; 用于噪声源预处理或对被拒样本的救回再处理.
VADFilter: silero-vad 语音占比检测, 低于阈值拒绝(纯音乐/静音/环境声).
  speech_ratio 写入 meta.

两者默认关闭, 面向噪声源数据集在 config 中按需开启.
"""

from __future__ import annotations

import io
import logging

import numpy as np

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


def _shim_torchaudio_backend() -> None:
    """torchaudio>=2.9 移除 backend 子模块; df 仅 import 时引用 AudioMetaData."""
    import sys
    import types

    if "torchaudio.backend" not in sys.modules:
        m = types.ModuleType("torchaudio.backend")
        c = types.ModuleType("torchaudio.backend.common")
        c.AudioMetaData = type("AudioMetaData", (), {})
        m.common = c
        sys.modules["torchaudio.backend"] = m
        sys.modules["torchaudio.backend.common"] = c


class DenoiseStage:
    """backend: zipenhancer (默认, 实测降噪还原能力与速度均显著优于 df3) | deepfilternet"""

    name = "denoise"

    def __init__(self, device: str = "cuda", backend: str = "zipenhancer", post_gain_db: float = 0.0):
        self.device = device
        self.backend = backend
        self.post_gain_db = post_gain_db
        self._model = None

    def _load(self):
        if self._model is None:
            if self.backend == "zipenhancer":
                from modelscope.pipelines import pipeline as ms_pipeline
                from modelscope.utils.constant import Tasks

                logger.info("loading ZipEnhancer ...")
                self._model = ms_pipeline(
                    Tasks.acoustic_noise_suppression,
                    model="iic/speech_zipenhancer_ans_multiloss_16k_base",
                    device=self.device,
                )
            else:
                _shim_torchaudio_backend()
                from df.enhance import init_df

                logger.info("loading DeepFilterNet3 ...")
                model, df_state, _ = init_df()
                model = model.to(self.device).eval()
                self._model = (model, df_state)
        return self._model

    def _enhance_zip(self, s):
        import tempfile, os
        import librosa
        import numpy as np
        import soundfile as sf
        import io as _io

        wav, sr = sf.read(_io.BytesIO(s.audio_bytes), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        w16 = librosa.resample(mono, orig_sr=sr, target_sr=16000) if sr != 16000 else mono
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, w16, 16000)
            path = f.name
        try:
            r = self._model(path)
        finally:
            os.unlink(path)
        enh = np.frombuffer(r["output_pcm"], dtype=np.int16).astype(np.float32) / 32768
        buf = _io.BytesIO()
        sf.write(buf, enh, 16000, format="FLAC", subtype="PCM_16")
        s.audio_bytes = buf.getvalue()
        s.ext = "flac"
        s.free_wav()
        s.meta["denoised"] = True

    def process(self, samples: list[Sample]) -> None:
        import soundfile as sf
        import torch
        import torchaudio.functional as AF

        loaded = self._load()
        if self.backend == "zipenhancer":
            for s in samples:
                try:
                    self._enhance_zip(s)
                except Exception as e:
                    logger.warning("denoise failed for %s: %s", s.key, e)
                    s.meta["denoised"] = False
            return
        _shim_torchaudio_backend()
        from df.enhance import enhance

        model, df_state = loaded
        df_sr = df_state.sr()
        for s in samples:
            try:
                wav, sr = sf.read(io.BytesIO(s.audio_bytes), dtype="float32", always_2d=True)
                mono = torch.from_numpy(wav.mean(axis=1))
                x = AF.resample(mono, sr, df_sr).unsqueeze(0)
                with torch.inference_mode():
                    out = enhance(model, df_state, x)
                out = AF.resample(out.squeeze(0), df_sr, sr).clamp(-1, 1)
                buf = io.BytesIO()
                sf.write(buf, out.numpy(), sr, format="FLAC", subtype="PCM_16")
                s.audio_bytes = buf.getvalue()
                s.ext = "flac"
                s.free_wav()  # 16k 缓存失效, 下游按增强后音频重解码
                s.meta["denoised"] = True
            except Exception as e:
                logger.warning("denoise failed for %s: %s", s.key, e)
                s.meta["denoised"] = False


class VADFilter:
    name = "vad"

    def __init__(self, min_speech_ratio: float = 0.4, device: str = "cpu"):
        self.min_speech_ratio = min_speech_ratio
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()
        return self._model

    def process(self, samples: list[Sample]) -> None:
        import torch

        from silero_vad import get_speech_timestamps

        model = self._load()
        for s in samples:
            wav = torch.from_numpy(s.wav_16k())
            ts = get_speech_timestamps(wav, model, sampling_rate=16000)
            speech = sum(t["end"] - t["start"] for t in ts) / 16000.0
            ratio = speech / max(s.duration(), 1e-6)
            s.meta["speech_ratio"] = round(ratio, 3)
            if ratio < self.min_speech_ratio:
                s.reject(f"vad_low_speech:{ratio:.2f}")
