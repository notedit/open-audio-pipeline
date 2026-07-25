"""多说话人检测: nvidia/diar_streaming_sortformer_4spk-v2 (NeMo).

对每条样本跑说话人分离, 统计有效说话人数(累计说话时长超过阈值才算),
超过 max_speakers 即拒绝. 说话人数与各自时长写入 meta["diarization"].
"""

from __future__ import annotations

import logging
import re

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)

_SEG_RE = re.compile(r"^([\d.]+)\s+([\d.]+)\s+(\S+)$")


class DiarizenSpeakerFilter:
    """多说话人检测的第二意见: BUT-FIT DiariZen (WavLM+EEND, 与 Sortformer 架构独立).

    接口与 MultiSpeakerFilter 一致, 可在 config 中互换或串联做交叉验证.
    结果写入 meta["diarization_diarizen"].
    """

    name = "multi_speaker_diarizen"

    def __init__(
        self,
        model_name: str = "BUT-FIT/diarizen-wavlm-large-s80-md-v2",
        max_speakers: int = 1,
        min_speaker_sec: float = 0.5,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_speakers = max_speakers
        self.min_speaker_sec = min_speaker_sec
        self.device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            import torchaudio

            # torchaudio>=2.9 移除了 AudioMetaData/info/list_audio_backends;
            # pyannote 3.x 依赖它们, 用 soundfile 垫回等价实现
            if not hasattr(torchaudio, "AudioMetaData"):
                torchaudio.AudioMetaData = type("AudioMetaData", (), {})
            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]
            if not hasattr(torchaudio, "set_audio_backend"):
                torchaudio.set_audio_backend = lambda *a, **k: None
            import numpy as _np

            for _alias in ("NaN", "NAN"):  # numpy>=2 移除的别名, 老 pyannote fork 还在用
                if not hasattr(_np, _alias):
                    setattr(_np, _alias, _np.nan)
            # torch>=2.6 默认 weights_only 加载, 放行 BUT-FIT 官方权重里的元数据类
            from pyannote.audio.core.task import Problem, Resolution, Specifications

            torch.serialization.add_safe_globals(
                [torch.torch_version.TorchVersion, Specifications, Problem, Resolution]
            )
            # torchaudio 2.9 的 load 走 torchcodec(与当前 torch 组合缺 so), 换 soundfile 实现
            import soundfile as _sf

            def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                         channels_first=True, format=None, buffer_size=4096, backend=None):
                data, sr = _sf.read(uri, dtype="float32", always_2d=True)
                t = torch.from_numpy(data.T if channels_first else data)
                return t, sr

            torchaudio.load = _sf_load
            if not hasattr(torchaudio, "info"):

                def _info(path, backend=None):
                    i = _sf.info(str(path))
                    md = torchaudio.AudioMetaData()
                    md.sample_rate = i.samplerate
                    md.num_frames = i.frames
                    md.num_channels = i.channels
                    md.bits_per_sample = 16
                    md.encoding = i.subtype
                    return md

                torchaudio.info = _info
            from diarizen.pipelines.inference import DiariZenPipeline

            logger.info("loading diarizen model %s ...", self.model_name)
            self._pipe = DiariZenPipeline.from_pretrained(self.model_name)
            self._pipe.to(torch.device(self.device))
        return self._pipe

    def process(self, samples: list[Sample]) -> None:
        import io

        pipe = self._load()
        for s in samples:
            ann = pipe(io.BytesIO(s.audio_bytes))
            spk_sec: dict[str, float] = {}
            for turn, _, spk in ann.itertracks(yield_label=True):
                spk_sec[str(spk)] = spk_sec.get(str(spk), 0.0) + (turn.end - turn.start)
            active = {k: round(v, 2) for k, v in spk_sec.items() if v >= self.min_speaker_sec}
            n = len(active)
            s.meta["diarization_diarizen"] = {"num_speakers": n, "speaker_seconds": active}
            if n > self.max_speakers:
                s.reject(f"multi_speaker_diarizen:{n}")


class MultiSpeakerFilter:
    name = "multi_speaker"

    def __init__(
        self,
        model_name: str = "nvidia/diar_streaming_sortformer_4spk-v2",
        max_speakers: int = 1,
        min_speaker_sec: float = 0.5,
        batch_size: int = 16,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.max_speakers = max_speakers
        self.min_speaker_sec = min_speaker_sec
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is None:
            from nemo.collections.asr.models import SortformerEncLabelModel

            logger.info("loading sortformer diarization model %s ...", self.model_name)
            model = SortformerEncLabelModel.from_pretrained(self.model_name, map_location=self.device)
            model.eval()
            self._model = model
        return self._model

    def process(self, samples: list[Sample]) -> None:
        model = self._load()
        audio = [s.wav_16k() for s in samples]
        outputs = model.diarize(audio=audio, sample_rate=16000, batch_size=self.batch_size, verbose=False)
        for s, segs in zip(samples, outputs):
            spk_sec: dict[str, float] = {}
            for seg in segs or []:
                m = _SEG_RE.match(seg.strip())
                if not m:
                    continue
                start, end, spk = float(m.group(1)), float(m.group(2)), m.group(3)
                spk_sec[spk] = spk_sec.get(spk, 0.0) + (end - start)
            active = {k: round(v, 2) for k, v in spk_sec.items() if v >= self.min_speaker_sec}
            n = len(active)
            s.meta["diarization"] = {"num_speakers": n, "speaker_seconds": active}
            if n > self.max_speakers:
                s.reject(f"multi_speaker:{n}")
