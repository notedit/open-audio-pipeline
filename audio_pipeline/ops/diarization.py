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
        min_speaker_sec: float = 0.0,
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
    """Sortformer 系说话人分离 (NeMo SortformerEncLabelModel), 支持多模型并集判定.

    model_name 可为单个模型或模型列表 (HF 名或本地 .nemo 路径):
    - nvidia/diar_streaming_sortformer_4spk-v2 (最多 4 人)
    - nvidia/Nemotron-3-Diarization (最多 8 人, 需 NeMo >= 3.1)
    列表模式下每个模型都跑全部样本, 任一模型判出说话人数 > max_speakers 即拒绝.
    (2026-11 试听标注: Nemotron-3 会漏判 v2 能抓到的短附和 "嗯/对"; 两者判出的集合重合度低, 故取并集.)

    meta["diarization"]:
      num_speakers     各模型说话人数的最大值
      speaker_seconds  判出最多说话人的那个模型的分段时长
      models           {模型短名: {"num_speakers", "speaker_seconds"}}
      flagged_by       判多人的模型短名列表
    """

    name = "multi_speaker"

    def __init__(
        self,
        model_name: str | list[str] = "nvidia/diar_streaming_sortformer_4spk-v2",
        max_speakers: int = 1,
        min_speaker_sec: float = 0.0,
        batch_size: int = 16,
        device: str = "cuda",
    ):
        self.model_names = [model_name] if isinstance(model_name, str) else list(model_name)
        if not self.model_names:
            raise ValueError("diarization model list is empty")
        self.max_speakers = max_speakers
        self.min_speaker_sec = min_speaker_sec
        self.batch_size = batch_size
        self.device = device
        self._models: list | None = None

    @staticmethod
    def _short(name: str) -> str:
        return name.rsplit("/", 1)[-1].removesuffix(".nemo")

    def _load(self):
        if self._models is None:
            from nemo.collections.asr.models import SortformerEncLabelModel
            from nemo.collections.asr.models import sortformer_diar_models as _sdm

            # NeMo >= 3.1 的流式推理进度条只认 disable=self.training, 不受 verbose=False 控制,
            # 每批都往 worker 日志里刷一行; 替换模块内的 tqdm 引用强制关闭
            if hasattr(_sdm, "tqdm") and not getattr(_sdm.tqdm, "_quiet", False):
                _orig_tqdm = _sdm.tqdm

                def _quiet_tqdm(*args, **kwargs):
                    kwargs["disable"] = True
                    return _orig_tqdm(*args, **kwargs)

                _quiet_tqdm._quiet = True
                _sdm.tqdm = _quiet_tqdm

            models = []
            for name in self.model_names:
                logger.info("loading sortformer diarization model %s ...", name)
                if name.endswith(".nemo"):
                    model = SortformerEncLabelModel.restore_from(name, map_location=self.device)
                else:
                    model = SortformerEncLabelModel.from_pretrained(name, map_location=self.device)
                model.eval()
                models.append(model)
            self._models = models
        return self._models

    def _speaker_seconds(self, segs) -> dict[str, float]:
        spk_sec: dict[str, float] = {}
        for seg in segs or []:
            m = _SEG_RE.match(seg.strip())
            if not m:
                continue
            start, end, spk = float(m.group(1)), float(m.group(2)), m.group(3)
            spk_sec[spk] = spk_sec.get(spk, 0.0) + (end - start)
        return {k: round(v, 2) for k, v in spk_sec.items() if v >= self.min_speaker_sec}

    def process(self, samples: list[Sample]) -> None:
        models = self._load()
        audio = [s.wav_16k() for s in samples]
        per_model = [
            model.diarize(audio=audio, sample_rate=16000, batch_size=self.batch_size, verbose=False)
            for model in models
        ]
        for i, s in enumerate(samples):
            results = {}
            for name, outputs in zip(self.model_names, per_model):
                active = self._speaker_seconds(outputs[i])
                results[self._short(name)] = {"num_speakers": len(active), "speaker_seconds": active}
            top = max(results, key=lambda k: results[k]["num_speakers"])  # 平局取列表靠前的
            flagged = [k for k, r in results.items() if r["num_speakers"] > self.max_speakers]
            n = results[top]["num_speakers"]
            s.meta["diarization"] = {
                "num_speakers": n,
                "speaker_seconds": results[top]["speaker_seconds"],
                "models": results,
                "flagged_by": flagged,
            }
            if flagged:
                s.reject(f"multi_speaker:{n}@{','.join(flagged)}")
