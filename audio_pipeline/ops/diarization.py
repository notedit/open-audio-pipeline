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
