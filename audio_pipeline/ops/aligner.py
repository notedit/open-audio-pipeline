"""Qwen3-ForcedAligner 强制对齐.

将数据集文本与音频对齐, 得到字/词级时间戳, 写入 meta["alignment"]:
[{"text": ..., "start": ..., "end": ...}, ...]
时间戳供下游停顿检测和训练切分使用. 对齐本身不做过滤.
"""

from __future__ import annotations

import logging

from audio_pipeline.text import language_name
from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


class ForcedAlignStage:
    name = "align"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        batch_size: int = 32,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from qwen_asr import Qwen3ForcedAligner

            logger.info("loading forced aligner %s ...", self.model_name)
            self._model = Qwen3ForcedAligner.from_pretrained(
                self.model_name,
                dtype=getattr(torch, self.dtype),
                device_map=self.device,
            )
        return self._model

    def process(self, samples: list[Sample]) -> None:
        model = self._load()
        for i in range(0, len(samples), self.batch_size):
            chunk = samples[i : i + self.batch_size]
            audio = [(s.wav_16k(), 16000) for s in chunk]
            texts = [s.text or "" for s in chunk]
            langs = [language_name(s.language) or "Chinese" for s in chunk]
            try:
                results = model.align(audio=audio, text=texts, language=langs)
            except Exception as e:  # 单条异常不应拖垮整批
                logger.warning("align batch failed (%s), falling back to per-sample", e)
                results = []
                for a, t, l in zip(audio, texts, langs):
                    try:
                        results.append(model.align(audio=a, text=t, language=l)[0])
                    except Exception as e2:
                        logger.warning("align failed for one sample: %s", e2)
                        results.append(None)
            for s, r in zip(chunk, results):
                if r is None:
                    s.reject("align_failed")
                    continue
                s.meta["alignment"] = [
                    {"text": it.text, "start": round(float(it.start_time), 3), "end": round(float(it.end_time), 3)}
                    for it in r
                ]
