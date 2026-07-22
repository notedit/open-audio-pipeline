"""Audiobox-Aesthetics 美学评分过滤.

四个维度: PQ(制作质量) CE(内容愉悦度) CU(内容有用度) PC(制作复杂度).
语音清洗主要看 PQ/CE; PC 高通常意味着背景音乐/混杂声源, 可设上限.
分数写入 meta["audiobox"], 阈值不达标即拒绝.
"""

from __future__ import annotations

import logging

import torch

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


class AudioboxFilter:
    name = "audiobox"

    def __init__(
        self,
        min_pq: float | None = 6.5,
        min_ce: float | None = None,
        min_cu: float | None = None,
        max_pc: float | None = None,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.thresholds = {"PQ": ("min", min_pq), "CE": ("min", min_ce), "CU": ("min", min_cu), "PC": ("max", max_pc)}
        self.batch_size = batch_size
        self.device = device
        self._predictor = None

    def _load(self):
        if self._predictor is None:
            from audiobox_aesthetics.infer import initialize_predictor

            logger.info("loading audiobox-aesthetics predictor...")
            self._predictor = initialize_predictor()
            self._predictor.device = self.device
            self._predictor.model.to(self.device)
        return self._predictor

    def process(self, samples: list[Sample]) -> None:
        predictor = self._load()
        for i in range(0, len(samples), self.batch_size):
            chunk = samples[i : i + self.batch_size]
            batch = [
                {"path": torch.from_numpy(s.wav_16k()).unsqueeze(0), "sample_rate": 16000}
                for s in chunk
            ]
            scores = predictor.forward(batch)
            for s, sc in zip(chunk, scores):
                s.meta["audiobox"] = {k: round(v, 3) for k, v in sc.items()}
                for axis, (kind, thr) in self.thresholds.items():
                    if thr is None:
                        continue
                    if kind == "min" and sc[axis] < thr:
                        s.reject(f"audiobox_{axis}:{sc[axis]:.2f}<{thr}")
                        break
                    if kind == "max" and sc[axis] > thr:
                        s.reject(f"audiobox_{axis}:{sc[axis]:.2f}>{thr}")
                        break
