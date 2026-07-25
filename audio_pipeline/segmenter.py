"""原始长音频 -> 干净单说话人 3-30s 段落.

约束(构建即保证 + 落盘前复验):
- 段内无非人声(降噪后按 VAD 语音区切割, 复验 speech_ratio >= min_speech_ratio)
- 段内单说话人(按说话人回合切割, 变更点必切)
- 3s <= 段长 <= 30s(短段丢弃, 长段在最长内部停顿处递归二分)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    speech_ratio: float = 1.0

    @property
    def dur(self) -> float:
        return self.end - self.start


class RawSegmenter:
    def __init__(
        self,
        device: str = "cuda",
        min_sec: float = 3.0,
        max_sec: float = 30.0,
        join_gap_sec: float = 0.6,
        min_speech_ratio: float = 0.85,
        denoise: bool = True,
        separate: bool = False,
    ):
        self.device = device
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.join_gap = join_gap_sec
        self.min_speech_ratio = min_speech_ratio
        self.denoise = denoise
        self.separate_first = separate
        self._separator = None
        self._denoiser = None
        self._diar = None
        self._vad = None

    # ---- 模型 ----
    def _load(self):
        if self._diar is None:
            from audio_pipeline.ops.denoise import DenoiseStage
            from nemo.collections.asr.models import SortformerEncLabelModel
            from silero_vad import load_silero_vad

            if self.separate_first:
                from audio_pipeline.ops.separate import SeparateStage
                self._separator = SeparateStage(device=self.device)
                self._separator._load()
            if self.denoise:
                self._denoiser = DenoiseStage(device=self.device)
                self._denoiser._load()
            m = SortformerEncLabelModel.from_pretrained(
                "nvidia/diar_streaming_sortformer_4spk-v2", map_location=self.device
            )
            m.eval()
            self._diar = m
            self._vad = load_silero_vad()

    # ---- 主流程 ----
    def segment(self, wav16k: np.ndarray) -> tuple[np.ndarray, list[Segment]]:
        """输入 16k 单声道波形, 返回(可能被降噪替换的)波形与合规段落列表."""
        import re
        import torch

        from silero_vad import get_speech_timestamps

        self._load()
        if self.separate_first and self._separator is not None:
            wav16k, _did_sep = self._separator.maybe_separate(wav16k, SR)
        if self.denoise and self._denoiser is not None:
            wav16k = self._denoise_wave(wav16k)

        # 说话人回合
        outputs = self._diar.diarize(audio=[wav16k], sample_rate=SR, batch_size=1, verbose=False)
        seg_re = re.compile(r"^([\d.]+)\s+([\d.]+)\s+(\S+)$")
        turns = []
        for seg in outputs[0] or []:
            m = seg_re.match(seg.strip())
            if m:
                turns.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        # VAD 语音区
        ts = get_speech_timestamps(torch.from_numpy(wav16k), self._vad, sampling_rate=SR)
        speech = [(t["start"] / SR, t["end"] / SR) for t in ts]

        # 说话人回合 ∩ 语音区 -> 原子段
        atoms: list[Segment] = []
        for s0, s1, spk in turns:
            for v0, v1 in speech:
                a, b = max(s0, v0), min(s1, v1)
                if b - a > 0.2:
                    atoms.append(Segment(a, b, spk))
        atoms.sort(key=lambda x: x.start)

        # 同说话人相邻合并
        merged: list[Segment] = []
        for a in atoms:
            if (
                merged
                and merged[-1].speaker == a.speaker
                and a.start - merged[-1].end <= self.join_gap
                and a.end - merged[-1].start <= self.max_sec
            ):
                merged[-1].end = a.end
            else:
                merged.append(Segment(a.start, a.end, a.speaker))

        # 长段二分, 短段丢弃
        final: list[Segment] = []
        for seg in merged:
            final.extend(self._split_long(seg, wav16k))
        final = [s for s in final if s.dur >= self.min_sec]

        # 复验 1: 纯人声占比
        kept = []
        for seg in final:
            piece = wav16k[int(seg.start * SR) : int(seg.end * SR)]
            ts = get_speech_timestamps(torch.from_numpy(piece), self._vad, sampling_rate=SR)
            ratio = sum(t["end"] - t["start"] for t in ts) / SR / max(seg.dur, 1e-6)
            seg.speech_ratio = round(ratio, 3)
            if ratio >= self.min_speech_ratio:
                kept.append(seg)

        # 复验 2: 单说话人终审(段落级重跑 diar, 防切分粒度导致的漏网)
        if kept:
            pieces = [wav16k[int(s.start * SR) : int(s.end * SR)] for s in kept]
            outs = self._diar.diarize(audio=pieces, sample_rate=SR, batch_size=8, verbose=False)
            verified = []
            for seg, segs_out in zip(kept, outs):
                spk_sec: dict[str, float] = {}
                for line in segs_out or []:
                    m = seg_re.match(line.strip())
                    if m:
                        spk_sec[m.group(3)] = spk_sec.get(m.group(3), 0.0) + float(m.group(2)) - float(m.group(1))
                if sum(1 for v in spk_sec.values() if v >= 0.5) <= 1:
                    verified.append(seg)
            kept = verified
        return wav16k, kept

    def _denoise_wave(self, wav16k: np.ndarray) -> np.ndarray:
        import io

        import soundfile as sf

        from audio_pipeline.types import Sample

        buf = io.BytesIO()
        sf.write(buf, wav16k, SR, format="WAV")
        s = Sample(key="raw", audio_bytes=buf.getvalue(), ext="wav")
        self._denoiser.process([s])
        return s.wav_16k()

    def _split_long(self, seg: Segment, wav: np.ndarray) -> list[Segment]:
        """超长段在最长内部停顿(能量谷)处二分, 递归直至 <= max_sec."""
        if seg.dur <= self.max_sec:
            return [seg]
        piece = wav[int(seg.start * SR) : int(seg.end * SR)]
        # 100ms 窗能量, 在中间 60% 区域找最小能量点作为切点
        win = SR // 10
        n = len(piece) // win
        e = np.array([float((piece[i * win : (i + 1) * win] ** 2).mean()) for i in range(n)])
        lo, hi = int(n * 0.2), int(n * 0.8)
        cut_idx = lo + int(np.argmin(e[lo:hi]))
        cut = seg.start + cut_idx * win / SR
        left = Segment(seg.start, cut, seg.speaker)
        right = Segment(cut, seg.end, seg.speaker)
        return self._split_long(left, wav) + self._split_long(right, wav)
