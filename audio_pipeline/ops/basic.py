"""零成本预过滤与标注: 时长、文本存在性、质量分档."""

from __future__ import annotations

from audio_pipeline.types import Sample


class DurationFilter:
    name = "duration"

    def __init__(self, min_sec: float = 2.0, max_sec: float = 30.0, require_text: bool = True):
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.require_text = require_text

    def process(self, samples: list[Sample]) -> None:
        for s in samples:
            if self.require_text and not (s.text and s.text.strip()):
                s.reject("empty_text")
                continue
            # 按实际解码长度过滤, 不信任数据集自带的时长字段
            # (字段与真实长度不符时, 超长音频会击穿下游模型的显存)
            dur = len(s.wav_16k()) / 16000.0
            s._duration = dur
            s.meta["duration"] = round(dur, 3)
            if dur < self.min_sec:
                s.reject(f"too_short:{dur:.2f}")
            elif dur > self.max_sec:
                s.reject(f"too_long:{dur:.2f}")


class BandwidthFilter:
    """有效带宽过滤: 频谱 roll-off(默认 0.99 能量点)低于阈值即拒绝.

    低 roll-off = 高频缺失(电话音质/低带宽转码), audiobox PQ 对此不完全敏感,
    参照 emilia-hd-pipeline 的 min_bandwidth_hz 标准单独把关.
    roll-off 中位数写入 meta["rolloff_hz"].
    """

    name = "bandwidth"

    def __init__(self, min_rolloff_hz: float = 4000.0, roll_percent: float = 0.99):
        self.min_rolloff_hz = min_rolloff_hz
        self.roll_percent = roll_percent

    def process(self, samples: list[Sample]) -> None:
        import librosa
        import numpy as np

        for s in samples:
            ro = librosa.feature.spectral_rolloff(
                y=s.wav_16k(), sr=16000, roll_percent=self.roll_percent
            )
            med = float(np.median(ro))
            s.meta["rolloff_hz"] = round(med, 1)
            if med < self.min_rolloff_hz:
                s.reject(f"bandwidth:{med:.0f}<{self.min_rolloff_hz:.0f}")


class QualityTierStage:
    """按 emilia-hd-pipeline 标准打质量档标 (只标注不过滤).

    hd: PQ >= 7.0 且 PC <= 2.5 且 (有 dnsmos 时) dnsmos >= 3.2; 其余 basic.
    写入 meta["tier"], 下游打包/训练按档筛选零成本.
    """

    name = "tier"

    def __init__(self, hd_min_pq: float = 7.0, hd_max_pc: float = 2.5, hd_min_dnsmos: float = 3.2):
        self.hd_min_pq = hd_min_pq
        self.hd_max_pc = hd_max_pc
        self.hd_min_dnsmos = hd_min_dnsmos

    def process(self, samples: list[Sample]) -> None:
        for s in samples:
            ab = s.meta.get("audiobox")
            dns = s.meta.get("dnsmos")
            hd = (
                ab is not None
                and ab["PQ"] >= self.hd_min_pq
                and ab["PC"] <= self.hd_max_pc
                and (dns is None or dns >= self.hd_min_dnsmos)
            )
            s.meta["tier"] = "hd" if hd else "basic"
