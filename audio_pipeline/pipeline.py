"""Stage 协议与 Pipeline 组合器.

每个原子能力实现为一个 Stage: 接收一批 Sample, 就地写入 meta 标注,
对不合格样本调用 sample.reject(reason). Pipeline 按顺序执行 stage,
每个 stage 之后剔除被拒样本(不再进入更贵的后续 stage), 并记录丢弃原因.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Protocol

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


class Stage(Protocol):
    name: str

    def process(self, samples: list[Sample]) -> None: ...


class Pipeline:
    def __init__(self, stages: list[Stage], on_drop: Callable[[Sample, str], None] | None = None):
        self.stages = stages
        self.on_drop = on_drop
        self.stats: dict[str, int] = {"input": 0, "output": 0}
        self.stage_time: dict[str, float] = {}  # 各 stage 累计耗时(秒)
        self.stage_items: dict[str, int] = {}   # 各 stage 实际处理的样本数
        self.audio_sec: float = 0.0             # 进入 pipeline 的音频总时长

    def run(self, samples: list[Sample]) -> list[Sample]:
        """处理一批样本, 返回通过全部 stage 的样本."""
        self.stats["input"] += len(samples)
        alive = []
        for s in samples:
            if s.rejected:  # 进入 pipeline 前已被拒 (如解码失败)
                self.stats["drop:input"] = self.stats.get("drop:input", 0) + 1
                if self.on_drop:
                    self.on_drop(s, "input")
            else:
                alive.append(s)
        for s in alive:
            try:
                self.audio_sec += s.duration()
            except Exception:
                pass
        for stage in self.stages:
            if not alive:
                break
            t0 = time.perf_counter()
            stage.process(alive)
            self.stage_time[stage.name] = self.stage_time.get(stage.name, 0.0) + time.perf_counter() - t0
            self.stage_items[stage.name] = self.stage_items.get(stage.name, 0) + len(alive)
            survivors = []
            for s in alive:
                if s.rejected:
                    self.stats[f"drop:{stage.name}"] = self.stats.get(f"drop:{stage.name}", 0) + 1
                    if self.on_drop:
                        self.on_drop(s, stage.name)
                    s.free_wav()
                else:
                    survivors.append(s)
            alive = survivors
        self.stats["output"] += len(alive)
        return alive

    def format_stats(self) -> str:
        parts = [f"input={self.stats['input']}", f"output={self.stats['output']}"]
        parts += [f"{k}={v}" for k, v in sorted(self.stats.items()) if k.startswith("drop:")]
        return " ".join(parts)

    def format_timing(self) -> str:
        """各 stage 耗时占比与吞吐 (xRT = 每秒钟处理多少秒音频)."""
        total = sum(self.stage_time.values())
        if total == 0:
            return "no timing data"
        lines = []
        for name, t in sorted(self.stage_time.items(), key=lambda kv: -kv[1]):
            n = self.stage_items.get(name, 0)
            lines.append(f"{name}: {t:.1f}s ({100*t/total:.0f}%) {n/t:.1f} samples/s")
        rtf = self.audio_sec / total if total else 0
        lines.append(f"total: {total:.1f}s for {self.audio_sec/3600:.2f}h audio => {rtf:.1f}x realtime")
        return " | ".join(lines)
