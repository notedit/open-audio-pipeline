"""异常停顿检测.

复用强制对齐产出的字/词级时间戳(meta["alignment"]), 零额外模型开销:
- 句内相邻单元之间静音 > max_gap_sec 视为异常停顿
- 首/尾静音 > max_edge_silence_sec 视为异常
统计写入 meta["pause"], 命中即拒绝.

必须放在 ForcedAlignStage 之后.
"""

from __future__ import annotations

from audio_pipeline.types import Sample


class AbnormalPauseFilter:
    name = "pause"

    def __init__(self, max_gap_sec: float = 1.5, max_edge_silence_sec: float = 1.0):
        self.max_gap_sec = max_gap_sec
        self.max_edge_silence_sec = max_edge_silence_sec

    def process(self, samples: list[Sample]) -> None:
        for s in samples:
            items = s.meta.get("alignment")
            if not items:
                s.reject("pause_no_alignment")
                continue
            dur = s.duration()
            max_gap = 0.0
            for prev, cur in zip(items, items[1:]):
                max_gap = max(max_gap, cur["start"] - prev["end"])
            head = items[0]["start"]
            tail = max(0.0, dur - items[-1]["end"])
            s.meta["pause"] = {
                "max_gap": round(max_gap, 3),
                "head_silence": round(head, 3),
                "tail_silence": round(tail, 3),
            }
            if max_gap > self.max_gap_sec:
                s.reject(f"pause_gap:{max_gap:.2f}")
            elif head > self.max_edge_silence_sec:
                s.reject(f"pause_head:{head:.2f}")
            elif tail > self.max_edge_silence_sec:
                s.reject(f"pause_tail:{tail:.2f}")
