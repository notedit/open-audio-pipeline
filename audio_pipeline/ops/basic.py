"""零成本预过滤: 时长、文本存在性."""

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
            dur = s.duration()
            s.meta["duration"] = round(dur, 3)
            if dur < self.min_sec:
                s.reject(f"too_short:{dur:.2f}")
            elif dur > self.max_sec:
                s.reject(f"too_long:{dur:.2f}")
