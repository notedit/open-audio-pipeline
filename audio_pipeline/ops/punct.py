"""基于停顿的标点修正.

复用强制对齐时间戳: 相邻发音单元之间的静音超过阈值而文本中无标点时,
按停顿时长补标点 —— >= comma_gap_sec 补逗号, >= period_gap_sec 补句号.
让文本标点忠实反映音频的真实停顿, TTS 训练时标点与韵律一致.

必须放在 ForcedAlignStage 之后. 插入明细写入 meta["punct_fix"] 供审计.
"""

from __future__ import annotations

from audio_pipeline.types import Sample

# 视为"已有标点"的字符: 出现在两个发音单元之间则不再插入
PUNCT_CHARS = set("，。、；：？！…—,.;:?!\"'“”‘’()（）《》〈〉[]【】~～· ")


def insert_pause_punct(
    text: str,
    items: list[dict],
    comma_gap_sec: float,
    period_gap_sec: float,
) -> tuple[str, list[dict]]:
    """按停顿插入标点. 返回 (新文本, 插入明细); 对齐单元与文本对不上时原样返回."""
    spans = []
    pos = 0
    for it in items:
        idx = text.find(it["text"], pos)
        if idx < 0:
            return text, []
        spans.append((idx, idx + len(it["text"])))
        pos = idx + len(it["text"])

    insertions = []
    for i in range(len(items) - 1):
        gap = items[i + 1]["start"] - items[i]["end"]
        if gap < comma_gap_sec:
            continue
        between = text[spans[i][1] : spans[i + 1][0]]
        if any(c in PUNCT_CHARS for c in between):
            continue
        ch = "。" if gap >= period_gap_sec else "，"
        insertions.append({"pos": spans[i][1], "char": ch, "gap": round(gap, 3)})

    if not insertions:
        return text, []
    out, prev = [], 0
    for ins in insertions:
        out.append(text[prev : ins["pos"]])
        out.append(ins["char"])
        prev = ins["pos"]
    out.append(text[prev:])
    return "".join(out), insertions


class PausePunctuationStage:
    name = "punct"

    def __init__(self, comma_gap_sec: float = 0.3, period_gap_sec: float = 0.8):
        self.comma_gap_sec = comma_gap_sec
        self.period_gap_sec = period_gap_sec

    def process(self, samples: list[Sample]) -> None:
        for s in samples:
            items = s.meta.get("alignment")
            if not items or not s.text:
                continue
            new_text, insertions = insert_pause_punct(
                s.text, items, self.comma_gap_sec, self.period_gap_sec
            )
            if insertions:
                s.meta["punct_fix"] = {
                    "n_inserted": len(insertions),
                    "inserted": insertions,
                    "text_before": s.text,
                }
                s.text = new_text
