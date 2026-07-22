"""文本归一化与 WER/CER 计算.

中文按字算 CER, 英文按词算 WER. 计算前做归一化:
去标点、转小写、全角转半角, 避免格式差异虚增错误率.
"""

from __future__ import annotations

import re
import unicodedata

import jiwer

# 保留中日韩统一表意文字、字母、数字; 其余(标点/符号/空白)替换为空格
_KEEP_RE = re.compile(r"[^\w一-鿿㐀-䶿]+", re.UNICODE)
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # 全角 -> 半角
    text = _KEEP_RE.sub(" ", text)
    text = text.lower().strip()
    # 下划线属于 \w 但没有语言意义
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text)


def to_units(text: str, language: str | None) -> list[str]:
    """切分为计算错误率的单元: 中文单字 + 非中文按空格分词."""
    text = normalize_text(text)
    if not text:
        return []
    # 中文字符两侧插空格, 单字成单元; 英文等保持词级
    spaced = _CJK_RE.sub(lambda m: f" {m.group(0)} ", text)
    return spaced.split()


def compute_wer(
    reference: str, hypothesis: str, language: str | None = None, phonetic: bool = False
) -> float:
    """归一化后的 WER (中文实际为 CER). 参考为空时返回 inf.

    phonetic=True 时中文按拼音比较: 同音字替换(贵绅/贵身)和数字写法差异
    (一八七六/1876)不计错 —— 发音与音频一致的样本对 TTS 是好样本.
    """
    if phonetic:
        reference = _numbers_to_cn(reference)
        hypothesis = _numbers_to_cn(hypothesis)
    ref_units = to_units(reference, language)
    hyp_units = to_units(hypothesis, language)
    if phonetic:
        ref_units = _to_pinyin(ref_units)
        hyp_units = _to_pinyin(hyp_units)
    if not ref_units:
        return float("inf")
    if not hyp_units:
        return 1.0
    return jiwer.wer(" ".join(ref_units), " ".join(hyp_units))


def _numbers_to_cn(text: str) -> str:
    """阿拉伯数字转中文读法, 失败时原样返回."""
    import cn2an

    try:
        return cn2an.transform(text, "an2cn")
    except Exception:
        return text


def _to_pinyin(units: list[str]) -> list[str]:
    """CJK 单元转无声调拼音; 非中文单元原样保留 (lazy_pinyin 对其透传)."""
    from pypinyin import lazy_pinyin

    return ["".join(lazy_pinyin(u)) for u in units]


# qwen-asr 需要完整语言名
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
}


def language_name(code: str | None) -> str | None:
    if code is None:
        return None
    return LANGUAGE_NAMES.get(code.lower(), code)
