"""LLM 文本正则化 (TN): 书面形式 -> 口语读法.

Qwen3-ASR 偶尔输出书面形式(阿拉伯数字/符号, 如 "1876年"), 对 TTS 训练是
歧义文本(多种读法, 音频只读了一种). 本 stage 用本地通用 LLM 把书面形式
改写为与音频一致的口语读法:

- 只对含数字/符号的文本触发 (实测 <1%, 开销可忽略);
- prompt 中携带 orig_text(旧 ASR 的口语形式转写)作为读法参考,
  音频的真实读法大多记录在其中, 模型主要做对齐改写而非凭空猜测;
- 输出校验: 不得再含数字/符号、长度比例合理, 否则回退原文并标记.

放在 AsrWerFilter(文本定稿)之后、ForcedAlignStage 之前.
改写明细写入 meta["tn"].
"""

from __future__ import annotations

import logging
import re

from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)

WRITTEN_FORM_RE = re.compile(r"[\d%℃$¥€]")

_PROMPT = """把下面文本中的阿拉伯数字、符号等书面形式改写成中文口语读法。这是语音转写文本,改写结果必须与音频读法一致:
- 严格按读法逐字改写,禁止意译、换说法、增删内容;其余文字和标点一字不动;
- 优先参考「参考转写」中的读法(它来自同一段音频的另一份转写);
- 年份、电话、编号逐位读;数量按数值读;百分号读"百分之"。

示例:
- 1876年出生 → 一八七六年出生
- 投资210亿元 → 投资二百一十亿元
- 有50%的把握 → 有百分之五十的把握
- 拨打110 → 拨打幺幺零
- 下午2:30开会 → 下午两点三十分开会
- 房间号1204 → 房间号幺二零四

参考转写: {ref}
待改写文本: {text}
只输出改写后的文本,不要任何解释。"""


class TextNormalizationStage:
    name = "tn"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        batch_size: int = 8,
        max_new_tokens: int = 512,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("loading TN model %s ...", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype), device_map=self.device
            )
            self._model.eval()
        return self._model, self._tokenizer

    def normalize_texts(self, texts: list[str], refs: list[str]) -> list[str]:
        """批量改写. 返回与输入等长的结果列表(校验失败的返回原文)."""
        import torch

        model, tok = self._load()
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": _PROMPT.format(ref=r or "(无)", text=t)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for t, r in zip(texts, refs)
        ]
        out: list[str] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            # 输出与输入等长量级, 按本批最长文本动态收紧生成上限, 避免小批次空转
            chunk_max = max(len(t) for t in texts[i : i + self.batch_size])
            max_new = min(self.max_new_tokens, chunk_max * 2 + 32)
            inputs = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **inputs, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            for j, seq in enumerate(gen):
                text = tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                out.append(text.strip())
        return [
            self._validate(orig, new) for orig, new in zip(texts, out)
        ]

    @staticmethod
    def _validate(orig: str, new: str) -> str:
        """改写结果不含书面形式、长度合理才接受, 否则回退原文."""
        if not new or WRITTEN_FORM_RE.search(new):
            return orig
        if not 0.5 <= len(new) / max(len(orig), 1) <= 3.0:
            return orig
        return new

    def process(self, samples: list[Sample]) -> None:
        todo = [s for s in samples if s.text and WRITTEN_FORM_RE.search(s.text)]
        if not todo:
            return
        refs = [s.meta.get("orig_text") or "" for s in todo]
        results = self.normalize_texts([s.text for s in todo], refs)
        for s, new in zip(todo, results):
            if new != s.text:
                s.meta["tn"] = {"text_before": s.text}
                s.text = new
            else:
                s.meta["tn"] = {"unchanged": True}
