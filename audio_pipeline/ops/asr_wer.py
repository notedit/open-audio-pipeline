"""Qwen3-ASR 转写 + 双 ASR 一致性过滤.

数据集自带文本本身是旧 ASR 的输出, 不是 ground truth. 因此:
- 拼音级 WER 作为"两个独立 ASR 系统是否一致"的置信门, 超阈值即拒绝
  (两系统都不可信, 音频可能嘈杂/困难);
- 通过后默认采用更强的 Qwen3-ASR 转写作为最终文本 (adopt_text),
  顺带修复源文本错别字并获得标点; 原文本保留在 meta["orig_text"].
"""

from __future__ import annotations

import logging

from audio_pipeline.text import compute_wer, language_name
from audio_pipeline.types import Sample

logger = logging.getLogger(__name__)


class AsrWerFilter:
    name = "asr_wer"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-ASR-1.7B",
        max_wer: float = 0.1,
        batch_size: int = 16,
        device: str = "cuda",
        dtype: str = "bfloat16",
        phonetic: bool = True,
        adopt_text: bool = True,
        backend: str = "transformers",
        gpu_memory_utilization: float = 0.2,
        context: str = "",
    ):
        self.model_name = model_name
        self.max_wer = max_wer
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.phonetic = phonetic
        self.adopt_text = adopt_text
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        # context 偏置可直接控制输出形态(如强制中文数字), 免去独立 TN stage
        self.context = context
        self._model = None

    def _load(self):
        if self._model is None:
            import os
            import time

            import torch
            from qwen_asr import Qwen3ASRModel

            logger.info("loading qwen3-asr model %s (backend=%s) ...", self.model_name, self.backend)
            if self.backend == "vllm":
                # 引擎跑在本进程内(不开子进程): worker 死亡时引擎由内核保证一并退出,
                # 结构性消灭孤儿 EngineCore 占显存的问题
                os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
                # vLLM 连续批处理, 吞吐远高于 transformers; 显存占比要按
                # 同卡 worker 数量预留 (共享卡/多 worker 时调低)
                last_err = None
                for attempt in range(1, 6):
                    try:
                        self._model = Qwen3ASRModel.LLM(
                            self.model_name,
                            max_inference_batch_size=-1,
                            dtype=self.dtype,
                            gpu_memory_utilization=self.gpu_memory_utilization,
                            # ASR 输入 <=30s 音频, 无需模型默认的 65536 上下文;
                            # 不设的话小显存占比下 KV cache 校验直接拒绝启动
                            max_model_len=8192,
                            # 显式 KV 大小 => vLLM 跳过启动期显存画像;
                            # 画像对邻居显存波动极度敏感(释放即断言自杀),
                            # 固定值让引擎初始化与环境变化完全解耦
                            kv_cache_memory_bytes=8 * 2**30,
                        )
                        break
                    except Exception as e:  # 启动失败不带伤运行: 原地退避重试
                        last_err = e
                        wait = 60 * attempt
                        logger.warning(
                            "vllm engine init failed (attempt %d/5): %s; retry in %ds",
                            attempt, e, wait,
                        )
                        time.sleep(wait)
                if self._model is None:
                    raise RuntimeError(f"vllm engine init failed after 5 attempts: {last_err}")
            else:
                self._model = Qwen3ASRModel.from_pretrained(
                    self.model_name,
                    max_inference_batch_size=self.batch_size,
                    dtype=getattr(torch, self.dtype),
                    device_map=self.device,
                )
        return self._model

    def process(self, samples: list[Sample]) -> None:
        model = self._load()
        for i in range(0, len(samples), self.batch_size):
            chunk = samples[i : i + self.batch_size]
            audio = [(s.wav_16k(), 16000) for s in chunk]
            langs = [language_name(s.language) for s in chunk]
            # context 偏置(中文数字口语化)只对中文样本生效, 英文等语种传空
            contexts = [
                self.context if (s.language or "zh").lower().startswith("zh") else ""
                for s in chunk
            ]
            results = model.transcribe(audio=audio, context=contexts, language=langs)
            for s, r in zip(chunk, results):
                char_wer = compute_wer(s.text or "", r.text, s.language)
                wer = (
                    compute_wer(s.text or "", r.text, s.language, phonetic=True)
                    if self.phonetic
                    else char_wer
                )
                s.meta["asr"] = {
                    "text": r.text,
                    "language": r.language,
                    "wer": round(wer, 4) if wer != float("inf") else None,
                    "char_wer": round(char_wer, 4) if char_wer != float("inf") else None,
                }
                if wer > self.max_wer:
                    s.reject(f"wer:{wer:.3f}")
                elif self.adopt_text and r.text.strip():
                    s.meta["orig_text"] = s.text
                    s.meta["text_source"] = "qwen3-asr"
                    s.text = r.text
