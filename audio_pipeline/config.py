"""yaml 配置 -> stage 列表的装配."""

from __future__ import annotations

from pathlib import Path

import yaml

from audio_pipeline.ops import (
    AbnormalPauseFilter,
    AsrWerFilter,
    AudioboxFilter,
    DurationFilter,
    EdgeSilenceTrimStage,
    ForcedAlignStage,
    MultiSpeakerFilter,
    PausePunctuationStage,
    TextNormalizationStage,
)
from audio_pipeline.pipeline import Pipeline, Stage


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_stages(cfg: dict, device: str = "cuda") -> list[Stage]:
    """按固定成本序装配 stage: 时长 -> 美学 -> 说话人 -> ASR -> 对齐 -> 停顿."""
    stages: list[Stage] = []
    if "duration" in cfg:
        d = cfg["duration"]
        stages.append(DurationFilter(min_sec=d["min_sec"], max_sec=d["max_sec"]))
    ab = cfg.get("audiobox", {})
    if ab.get("enabled"):
        stages.append(
            AudioboxFilter(
                min_pq=ab.get("min_pq"), min_ce=ab.get("min_ce"),
                min_cu=ab.get("min_cu"), max_pc=ab.get("max_pc"),
                batch_size=ab.get("batch_size", 32), device=device,
            )
        )
    di = cfg.get("diarization", {})
    if di.get("enabled"):
        stages.append(
            MultiSpeakerFilter(
                model_name=di.get("model", "nvidia/diar_streaming_sortformer_4spk-v2"),
                max_speakers=di.get("max_speakers", 1),
                min_speaker_sec=di.get("min_speaker_sec", 0.5),
                batch_size=di.get("batch_size", 16), device=device,
            )
        )
    asr = cfg.get("asr", {})
    if asr.get("enabled"):
        stages.append(
            AsrWerFilter(
                model_name=asr.get("model", "Qwen/Qwen3-ASR-1.7B"),
                max_wer=asr.get("max_wer", 0.1),
                phonetic=asr.get("phonetic", True),
                adopt_text=asr.get("adopt_text", True),
                backend=asr.get("backend", "transformers"),
                gpu_memory_utilization=asr.get("gpu_memory_utilization", 0.2),
                context=asr.get("context", ""),
                batch_size=asr.get("batch_size", 16), device=device,
            )
        )
    tn = cfg.get("tn", {})
    if tn.get("enabled"):
        stages.append(
            TextNormalizationStage(
                model_name=tn.get("model", "Qwen/Qwen3-4B"),
                batch_size=tn.get("batch_size", 8), device=device,
            )
        )
    al = cfg.get("align", {})
    if al.get("enabled"):
        stages.append(
            ForcedAlignStage(
                model_name=al.get("model", "Qwen/Qwen3-ForcedAligner-0.6B"),
                batch_size=al.get("batch_size", 32), device=device,
            )
        )
    tr = cfg.get("trim", {})
    if tr.get("enabled"):
        stages.append(
            EdgeSilenceTrimStage(
                head_silence_sec=tr.get("head_silence_sec", 0.1),
                tail_silence_sec=tr.get("tail_silence_sec", 0.3),
            )
        )
    pa = cfg.get("pause", {})
    if pa.get("enabled"):
        stages.append(
            AbnormalPauseFilter(
                max_gap_sec=pa.get("max_gap_sec", 1.5),
                max_edge_silence_sec=pa.get("max_edge_silence_sec", 1.0),
            )
        )
    pu = cfg.get("punct", {})
    if pu.get("enabled"):
        stages.append(
            PausePunctuationStage(
                comma_gap_sec=pu.get("comma_gap_sec", 0.3),
                period_gap_sec=pu.get("period_gap_sec", 0.8),
            )
        )
    return stages


def build_pipeline(cfg: dict, device: str = "cuda") -> Pipeline:
    return Pipeline(build_stages(cfg, device=device))
