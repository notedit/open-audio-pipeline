#!/usr/bin/env bash
# 多卡多 worker 启动: 每卡 PER_GPU 个进程, 按分片取模切分任务.
# 用法:
#   bash scripts/launch_workers.sh pipelines/run_emilia.py \
#       "/data/Emilia/ZH/*.tar" /data/emilia-clean configs/emilia.yaml [GPUS=8] [PER_GPU=2]
# 注意: PER_GPU=2 时 config 里 asr.gpu_memory_utilization 建议 <=0.15
set -euo pipefail

RUN=$1; INPUT=$2; OUTPUT=$3; CONFIG=$4
GPUS=${5:-8}; PER_GPU=${6:-2}
TOTAL=$((GPUS * PER_GPU))
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$OUTPUT/logs"
for g in $(seq 0 $((GPUS - 1))); do
  for m in $(seq 0 $((PER_GPU - 1))); do
    w=$((g * PER_GPU + m))
    CUDA_VISIBLE_DEVICES=$g nohup "$ROOT/.venv/bin/python" "$RUN" \
      --input "$INPUT" --output "$OUTPUT" --config "$CONFIG" \
      --worker-id "$w" --num-workers "$TOTAL" \
      > "$OUTPUT/logs/worker$w.log" 2>&1 &
    echo "worker $w -> GPU $g (pid $!)"
  done
done
echo "launched $TOTAL workers ($GPUS GPUs x $PER_GPU). logs: $OUTPUT/logs/"
echo "progress: grep -h kept $OUTPUT/logs/*.log | tail"