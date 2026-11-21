#!/bin/bash
# 启动 refilter_diar.py worker: GPU_LIST 每卡 SLOTS_PER_GPU 个, 每个 worker 崩溃后自动重启,
# 正常退出 (没有可领的分片) 即结束. 分片动态领取, worker 数可随时增减.
#   GPU_LIST="0 1" SLOTS_PER_GPU=3 bash scripts/refilter_launch.sh
set -u
ROOT=/opt/dlami/nvme/leolxliu/open-audio-pipeline
SRC=${SRC:-/mnt/fsx/leolxliu/open-audio-data}
WORK=${WORK:-/opt/dlami/nvme/leolxliu/open-audio-refilter}
NAMES=${NAMES:-"emilia-yodas-zh emilia-ko emilia-yodas-ja emilia-ja emilia-yodas-ko wenetspeech emilia-en emilia-yodas-en emilia"}
GPU_LIST=(${GPU_LIST:-0 1})
SLOTS_PER_GPU=${SLOTS_PER_GPU:-3}
mkdir -p "$WORK/logs"
cd "$ROOT"
for g in "${GPU_LIST[@]}"; do
  for k in $(seq 0 $((SLOTS_PER_GPU - 1))); do
    log="$WORK/logs/gpu${g}-w${k}.log"
    CUDA_VISIBLE_DEVICES=$g HF_HOME=/opt/dlami/nvme/leolxliu/.cache/huggingface OMP_NUM_THREADS=4 \
    setsid nohup bash -c "until $ROOT/.venv/bin/python tools/refilter_diar.py --src $SRC --work $WORK --names $NAMES; do echo \"WORKER_CRASH code=\$? \$(date +%F_%T), restart in 60s\"; sleep 60; done; echo \"WORKER_EXIT \$(date +%F_%T)\"" \
      >> "$log" 2>&1 < /dev/null &
    echo "gpu $g slot $k -> $log"
  done
done
