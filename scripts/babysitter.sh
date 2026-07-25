#!/bin/bash
# Emilia worker 常驻看护 daemon:
# - 全局唯一实例 (flock)
# - 维持 24 个 worker (num-workers 24, ids 0-23, 每卡 3 个: GPU=id/3)
# - 缺位逐个补: 全局一次只起一个, 起完等引擎就绪(或 5 分钟超时)再补下一个
#   => 杜绝并发初始化竞态与重复启动
# - 精确存活判定: 该 worker-id 的 .venv/bin/python 进程存在即算活
set -u
SLOTS_PER_GPU=${SLOTS_PER_GPU:-3}   # 每卡 worker 数(显存紧张时降为 2)
exec 9>"${LOCK:-/tmp/emilia-babysitter.lock}"
flock -n 9 || { echo "babysitter already running"; exit 0; }

ROOT=/opt/dlami/nvme/leolxliu/open-audio-pipeline
# 可用环境变量覆盖以看护其他数据集
RUN_SCRIPT=${RUN_SCRIPT:-pipelines/run_emilia.py}
INPUT_GLOB=${INPUT_GLOB:-/opt/dlami/nvme/leolxliu/emilia-zh-full/source/*.tar}
O=${OUT_DIR:-/opt/dlami/nvme/leolxliu/emilia-clean}
CFG=${CFG:-configs/emilia.yaml}
LOCK=${LOCK:-/tmp/emilia-babysitter.lock}
RUN_BASE=$(basename "$RUN_SCRIPT")
cd "$ROOT"

alive() {  # $1=worker-id -> 0 if alive
  pgrep -f "\.venv/bin/python pipelines/$RUN_BASE .* --worker-id $1 --num-workers 24" >/dev/null
}

launch() {  # $1=worker-id
  local w=$1 G=$(( $1 / 3 ))
  echo "$(date +%H:%M:%S) launch worker $w -> GPU $G"
  CUDA_VISIBLE_DEVICES=$G VLLM_LOGGING_LEVEL=WARNING \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 NUMEXPR_NUM_THREADS=6 \
  setsid nohup bash -c "$ROOT/.venv/bin/python pipelines/$RUN_BASE \
      --input '$INPUT_GLOB' \
      --output $O --config $CFG \
      --worker-id $w --num-workers 24; echo \"WORKER_EXIT code=\$? time=\$(date +%F_%T)\"" \
      >> "$O/logs/worker$w.log" 2>&1 &
}

wait_engine() {  # $1=worker-id, 最多 300s; 就绪或死亡即返回
  local w=$1 before now t
  before=$(grep -ac "loading forced aligner" "$O/logs/worker$w.log" 2>/dev/null); before=${before:-0}
  for t in $(seq 1 20); do
    sleep 15
    alive "$w" || return 1
    now=$(grep -ac "loading forced aligner" "$O/logs/worker$w.log" 2>/dev/null); now=${now:-0}
    [ "$now" -gt "$before" ] && return 0
  done
  return 0
}

echo "$(date +%H:%M:%S) babysitter started"
while true; do
  for w in $(seq 0 23); do
    [ $((w % 3)) -ge "$SLOTS_PER_GPU" ] && continue
    if ! alive "$w"; then
      # 正常完工(code=0)的 worker 不再拉起, 避免尾声期空转
      last=$(grep -a "WORKER_EXIT" "$O/logs/worker$w.log" 2>/dev/null | tail -1)
      case "$last" in *"code=0"*) continue;; esac
      launch "$w"
      wait_engine "$w"
      sleep 30   # 沉降
    fi
  done
  sleep 60
done
