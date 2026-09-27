#!/bin/bash
# Train the shared PSID/T5 generative recommender on a given SID vocabulary.
# Usage: bash run_psid.sh <Category> <gpu_id> <vq_method> [epochs] [num_beams]
set -u
ROOT=/data/fszhang/RecBoard-master
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
CAT=${1:?category}
GPU=${2:-0}
VQ=${3:-prq}
EPOCHS=${4:-150}
BEAMS=${5:-20}
LOG="$ROOT/PRQ-SID/logs/psid_${CAT}_${VQ}.log"
mkdir -p "$ROOT/PRQ-SID/logs"
cd "$ROOT/Latte"
echo "[run_psid] cat=$CAT gpu=$GPU vq=$VQ epochs=$EPOCHS beams=$BEAMS $(date)"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -u run_local.py \
  --model PSID --category "$CAT" --vq_method "$VQ" \
  --data_root "$ROOT/data/processed" \
  --epochs "$EPOCHS" --eval_interval 1 --patience 50 \
  --num_beams "$BEAMS" --train_batch_size 256 --eval_batch_size 128 \
  > "$LOG" 2>&1
echo "[run_psid] DONE $CAT $VQ $(date)"
