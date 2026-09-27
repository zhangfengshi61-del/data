#!/bin/bash
# CoFiRec local-protocol runner.
#
# Beauty uses the official CoFiRec Amazon-2014 Beauty assets. We verified that
# the official item catalog (12101), user set (22363), every user sequence and
# the leave-two-out split are identical to our canonical parquet protocol, so
# the official index/embedding is a faithful reproduction on our split.
#
# Sports/Toys are not shipped by CoFiRec and use the locally trained tokenizer.
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
GEN="$ROOT/CoFiRec/generation"
RUN="$ROOT/baseline_runs/CoFiRec"
PY="/data/fszhang/anaconda/envs/myenv/bin/python"

DATASET="${1:-Beauty}"
GPU="${2:-0}"
EPOCHS="${3:-200}"
BEAMS="${4:-30}"

mkdir -p "$RUN/ckpt" "$RUN/logs"

cd "$GEN"

echo "[CoFiRec] dataset=$DATASET gpu=$GPU epochs=$EPOCHS beams=$BEAMS"

CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled "$PY" finetune.py \
  --output_dir "$RUN/ckpt/${DATASET}_full" \
  --dataset "$DATASET" \
  --data_path "$ROOT/CoFiRec/data" \
  --per_device_batch_size 256 \
  --gradient_accumulation_steps 1 \
  --learning_rate 7e-4 \
  --epochs "$EPOCHS" \
  --index_file .index.json \
  --temperature 1.0 \
  --fp16 \
  --logging_step 50 \
  --wandb_name "cofirec_${DATASET}" \
  2>&1 | tee "$RUN/logs/${DATASET}_full.log"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" test.py \
  --gpu_id 0 \
  --ckpt_path "$RUN/ckpt/${DATASET}_full" \
  --dataset "$DATASET" \
  --data_path "$ROOT/CoFiRec/data" \
  --results_file "$RUN/logs/${DATASET}_full_test.json" \
  --test_batch_size 32 \
  --num_beams "$BEAMS" \
  --test_prompt_ids 0 \
  --index_file .index.json \
  --metrics "hit@5,hit@10,hit@20,ndcg@5,ndcg@10,ndcg@20,mrr" \
  2>&1 | tee "$RUN/logs/${DATASET}_full_test.log"
