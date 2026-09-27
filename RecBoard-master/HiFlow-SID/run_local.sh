#!/bin/bash
# HiFlow-SID local-protocol runner (Amazon-2014 Beauty/Sports/Toys).
#
# RecBoard-master comparison contract:
#   - canonical Amazon-2014 5-core leave-two-out parquet splits
#   - max history 20, full-ranking evaluation, seen items masked
#   - HitRate/NDCG/MRR@5/10/20, seed 2025
#   - SID: the SDQ-VAE 3-token SID used by the TIGER-style T5 baseline
#
# Usage: bash run_local.sh <Dataset> <gpu> [phase]
#   phase: mtp | hiflow | all (default: all)
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
HIF="$ROOT/HiFlow-SID"
SDQ="$ROOT/SDQ-403"
PY="${PY:-/data/fszhang/anaconda/envs/myenv_t5/bin/python}"

DATASET="${1:?dataset (Beauty|Sports|Toys)}"
GPU="${2:-1}"
PHASE="${3:-all}"

DS="Amazon2014${DATASET}_550_LOU"
SID_VOCAB="$SDQ/logs/SDQ/$DS/vae/sid_vocab.json"
LOG_DIR="$HIF/logs/HiFlow-SID/$DS"

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HIF/logs"

cd "$HIF"

run_mtp() {
  echo "[HiFlow] phase 1: MTP ($DATASET)"
  "$PY" train_mtp.py \
    --category "$DATASET" --dataset "$DS" \
    --sid-vocab-file "$SID_VOCAB" \
    --id mtp --epochs 100 --seed 2025 --device 0 \
    2>&1 | tee "$HIF/logs/mtp_${DATASET}.log"
}

run_hiflow() {
  local mtp_best="$LOG_DIR/mtp/best.pt"
  [[ -f "$mtp_best" ]] || { echo "[HiFlow] missing $mtp_best"; return 1; }
  echo "[HiFlow] phase 2: HiFlow-SID-OneStep ($DATASET)"
  "$PY" train_hiflow.py \
    --category "$DATASET" --dataset "$DS" \
    --sid-vocab-file "$SID_VOCAB" \
    --mtp-checkpoint "$mtp_best" \
    --id hiflow-s1 --epochs 100 --seed 2025 --device 0 \
    2>&1 | tee "$HIF/logs/hiflow_${DATASET}.log"
}

case "$PHASE" in
  mtp)    run_mtp ;;
  hiflow) run_hiflow ;;
  all)    run_mtp; run_hiflow ;;
  *) echo "unknown phase: $PHASE"; exit 2 ;;
esac

echo "[HiFlow] done: $DATASET ($PHASE)"
