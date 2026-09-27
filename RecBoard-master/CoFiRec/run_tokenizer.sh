#!/bin/bash
# Train a local CoFiRec tokenizer for a dataset CoFiRec did not ship and
# generate the 4-token item index used by the generation stage.
#
# Usage: bash run_tokenizer.sh <Dataset> <gpu> <epochs>
set -euo pipefail

ROOT="/data/fszhang/tiger协同语义训练/experiments"
COFI="$ROOT/baselines/CoFiRec"
RUN="$ROOT/baseline_runs/CoFiRec"
PY="/data/fszhang/anaconda/envs/myenv/bin/python"

DATASET="${1:?dataset}"
GPU="${2:-1}"
EPOCHS="${3:-200}"

cd "$COFI"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_MODE=disabled

echo "[CoFiRec tokenizer] dataset=$DATASET gpu=$GPU epochs=$EPOCHS"

"$PY" tokenizer/main.py \
  --device cuda:0 \
  --data_path "./data/${DATASET}/${DATASET}.sem-3-level.npy" \
  --ckpt_dir "./checkpoint/${DATASET}" \
  --batch_size 1024 \
  --epochs "$EPOCHS" \
  --early_patience 30 \
  --num_workers 8 \
  2>&1 | tee "$RUN/logs/tokenizer_${DATASET}.log"

CKPT=$(find "./checkpoint/${DATASET}" -name best_collision_model.pth -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
echo "[CoFiRec tokenizer] best checkpoint: $CKPT"

"$PY" tokenizer/generate_indices.py \
  --dataset "$DATASET" \
  --checkpoint "$CKPT" \
  --epoch "$EPOCHS" \
  2>&1 | tee "$RUN/logs/tokenizer_index_${DATASET}.log"

INDEX=$(find "./data/${DATASET}" -maxdepth 1 -name "${DATASET}.index.epoch*.json" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
cp "$INDEX" "./data/${DATASET}/${DATASET}.index.json"
echo "[CoFiRec tokenizer] index: $INDEX -> ./data/${DATASET}/${DATASET}.index.json"
