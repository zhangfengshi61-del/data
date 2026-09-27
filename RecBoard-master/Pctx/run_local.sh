#!/bin/bash
# Pctx local-protocol runner (Amazon-2014 Beauty/Sports/Toys).
#
# Stage A: train the DuoRec auxiliary model on our split.
# Stage B: Pctx upstream -> personalized semantic embeddings + clustering.
# Stage C: Pctx generative recommender training/test.
#
# Usage: bash run_local.sh <Dataset> <gpu> [duorec_epochs] [gr_epochs]
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
PCTX="$ROOT/Pctx"
RUN="$ROOT/baseline_runs/Pctx"
PY="${PY:-/data/fszhang/anaconda/envs/myenv/bin/python}"

DATASET="${1:?dataset}"
GPU="${2:-1}"
DUOREC_EPOCHS="${3:-50}"
GR_EPOCHS="${4:-150}"
SMOKE_SAMPLES="${5:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SMOKE_CONFIG=()
if [[ "$SMOKE_SAMPLES" != "0" ]]; then
  SMOKE_CONFIG+=("--smoke_samples=$SMOKE_SAMPLES")
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false

cd "$PCTX"
mkdir -p "$RUN/logs"

echo "[Pctx] stage A: DuoRec auxiliary model (dataset=$DATASET)"
"$PY" main.py \
  --model=DuoRec --dataset=Amazon2014 --category="$DATASET" --data_root="$ROOT/data/processed" \
  --epochs="$DUOREC_EPOCHS" --eval_interval=1 --patience=5 \
  --train_batch_size=256 --eval_batch_size=256 \
  "${SMOKE_CONFIG[@]}" \
  2>&1 | tee "$RUN/logs/duorec_${DATASET}.log"

DUOREC_CKPT=$(ls -t ckpt/*DuoRec*"$DATASET"*.pth | head -1)
cp "$DUOREC_CKPT" "pretrained_auxiliary_model_DuoRec/${DATASET}_duorec.pth"
echo "[Pctx] DuoRec checkpoint -> pretrained_auxiliary_model_DuoRec/${DATASET}_duorec.pth"

echo "[Pctx] stage B: upstream personalized context + clustering"
"$PY" main.py \
  --model=Pctx --dataset=Amazon2014 --category="$DATASET" --data_root="$ROOT/data/processed" \
  --run_GR_or_not=False --refresh_cluster_result=True \
  --pretrained_model_path=pretrained_auxiliary_model_DuoRec \
  --pretrained_model_name="${DATASET}_duorec.pth" \
  --n_groups=11 --distance=4 --start=2 --k_gamma=4.6 \
  "${SMOKE_CONFIG[@]}" \
  2>&1 | tee "$RUN/logs/upstream_${DATASET}.log"

echo "[Pctx] stage C: generative recommender"
"$PY" main.py \
  --model=Pctx --dataset=Amazon2014 --category="$DATASET" --data_root="$ROOT/data/processed" \
  --run_GR_or_not=True --rq_faiss=True \
  --frequency_threshold=0.2 --augmentation_probability=0.6 \
  --epochs="$GR_EPOCHS" --eval_interval=1 --patience=10 \
  --train_batch_size=256 --eval_batch_size="$EVAL_BATCH_SIZE" \
  "${SMOKE_CONFIG[@]}" \
  2>&1 | tee "$RUN/logs/gr_${DATASET}.log"
