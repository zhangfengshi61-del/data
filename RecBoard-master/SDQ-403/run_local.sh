#!/bin/bash
# SDQ local-protocol runner (Amazon-2014 Beauty/Sports/Toys).
#
# Reproduces the SDQ-403 repository under the RecBoard-master comparison
# contract: canonical Amazon-2014 5-core leave-two-out parquet splits,
# max history 20, full-ranking evaluation with HitRate/NDCG/MRR@5/10/20.
#
# Stage 0: export freerec-format data from data/processed + sentence-t5-xl
#          item features.
# Stage 1: SDQ quantization (Structure-Diffusion VAE or KMeans) -> sid_vocab.json.
# Stage 2: TIGER-style T5 next-item training/test over the learned SIDs.
#
# Usage: bash run_local.sh <Dataset> <gpu> [variant] [sdq_epochs] [t5_epochs]
#   variant: vae | kmeans | both   (default: vae, the main SDQ method)
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
SDQ="$ROOT/SDQ-403"
RUN="$ROOT/baseline_runs/SDQ-403"
PY="${PY:-/data/fszhang/anaconda/envs/myenv_t5/bin/python}"
DATA="$SDQ/data"
MODEL_DIR="$SDQ/models"
SEM_FEAT="sentence-t5-xl_title_categories_brand.pkl"

DATASET="${1:?dataset (Beauty|Sports|Toys)}"
GPU="${2:-1}"
VARIANT="${3:-vae}"
SDQ_EPOCHS="${4:-100}"
T5_EPOCHS="${5:-200}"
NUM_BEAMS="${NUM_BEAMS:-30}"
SEED="${SEED:-2025}"

DS="Amazon2014${DATASET}_550_LOU"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN/logs"

cd "$SDQ"

# ---------------------------------------------------------------- stage 0
if [[ ! -f "$DATA/Processed/$DS/train.txt" ]]; then
  echo "[SDQ] stage 0: exporting freerec data for $DATASET"
  "$PY" prepare_local_data.py --datasets "$DATASET" --out-root "$DATA"
fi
if [[ ! -f "$DATA/Processed/$DS/$SEM_FEAT" ]]; then
  echo "[SDQ] stage 0: encoding sentence-t5-xl features for $DATASET"
  "$PY" encode_textual_features.py \
    --root "$DATA" --dataset "$DS" \
    --model sentence-t5-xl --model-dir "$MODEL_DIR" --device cuda \
    2>&1 | tee "$RUN/logs/encode_${DATASET}.log"
fi

run_vae() {
  echo "[SDQ] stage 1a: SDQ-VAE quantization ($DATASET, epochs=$SDQ_EPOCHS)"
  "$PY" train_sdq_vae.py \
    --config "configs/sdq/$DS.yaml" \
    --root "$DATA" --description SDQ --id vae --device 0 \
    --epochs "$SDQ_EPOCHS" --seed "$SEED" \
    2>&1 | tee "$RUN/logs/sdq_vae_${DATASET}.log"

  local sid="$SDQ/logs/SDQ/$DS/vae/sid_vocab.json"
  [[ -f "$sid" ]] || { echo "[SDQ] missing $sid"; return 1; }

  echo "[SDQ] stage 2a: T5 training/test ($DATASET, epochs=$T5_EPOCHS, beams=$NUM_BEAMS)"
  "$PY" train_t5.py \
    --config "configs/t5/$DS.yaml" \
    --root "$DATA" --sid-vocab-file "$sid" \
    --description SDQ-T5-VAE --id t5vae --device 0 \
    --num-beams "$NUM_BEAMS" --epochs "$T5_EPOCHS" --seed "$SEED" \
    2>&1 | tee "$RUN/logs/t5_vae_${DATASET}.log"
}

run_kmeans() {
  echo "[SDQ] stage 1b: SDQ-KMeans quantization ($DATASET)"
  "$PY" train_sdq_kmeans.py \
    --dataset "$DS" --root "$DATA" \
    --description SDQ-KMeans --id kmeans --device 0 --seed "$SEED" \
    2>&1 | tee "$RUN/logs/sdq_kmeans_${DATASET}.log"

  local sid="$SDQ/logs/SDQ-KMeans/$DS/kmeans/sid_vocab.json"
  [[ -f "$sid" ]] || { echo "[SDQ] missing $sid"; return 1; }

  echo "[SDQ] stage 2b: T5 training/test ($DATASET, epochs=$T5_EPOCHS, beams=$NUM_BEAMS)"
  "$PY" train_t5.py \
    --config "configs/t5/$DS.yaml" \
    --root "$DATA" --sid-vocab-file "$sid" \
    --description SDQ-T5-KMeans --id t5kmeans --device 0 \
    --num-beams "$NUM_BEAMS" --epochs "$T5_EPOCHS" --seed "$SEED" \
    2>&1 | tee "$RUN/logs/t5_kmeans_${DATASET}.log"
}

case "$VARIANT" in
  vae)    run_vae ;;
  kmeans) run_kmeans ;;
  both)   run_vae; run_kmeans ;;
  *) echo "unknown variant: $VARIANT"; exit 2 ;;
esac

echo "[SDQ] done: $DATASET ($VARIANT)"
