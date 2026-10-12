#!/usr/bin/env bash
# Sequential reproducibility launcher for external KD baselines on each student.
# No automatic claim that these fresh checkpoints reproduce historical tables.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${TEACHER_CKPT:?set TEACHER_CKPT to the exact fixed TIGER checkpoint}"
: "${SEM_IDS:?set SEM_IDS to the semantic-ID file}"
: "${GRAPH_DIR:?set GRAPH_DIR to the packed encoder-cache directory}"
: "${DATASET_TAG:?set DATASET_TAG for train}"
: "${VAL_DATASET_TAG:?set VAL_DATASET_TAG for validation (not test)}"
: "${CATEGORY:?set CATEGORY}"
: "${BEAM_DUMP:?set BEAM_DUMP for candidate-based KD baselines}"
DATASET="${DATASET:-Amazon2014}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-512}"
OUT_DIR="${OUT_DIR:-${GRAPH_DIR}/baseline_reproduction}"
read -r -a METHODS_ARRAY <<< "${METHODS:-rd rrd rcek lohrec gkd distillm2 smartgr}"
read -r -a ARCHES_ARRAY <<< "${ARCHES:-mlp gru cnn}"
read -r -a SEEDS_ARRAY <<< "${SEEDS:-42 43 44}"
mkdir -p "$OUT_DIR/ckpt" "$OUT_DIR/logs"

valid_args=()
if [[ -n "${VALID_SET:-}" ]]; then
  valid_args=(--valid_set "$VALID_SET")
fi
for method in "${METHODS_ARRAY[@]}"; do
  for arch in "${ARCHES_ARRAY[@]}"; do
    for seed in "${SEEDS_ARRAY[@]}"; do
      echo "TRAIN baseline=$method arch=$arch seed=$seed (best selected by FULL val NDCG@10)"
      PYTHONPATH=. python -m sid_mlp.kd_baselines \
        --method "$method" --arch "$arch" --seed "$seed" \
        --checkpoint "$TEACHER_CKPT" --sem_ids "$SEM_IDS" \
        --dataset "$DATASET" --category "$CATEGORY" \
        --dataset_tag "$DATASET_TAG" --val_dataset_tag "$VAL_DATASET_TAG" \
        --graph_dir "$GRAPH_DIR" --beam_dump "$BEAM_DUMP" \
        --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
        --val_batch_size 32 --beam_size 50 --final_top_n 50 \
        --head_hidden "${HEAD_HIDDEN:-512}" --head_layers "${HEAD_LAYERS:-1}" \
        --gru_hidden "${GRU_HIDDEN:-384}" --gru_layers "${GRU_LAYERS:-1}" \
        --conv_dim "${CONV_DIM:-256}" --conv_kernel "${CONV_KERNEL:-3}" \
        --conv_layers "${CONV_LAYERS:-2}" \
        --ckpt_dir "$OUT_DIR/ckpt" --log_dir "$OUT_DIR/logs" \
        --run_tag "${RUN_TAG:-reproduce}" "${valid_args[@]}"
    done
  done
done
echo "Training complete. EVALUATE *_best_val_ndcg10.pth, not *_final.pth."
