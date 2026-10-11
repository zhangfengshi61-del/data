#!/usr/bin/env bash
# ServeSID WWW paper run: identical architecture/data for seeds 42, 43 and 44.
# Usage: GRAPH_DIR=/path/graph MODEL_DIR=/path/models BEAM_DUMP=/path/teacher.pt \
#        bash scripts/run_paper_seeds.sh sidmlp Beauty 0
#        DATASET=AmazonReviews2023 ... bash scripts/run_paper_seeds.sh sidmlp Video_Games 0
# Override SEEDS="42 43 44" (default) or SEEDS="43" to resume a single seed.
set -euo pipefail
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
MODE="${1:-sidmlp}"
CATEGORY="${2:-}"
DEVICE="${3:-0}"
if [[ "$MODE" != "sidmlp" ]]; then
  echo "This public code path only implements the MLP student; GRU/CNN require their real implementations." >&2
  exit 2
fi
: "${CATEGORY:?Usage: $0 sidmlp <Beauty|Toys|Video_Games> [GPU_INDEX]}"
DATASET="${DATASET:-Amazon2014}"
if [[ "$CATEGORY" == "Video_Games" && -z "${DATASET_OVERRIDE:-}" ]]; then DATASET="AmazonReviews2023"; fi
: "${GRAPH_DIR:?Set GRAPH_DIR to the prepared teacher caches and packed datasets}"
: "${MODEL_DIR:?Set MODEL_DIR to the TIGER checkpoints and semantic-ID inventory}"
: "${BEAM_DUMP:?Set BEAM_DUMP to the matching TRAIN-request teacher beam dump}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/servesid_final.yaml}"
DATASET_TAG="${DATASET_TAG:-${DATASET}_${CATEGORY}}"
VAL_DATASET_TAG="${VAL_DATASET_TAG:-${DATASET_TAG}_val}"
TIGER_CKPT="${TIGER_CKPT:-$MODEL_DIR/TIGER-${DATASET}-category_${CATEGORY}.pth}"
SEM_IDS="${SEM_IDS:-$MODEL_DIR/semantic_ids/${DATASET}-${CATEGORY}_sentence-t5-base_256,256,256,256.sem_ids}"
SEEDS="${SEEDS:-42 43 44}"
CKPT_DIR="${CKPT_DIR:-$GRAPH_DIR/SID-MLP/ckpt}"
LOG_DIR="${LOG_DIR:-$GRAPH_DIR/SID-MLP/logs}"
for file in "$CONFIG" "$BEAM_DUMP" "$TIGER_CKPT" "$SEM_IDS"; do
  if [[ ! -f "$file" ]]; then echo "Missing required file: $file" >&2; exit 3; fi
done
mkdir -p "$CKPT_DIR" "$LOG_DIR"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$DEVICE"
for SEED in $SEEDS; do
  case "$SEED" in 42|43|44) ;; *) echo "Seed $SEED is not in paper's approved list 42/43/44" >&2; exit 4;; esac
  RUN_TAG="${RUN_TAG_PREFIX:-servesid_paper}_${DATASET}_${CATEGORY}_seed${SEED}"
  echo "Starting paper reproduction dataset=$DATASET category=$CATEGORY seed=$SEED config=$CONFIG"
  python -m sid_mlp.cli.train sidmlp \
    --config "$CONFIG" --checkpoint "$TIGER_CKPT" \
    --dataset "$DATASET" --category "$CATEGORY" \
    --dataset_tag "$DATASET_TAG" --val_dataset_tag "$VAL_DATASET_TAG" \
    --sem_ids "$SEM_IDS" --graph_dir "$GRAPH_DIR" \
    --beam_dump "$BEAM_DUMP" --ckpt_dir "$CKPT_DIR" --log_dir "$LOG_DIR" \
    --seed "$SEED" --run_tag "$RUN_TAG"
  echo "Completed seed=$SEED; use *_best_val_ndcg10.pth for test evaluation."
done
