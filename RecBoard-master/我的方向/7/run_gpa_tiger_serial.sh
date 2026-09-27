#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:-20260926_gpa_tiger_serial}"
GPU="${GPU:-3}"
PYTHON="/data/fszhang/anaconda/envs/myenv_t5/bin/python"
ROOT="/data/fszhang/RecBoard-master"
for DATASET in Beauty Sports Toys; do
  echo "[$(date -Is)] START ${DATASET} GPA-TIGER"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${ROOT}/SDQ-403/train_t5_hpa.py" \
    --config "${ROOT}/SDQ-403/configs/t5/Amazon2014${DATASET}_550_LOU.yaml" \
    --root "${ROOT}/SDQ-403/data" \
    --sid-vocab-file "${ROOT}/SDQ-403/logs/SDQ/Amazon2014${DATASET}_550_LOU/vae/sid_vocab.json" \
    --description GPA-TIGER --id gpa_tiger --device 0 \
    --num-beams 30 --eval-batch-size 96 --apply-constrained-beam-search False \
    --epochs 200 --seed 2025 --lambda-prefix 0.08 --lambda-metric 0.0 \
    --prefix-temperature 0.10 --prefix-levels 1 --aux-max-ratio 0.05
done
echo "[$(date -Is)] ALL_DONE ${RUN_NAME}"
