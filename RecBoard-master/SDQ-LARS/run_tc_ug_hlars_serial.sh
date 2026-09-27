#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:-20260926_tc_ug_hlars}"
GPU="${GPU:-3}"
PYTHON="/data/fszhang/anaconda/envs/myenv_t5/bin/python"
ROOT="/data/fszhang/RecBoard-master/SDQ-LARS"
for DATASET in Beauty Sports Toys; do
  echo "[$(date -Is)] START ${DATASET} tc_ug"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${ROOT}/run_experiment.py" \
    --dataset "${DATASET}" --variant tc_ug --gpu 0 --run-name "${RUN_NAME}"
done
echo "[$(date -Is)] ALL_DONE ${RUN_NAME}"
