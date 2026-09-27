#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:-20260927_lars_only}"
ROOT="/data/fszhang/RecBoard-master"; LARS="${ROOT}/SDQ-LARS"
mkdir -p "${LARS}/runs/${RUN_NAME}"
for pair in "Beauty 0" "Sports 1" "Toys 2"; do
  set -- ${pair}; DATASET="$1"; GPU="$2"
  mkdir -p "${LARS}/runs/${RUN_NAME}/${DATASET}"
  nohup "${LARS}/run_lars_only_one.sh" "${DATASET}" "${GPU}" "${RUN_NAME}" > "${LARS}/runs/${RUN_NAME}/${DATASET}/launcher.log" 2>&1 &
done
wait
