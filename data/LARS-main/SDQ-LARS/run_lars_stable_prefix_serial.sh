#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:-20260928_lars_stable_prefix}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LARS="${ROOT}/SDQ-LARS"
LOCK="${LARS}/runs/${RUN_NAME}/.serial.lock"
mkdir -p "${LARS}/runs/${RUN_NAME}"
if ! mkdir "${LOCK}" 2>/dev/null; then
  echo "A StablePrefix launcher is already running for ${RUN_NAME}" >&2
  exit 3
fi
trap 'rmdir "${LOCK}" 2>/dev/null || true' EXIT

for DATASET in Beauty Sports Toys; do
  if [ -f "${LARS}/runs/${RUN_NAME}/${DATASET}/status.json" ] && grep -Eq '"status"[[:space:]]*:[[:space:]]*"complete"' "${LARS}/runs/${RUN_NAME}/${DATASET}/status.json" && [ -f "${LARS}/runs/${RUN_NAME}/${DATASET}/result.json" ]; then
    echo "$(date -Is) skipping completed ${DATASET}"
    continue
  fi
  while true; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | tr -d '[:space:]' || echo 999999)
    if [ "${USED:-999999}" -lt 30000 ]; then break; fi
    echo "$(date -Is) waiting for GPU ${GPU}: ${USED} MiB used"
    sleep 120
  done
  echo "$(date -Is) starting LARS-StablePrefix ${DATASET} on GPU ${GPU}"
  bash "${LARS}/run_lars_stable_prefix_one.sh" "${DATASET}" "${GPU}" "${RUN_NAME}"
  echo "$(date -Is) completed LARS-StablePrefix ${DATASET}"
done

