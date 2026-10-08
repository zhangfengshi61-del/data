#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LARS="${ROOT}/SDQ-LARS"
SDQ="${ROOT}/SDQ-403"
PY="${PY:-python}"
RUN="20260928_lars_main_common_fix"
DS="Amazon2014Clothing_550_LOU"
OUT="${LARS}/runs/${RUN}/Clothing"
DESC="LARS-${RUN}-main-Clothing"
T5DESC="${DESC}-T5"
mkdir -p "${OUT}"
cd "${LARS}"
echo "[start] lars_vae $(date -Is)" > "${OUT}/worker.log"
CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${LARS}/train_lars_vae.py" \
  --config "${SDQ}/configs/sdq/${DS}.yaml" --root "${SDQ}/data" \
  --description "${DESC}" --id vae --device 0 --epochs 100 --seed 2025 \
  --lars-steps 3 --sparse-weight 0.1 --bridge-weight 0.1 \
  --lars-warmup 10 --lars-ridge 1e-4 \
  > "${OUT}/lars_vae.log" 2>&1
SID="${LARS}/logs/${DESC}/${DS}/vae/sid_vocab.json"
echo "[done] lars_vae; [start] t5 $(date -Is)" >> "${OUT}/worker.log"
CUDA_VISIBLE_DEVICES=1 "${PY}" -u "${SDQ}/train_t5_fp32.py" \
  --config "${SDQ}/configs/t5/${DS}.yaml" --root "${SDQ}/data" \
  --sid-vocab-file "${SID}" --description "${T5DESC}" --id t5 \
  --device 0 --num-workers 0 --num-beams 30 --batch-size 512 \
  --eval-batch-size 96 --epochs 200 --seed 2025 \
  --apply-constrained-beam-search False --prefix-loss-weight 0.0 \
  --log2file --log2console > "${OUT}/t5.log" 2>&1
LOG="${LARS}/logs/${T5DESC}/${DS}/t5"
"${PY}" "${LARS}/finalize_t5_result.py" "${LOG}"
cp "${LOG}/result.json" "${OUT}/result.json"
echo "[done] t5 $(date -Is)" >> "${OUT}/worker.log"

