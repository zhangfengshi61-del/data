#!/usr/bin/env bash
set -euo pipefail
RUN_NAME="${1:-20260926_cp_ug_hlars}"
GPU="${GPU:-3}"
PYTHON="/data/fszhang/anaconda/envs/myenv_t5/bin/python"
ROOT="/data/fszhang/RecBoard-master"
LARS="${ROOT}/SDQ-LARS"
OUT="${LARS}/runs/${RUN_NAME}"
mkdir -p "${OUT}"
for DATASET in Beauty Sports Toys; do
  DS="Amazon2014${DATASET}_550_LOU"
  SID="${ROOT}/SDQ-403/logs/SDQ/${DS}/vae/sid_vocab.json"
  DESC="LARS-${RUN_NAME}-T5"
  LOG="${LARS}/logs/${DESC}/${DS}/t5"
  DOUT="${OUT}/${DATASET}"
  mkdir -p "${DOUT}"
  cat > "${DOUT}/status.json" <<JSON
{"dataset":"${DATASET}","variant":"cp_ug_t5","protocol":"TIGER-fast","status":"training_t5","gpu":${GPU},"started":"$(date -Is)","output":"${DOUT}","beam":30,"fp32":true,"train_batch_size":512,"eval_batch_size":96,"constrained_beam":false,"num_workers":0,"seed":2025,"epochs":200,"prefix_loss_weight":0.5,"sid_vocab":"${SID}"}
JSON
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/SDQ-403/train_t5_fp32.py" \
    --config "${ROOT}/SDQ-403/configs/t5/${DS}.yaml" \
    --root "${ROOT}/SDQ-403/data" --sid-vocab-file "${SID}" \
    --description "${DESC}" --id t5 --device 0 --num-workers 0 \
    --num-beams 30 --batch-size 512 --eval-batch-size 96 --epochs 200 \
    --seed 2025 --apply-constrained-beam-search False --prefix-loss-weight 0.50 \
    --log2file --log2console > "${DOUT}/t5.log" 2>&1
  "${PYTHON}" "${LARS}/finalize_t5_result.py" "${LOG}"
  cat > "${DOUT}/status.json" <<JSON
{"dataset":"${DATASET}","variant":"cp_ug_t5","protocol":"TIGER-fast","status":"complete","gpu":${GPU},"finished":"$(date -Is)","output":"${DOUT}","beam":30,"fp32":true,"train_batch_size":512,"eval_batch_size":96,"constrained_beam":false,"num_workers":0,"seed":2025,"epochs":200,"prefix_loss_weight":0.5,"sid_vocab":"${SID}"}
JSON
done
echo "[$(date -Is)] ALL_DONE ${RUN_NAME}"
