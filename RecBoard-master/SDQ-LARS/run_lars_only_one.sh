#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:?dataset}"; GPU="${2:?gpu}"; RUN_NAME="${3:?run_name}"
PYTHON="/data/fszhang/anaconda/envs/myenv_t5/bin/python"
ROOT="/data/fszhang/RecBoard-master"; LARS="${ROOT}/SDQ-LARS"
OUT="${LARS}/runs/${RUN_NAME}/${DATASET}"; VAE_OUT="${OUT}/vae"
DS="Amazon2014${DATASET}_550_LOU"; SID="${VAE_OUT}/sid_vocab.json"
DESC="LARS-${RUN_NAME}-LARS-only"; T5DESC="${DESC}-T5"
LOG="${LARS}/logs/${T5DESC}/${DS}/t5"
mkdir -p "${OUT}"
cat > "${OUT}/status.json" <<JSON
{"dataset":"${DATASET}","variant":"lars_only_hrq","protocol":"TIGER-fast","status":"training_vae","gpu":${GPU},"started":"$(date -Is)","beam":30,"fp32":true,"train_batch_size":512,"eval_batch_size":96,"seed":2025,"vae_epochs":100,"t5_epochs":200,"constrained_beam":false,"sid_source":"fresh_lars_only_hrq","sdq_imports":false}
JSON
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${LARS}/train_lars_only_vae.py" \
  --root "${ROOT}/SDQ-403/data" --dataset "${DS}" --output "${VAE_OUT}" \
  --device cuda:0 --epochs 100 --batch-size 512 --seed 2025 \
  > "${OUT}/vae.log" 2>&1
python3 - <<PY2
import json
p="${OUT}/status.json"; z=json.load(open(p)); z.update(status="training_t5", stage="t5", sid_vocab="${SID}"); json.dump(z,open(p,"w"),indent=2)
PY2
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${ROOT}/SDQ-403/train_t5_fp32.py" \
  --config "${ROOT}/SDQ-403/configs/t5/${DS}.yaml" --root "${ROOT}/SDQ-403/data" \
  --sid-vocab-file "${SID}" --description "${T5DESC}" --id t5 --device 0 --num-workers 0 \
  --num-beams 30 --batch-size 512 --eval-batch-size 96 --epochs 200 --seed 2025 \
  --apply-constrained-beam-search False --prefix-loss-weight 0.0 \
  --log2file --log2console > "${OUT}/t5.log" 2>&1
"${PYTHON}" "${LARS}/finalize_t5_result.py" "${LOG}"
python3 - <<PY2
import json
p="${OUT}/status.json"; z=json.load(open(p)); z.update(status="complete", finished="$(date -Is)"); json.dump(z,open(p,"w"),indent=2)
PY2
