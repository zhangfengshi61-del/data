#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset}"
GPU="${2:?gpu}"
RUN_NAME="${3:?run_name}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LARS="${ROOT}/SDQ-LARS"
SDQ="${ROOT}/SDQ-403"
PYTHON="${PY:-python}"
DS="Amazon2014${DATASET}_550_LOU"
OUT="${LARS}/runs/${RUN_NAME}/${DATASET}"
DESC="LARS-${RUN_NAME}-StablePrefix-T5"
LOG="${LARS}/logs/${DESC}/${DS}/t5"
SID="${LARS}/logs/LARS-20260922_r1-main/${DS}/vae/sid_vocab.json"
mkdir -p "${OUT}" "${LOG}"

cat > "${OUT}/config.json" <<JSON
{
  "method": "LARS-StablePrefix",
  "dataset": "${DATASET}",
  "run_name": "${RUN_NAME}",
  "protocol": "TIGER-fast",
  "gpu": ${GPU},
  "seed": 2025,
  "t5_epochs": 200,
  "train_batch_size": 512,
  "eval_batch_size": 96,
  "beam": 30,
  "fp32": true,
  "apply_constrained_beam_search": false,
  "prefix_loss_weight": 0.10,
  "prefix_loss_min_weight": 0.02,
  "prefix_loss_decay_epochs": 80,
  "sid_source": "LARS-main-20260922_r1",
  "sid_vocab": "${SID}",
  "vae_reused": true,
  "reranker": false,
  "support_score": false,
  "second_stage": false
}
JSON

cat > "${OUT}/status.json" <<JSON
{"dataset":"${DATASET}","method":"LARS-StablePrefix","status":"training_t5","stage":"t5","gpu":${GPU},"started":"$(date -Is)","config":"${OUT}/config.json"}
JSON

cleanup() {
  rc=$?
  if [ $rc -ne 0 ]; then
    "${PYTHON}" - "${OUT}/status.json" "$rc" <<'PYERR'
import json,sys,datetime
p=sys.argv[1]; rc=sys.argv[2]
try: z=json.load(open(p))
except Exception: z={}
z.update(status="failed", exit_code=int(rc), finished=datetime.datetime.now().astimezone().isoformat())
json.dump(z,open(p,"w"),indent=2)
PYERR
  fi
}
trap cleanup EXIT

cd "$LARS"

if [ ! -f "${SID}" ]; then
  echo "Missing existing LARS SID vocabulary: ${SID}" >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u "${SDQ}/train_t5_fp32.py" \
  --config "${SDQ}/configs/t5/${DS}.yaml" \
  --root "${SDQ}/data" \
  --sid-vocab-file "${SID}" \
  --description "${DESC}" --id t5 --device 0 --num-workers 0 \
  --num-beams 30 --batch-size 512 --eval-batch-size 96 \
  --epochs 200 --seed 2025 --apply-constrained-beam-search False \
  --prefix-loss-weight 0.10 --prefix-loss-min-weight 0.02 \
  --prefix-loss-decay-epochs 80 \
  --log2file --log2console \
  > "${OUT}/t5.log" 2>&1

"${PYTHON}" "${LARS}/finalize_t5_result.py" "${LOG}"
if [ -f "${LOG}/result.json" ]; then
  cp "${LOG}/result.json" "${OUT}/result.json"
fi

"${PYTHON}" - "${OUT}/status.json" "${OUT}/result.json" <<'PYOK'
import json,sys,datetime,os
sp,rp=sys.argv[1],sys.argv[2]
z=json.load(open(sp)); z.update(status="complete", stage="done", finished=datetime.datetime.now().astimezone().isoformat())
if os.path.exists(rp): z["result"]=json.load(open(rp))
json.dump(z,open(sp,"w"),indent=2)
PYOK

