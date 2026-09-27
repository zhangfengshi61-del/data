#!/bin/bash
# Orchestrate the RecBoard-master baseline reproduction.
#
#   Methods run SERIALLY:  Latte -> UniGRec -> Pctx
#   Within a method, the three datasets run in PARALLEL (one GPU each).
#
# Data / rules: /data/fszhang/RecBoard-master/data/processed (canonical
# Amazon-2014 leave-two-out protocol), metrics Recall/NDCG/MRR@5/10/20.
#
# Env overrides:
#   FAST_PY, OLD_PY, GPUS, LATTE_EPOCHS, RQVAE_EPOCHS, JOINT_EPOCHS,
#   DUOREC_EPOCHS, GR_EPOCHS, METHODS
set -u

ROOT=/data/fszhang/RecBoard-master
FAST_PY=${FAST_PY:-/data/fszhang/anaconda/envs/myenv_t5/bin/python}
OLD_PY=${OLD_PY:-/data/fszhang/anaconda/envs/myenv/bin/python}
RUN=$ROOT/baseline_runs

DATASETS=(Beauty Sports Toys)
read -r -a GPUS <<< "${GPUS:-1 2 3}"

LATTE_EPOCHS=${LATTE_EPOCHS:-150}
RQVAE_EPOCHS=${RQVAE_EPOCHS:-3000}
JOINT_EPOCHS=${JOINT_EPOCHS:-200}
DUOREC_EPOCHS=${DUOREC_EPOCHS:-50}
GR_EPOCHS=${GR_EPOCHS:-150}
METHODS=${METHODS:-"Latte UniGRec Pctx"}

mkdir -p "$RUN/Latte/logs" "$RUN/UniGRec/logs" "$RUN/Pctx/logs"

run_latte() {
  local ds=$1 gpu=$2
  cd "$ROOT/Latte"
  "$FAST_PY" run_local.py \
    --category "$ds" --gpu_id "$gpu" --num_beams 20 \
    --data_root "$ROOT/data/processed" \
    --epochs "$LATTE_EPOCHS" --eval_interval 1 --patience 50
}

run_unigrec() {
  local ds=$1 gpu=$2
  cd "$ROOT/UniGRec"
  PY="$FAST_PY" bash run_local.sh "$ds" "$gpu" "$RQVAE_EPOCHS" "$JOINT_EPOCHS"
}

run_pctx() {
  local ds=$1 gpu=$2
  cd "$ROOT/Pctx"
  PY="$FAST_PY" EVAL_BATCH_SIZE="${PCTX_EVAL_BATCH:-32}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash run_local.sh "$ds" "$gpu" "$DUOREC_EPOCHS" "$GR_EPOCHS"
}

run_method_parallel() {
  local method=$1 fn=$2
  local pids=()
  for i in 0 1 2; do
    local ds=${DATASETS[$i]} gpu=${GPUS[$i]}
    ( "$fn" "$ds" "$gpu" ) > "$RUN/$method/orchestrator_${ds}.out" 2>&1 &
    pids+=($!)
    echo "[$method] launched $ds on GPU $gpu (pid ${pids[-1]}) at $(date +%H:%M:%S)"
  done
  local rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  return $rc
}

for m in $METHODS; do
  echo "=================================================="
  echo "[ORCH] $m START $(date)"
  echo "=================================================="
  case "$m" in
    Latte)   fn=run_latte ;;
    UniGRec) fn=run_unigrec ;;
    Pctx)    fn=run_pctx ;;
    *) echo "[ORCH] unknown method $m"; continue ;;
  esac
  if run_method_parallel "$m" "$fn"; then
    echo "[ORCH] $m DONE $(date)"
  else
    echo "[ORCH] $m FINISHED WITH ERRORS $(date)"
  fi
done
echo "[ORCH] ALL DONE $(date)"
