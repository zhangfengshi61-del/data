#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
PY="/data/fszhang/anaconda/envs/myenv/bin/python"
LOG="$ROOT/baseline_runs/sequence.log"
SMOKE=0
if [[ "${1:-}" == "--smoke" ]]; then SMOKE=1; fi
mkdir -p "$ROOT/baseline_runs" "$ROOT/baseline_runs/Latte/logs" "$ROOT/baseline_runs/UniGRec/logs" "$ROOT/baseline_runs/Pctx/logs"
exec > >(tee -a "$LOG") 2>&1

wait_for_method() {
  local needle="$1"
  while pgrep -f "$needle" >/dev/null 2>&1; do
    echo "[$(date -Is)] waiting for $needle"
    sleep 60
  done
}

run_parallel() {
  local method="$1"; shift
  local pids=()
  for spec in "$@"; do
    IFS='|' read -r name gpu command <<< "$spec"
    echo "[$(date -Is)] start $method/$name on GPU $gpu"
    bash -lc "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu $command" > "$ROOT/baseline_runs/$method/logs/${name}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date -Is)] $method failed; stopping sequence"
    exit 1
  fi
  echo "[$(date -Is)] $method completed"
}

echo "[$(date -Is)] sequence started"
if [[ "$SMOKE" -eq 1 ]]; then
  COFI_ARGS="1 2"
  LATTE_ARGS="--epochs 1 --eval_interval 1 --patience 1 --train_batch_size 8 --eval_batch_size 8 --num_beams 2 --smoke_samples 8"
  UNI_ARGS="1 1"
  PCTX_ARGS="1 1"
else
  COFI_ARGS="200 30"
  LATTE_ARGS=""
  UNI_ARGS="3000 200"
  PCTX_ARGS="50 150"
fi
# CoFiRec was already executed and its test artifacts are present under
# baseline_runs/CoFiRec; continue the serial queue with the next method.
echo "[$(date -Is)] CoFiRec already present; skipping to Latte"

run_parallel Latte \
  "Beauty|0|$PY Latte/run_local.py --category Beauty --gpu_id 0 --num_beams 20 --data_root $ROOT/data/processed $LATTE_ARGS" \
  "Sports|2|$PY Latte/run_local.py --category Sports --gpu_id 2 --num_beams 20 --data_root $ROOT/data/processed $LATTE_ARGS" \
  "Toys|3|$PY Latte/run_local.py --category Toys --gpu_id 3 --num_beams 20 --data_root $ROOT/data/processed $LATTE_ARGS"

run_parallel UniGRec \
  "Beauty|0|SMOKE=$SMOKE bash UniGRec/run_local.sh Beauty 0 $UNI_ARGS" \
  "Sports|2|SMOKE=$SMOKE bash UniGRec/run_local.sh Sports 2 $UNI_ARGS" \
  "Toys|3|SMOKE=$SMOKE bash UniGRec/run_local.sh Toys 3 $UNI_ARGS"

run_parallel Pctx \
  "Beauty|0|bash Pctx/run_local.sh Beauty 0 $PCTX_ARGS" \
  "Sports|2|bash Pctx/run_local.sh Sports 2 $PCTX_ARGS" \
  "Toys|3|bash Pctx/run_local.sh Toys 3 $PCTX_ARGS"

echo "[$(date -Is)] all baselines completed"
