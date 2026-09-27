#!/bin/bash
# Queue the SDQ-403 reproduction after the currently running serial baseline
# pipeline (run_all_serial.sh: Latte -> UniGRec -> Pctx) finishes.
#
#   Methods run SERIALLY w.r.t. the existing pipeline, and the three datasets
#   run in PARALLEL (one GPU each) exactly like run_all_serial.sh.
#
# Env overrides:
#   PY, GPUS, SDQ_VARIANT, SDQ_EPOCHS, T5_EPOCHS, NUM_BEAMS
set -u

ROOT=/data/fszhang/RecBoard-master
PY=${PY:-/data/fszhang/anaconda/envs/myenv_t5/bin/python}
RUN=$ROOT/baseline_runs/SDQ-403
LOG=$RUN/orchestrator_master.log
WATCH=${WATCH:-run_all_serial.sh}

SDQ_VARIANT=${SDQ_VARIANT:-vae}
SDQ_EPOCHS=${SDQ_EPOCHS:-100}
T5_EPOCHS=${T5_EPOCHS:-200}
NUM_BEAMS=${NUM_BEAMS:-30}

DATASETS=(Beauty Sports Toys)
read -r -a GPUS <<< "${GPUS:-1 2 3}"

mkdir -p "$RUN/logs"
echo "[ORCH-SDQ] $(date -Is) queued; waiting for '$WATCH' to finish" >> "$LOG"

while pgrep -f "bash $WATCH" >/dev/null 2>&1; do
  sleep 60
done

echo "[ORCH-SDQ] $(date -Is) pipeline free; SDQ START" >> "$LOG"

pids=()
for i in 0 1 2; do
  ds=${DATASETS[$i]}
  gpu=${GPUS[$i]}
  (
    cd "$ROOT" &&
    PY="$PY" NUM_BEAMS="$NUM_BEAMS" \
      bash SDQ-403/run_local.sh "$ds" "$gpu" "$SDQ_VARIANT" "$SDQ_EPOCHS" "$T5_EPOCHS"
  ) > "$RUN/orchestrator_${ds}.out" 2>&1 &
  pids+=($!)
  echo "[ORCH-SDQ] $(date -Is) launched $ds on GPU $gpu (pid ${pids[-1]})" >> "$LOG"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done

if [[ "$rc" -eq 0 ]]; then
  echo "[ORCH-SDQ] $(date -Is) SDQ DONE" >> "$LOG"
else
  echo "[ORCH-SDQ] $(date -Is) SDQ FINISHED WITH ERRORS" >> "$LOG"
fi
