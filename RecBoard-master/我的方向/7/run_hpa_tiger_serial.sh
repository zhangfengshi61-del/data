#!/usr/bin/env bash
set -u

ROOT=/data/fszhang/RecBoard-master
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
GPU=${CUDA_VISIBLE_DEVICES:-1}
DIR7=$(find "$ROOT" -maxdepth 3 -type d -name 7 | head -1)
RUN_ROOT="$DIR7/runs/hpa_tiger_serial_20250925"
mkdir -p "$RUN_ROOT"

DATASETS=(Beauty Sports Toys)
for name in "${DATASETS[@]}"; do
  case "$name" in
    Beauty) DATASET=Amazon2014Beauty_550_LOU ;;
    Sports) DATASET=Amazon2014Sports_550_LOU ;;
    Toys) DATASET=Amazon2014Toys_550_LOU ;;
  esac
  CONFIG="$ROOT/SDQ-403/configs/t5/${DATASET}.yaml"
  SID="$ROOT/SDQ-403/logs/SDQ/${DATASET}/vae/sid_vocab.json"
  LOG="$RUN_ROOT/${name}.log"
  STATUS="$RUN_ROOT/${name}.status"
  date -u +"%Y-%m-%dT%H:%M:%SZ START ${name}" | tee "$STATUS"
  echo "dataset=${DATASET} gpu=${GPU} method=HPA-TIGER" | tee -a "$STATUS"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PY" "$ROOT/SDQ-403/train_t5_hpa.py" \
    --config "$CONFIG" \
    --root "$ROOT/SDQ-403/data" \
    --sid-vocab-file "$SID" \
    --description HPA-TIGER \
    --id hpa_tiger \
    --device 0 \
    --num-beams 30 \
    --eval-batch-size 96 \
    --apply-constrained-beam-search False \
    --epochs 200 \
    --seed 2025 \
    --lambda-prefix 0.25 \
    --lambda-metric 0.05 \
    --prefix-temperature 0.10 \
    > "$LOG" 2>&1
  rc=$?
  date -u +"%Y-%m-%dT%H:%M:%SZ END ${name} rc=${rc}" | tee -a "$STATUS"
done

date -u +"%Y-%m-%dT%H:%M:%SZ ALL_DONE" | tee "$RUN_ROOT/ALL_DONE"
