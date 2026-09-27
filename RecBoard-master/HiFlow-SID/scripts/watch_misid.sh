#!/usr/bin/env bash
set -u
CAT="$1"
GPU="$2"
DS="Amazon2014""$CAT""_550_LOU"
HIF=/data/fszhang/RecBoard-master/HiFlow-SID
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
LOG="$HIF/logs/misid_rq_""$CAT"".log"
RESULT="$HIF/results/misid_rq_""$CAT"".json"
while pgrep -f "[t]rain_misid.py --category $CAT" >/dev/null; do
  sleep 300
done
if [ -f "$RESULT" ] && grep -q '"accepted": true' "$RESULT"; then
  echo "[watch] $CAT already has an accepted result; keep it" >> "$LOG"
  exit 0
fi
CKPT="$HIF/logs/MI-SID/$DS/misid-rq-""$CAT""/best.pt"
if [ -f "$CKPT" ]; then
  echo "[watch] $CAT training finished; evaluating $CKPT" >> "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false \
    "$PY" "$HIF/scripts/eval_misid_protocol.py" "$CAT" 0 >> "$LOG" 2>&1
else
  echo "[watch] $CAT finished without checkpoint: $CKPT" >> "$LOG"
fi
