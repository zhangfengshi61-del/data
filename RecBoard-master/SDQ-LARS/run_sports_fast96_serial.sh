#!/usr/bin/env bash
set -u
ROOT=/data/fszhang/RecBoard-master/SDQ-LARS
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
CFG=/data/fszhang/RecBoard-master/SDQ-403/configs/t5/Amazon2014Sports_550_LOU.yaml
FAST=/data/fszhang/RecBoard-master/SDQ-403/train_t5_fp32.py
RUN=$ROOT/runs/20260926_sports_fast96
cd "$ROOT"
run_one() {
  local variant="$1"
  local vocab="$2"
  local desc="$3"
  local out="$RUN/$variant"
  mkdir -p "$out"
  cat > "$out/status.json" <<EOF
{"dataset":"Sports","variant":"$variant","protocol":"TIGER-fast","gpu":2,"status":"training_t5","started":"$(date -Is)","output":"$out","beam":30,"fp32":true,"train_batch_size":512,"eval_batch_size":96,"constrained_beam":false,"num_workers":0,"seed":2025,"epochs":200}
EOF
  rc=0
  env CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u "$FAST" \
    --config "$CFG" \
    --root /data/fszhang/RecBoard-master/SDQ-403/data \
    --sid-vocab-file "$vocab" \
    --description "$desc" \
    --id t5 \
    --device 0 \
    --num-workers 0 \
    --num-beams 30 \
    --eval-batch-size 96 \
    --batch-size 512 \
    --epochs 200 \
    --seed 2025 \
    --apply-constrained-beam-search False \
    --log2file --log2console > "$out/t5.log" 2>&1 || rc=$?
  if [ "$rc" -eq 0 ]; then state=complete; else state=failed; fi
  printf '{"dataset":"Sports","variant":"%s","protocol":"TIGER-fast","gpu":2,"status":"%s","return_code":%s,"finished":"%s","output":"%s","beam":30,"fp32":true,"train_batch_size":512,"eval_batch_size":96,"constrained_beam":false,"num_workers":0,"seed":2025,"epochs":200}
' "$variant" "$state" "$rc" "$(date -Is)" "$out" > "$out/status.json"
  return "$rc"
}
run_one gated_margin /data/fszhang/RecBoard-master/SDQ-LARS/logs/LARS-20260923_gated_sports-gated_margin/Amazon2014Sports_550_LOU/vae/sid_vocab.json LARS-20260926_sports_fast96-gated_margin-T5
run_one trust_region /data/fszhang/RecBoard-master/SDQ-LARS/logs/LARS-20260923_trust_div-trust_region/Amazon2014Sports_550_LOU/vae/sid_vocab.json LARS-20260926_sports_fast96-trust_region-T5
