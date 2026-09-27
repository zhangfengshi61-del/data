#!/bin/bash
# Stage 2+3 launcher: after RQ-VAE finishes, copy final sid_vocab and launch
# TIGER-T5 teacher (Stage2) + MTP (Stage3) per dataset on RQ-VAE SIDs.
set -euo pipefail

PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
TIGER_DIR=/data/fszhang/RecBoard-master/TIGER
HIF=/data/fszhang/RecBoard-master/HiFlow-SID
SDQ=/data/fszhang/RecBoard-master/SDQ-403

declare -A GPU_T5=( [Beauty]=0 [Sports]=1 [Toys]=3 )
declare -A GPU_MTP=( [Beauty]=1 [Sports]=3 [Toys]=0 )

# wait for rqvae processes to exit
while pgrep -f "train_rqvae.py" > /dev/null; do sleep 60; done

for DS in Beauty Sports Toys; do
  SRC=$TIGER_DIR/logs/RQVAE/Amazon2014${DS}_550_LOU/rqvae-${DS}/sid_vocab.json
  DST=$HIF/logs/sid_vocab_rqvae_${DS}.json
  cp "$SRC" "$DST"
  echo "[stage] $DS RQ sid_vocab -> $DST"

  # Stage 2: TIGER-T5 teacher on RQ SIDs
  # Same settings as the protocol SDQ-T5 baseline: eval_freq default 5,
  # beam 30, 200 epochs, best by val NDCG@10, seed 2025.
  setsid bash -c "cd $SDQ && CUDA_VISIBLE_DEVICES=${GPU_T5[$DS]} PYTHONUNBUFFERED=1 $PY train_t5.py \
    --config configs/t5/Amazon2014${DS}_550_LOU.yaml \
    --root $SDQ/data \
    --sid-vocab-file $DST \
    --id t5-rq --device 0 --num-beams 30 --epochs 200 --seed 2025 \
    > $HIF/logs/stage2_t5_rq_${DS}.log 2>&1" < /dev/null & disown
  echo "[stage] launched T5 teacher $DS on GPU ${GPU_T5[$DS]}"

  # Stage 3: MTP on RQ SIDs
  setsid bash -c "cd $HIF && CUDA_VISIBLE_DEVICES=${GPU_MTP[$DS]} PYTHONUNBUFFERED=1 $PY train_mtp.py \
    --category $DS --dataset Amazon2014${DS}_550_LOU \
    --sid-vocab-file $DST \
    --id mtp-rq --epochs 100 --seed 2025 --device 0 \
    > $HIF/logs/stage3_mtp_rq_${DS}.log 2>&1" < /dev/null & disown
  echo "[stage] launched MTP $DS on GPU ${GPU_MTP[$DS]}"
done
echo "[stage] Stage2+3 launched"
