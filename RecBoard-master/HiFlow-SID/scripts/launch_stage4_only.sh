#!/bin/bash
set -u
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
HIF=/data/fszhang/RecBoard-master/HiFlow-SID
SDQ=/data/fszhang/RecBoard-master/SDQ-403
declare -A GPU_SIDMLP=( [Beauty]=0 [Sports]=1 [Toys]=3 )
for DS in Beauty Sports Toys; do
  VOCAB=$HIF/logs/sid_vocab_rqvae_${DS}.json
  T5_BEST=$SDQ/logs/TIGER-T5/Amazon2014${DS}_550_LOU/t5-rq-fast/best.pt
  while [ ! -f "$T5_BEST" ] || pgrep -f "train_t5_rq.py --config configs/t5/Amazon2014${DS}" > /dev/null; do sleep 300; done
  echo "[stage4] $DS: launching SID-MLP (RQ) ..."
  setsid bash -c "cd $HIF && CUDA_VISIBLE_DEVICES=${GPU_SIDMLP[$DS]} PYTHONUNBUFFERED=1 $PY train_sidmlp.py \
    --category $DS --dataset Amazon2014${DS}_550_LOU \
    --sid-vocab-file $VOCAB \
    --teacher-checkpoint $T5_BEST \
    --id sidmlp-rq-${DS} --epochs 200 --early-stop-patience 15 --seed 2025 --device 0 \
    > $HIF/logs/stage4_sidmlp_rq_${DS}.log 2>&1" < /dev/null & disown
done
echo "[stage4] watcher done"
