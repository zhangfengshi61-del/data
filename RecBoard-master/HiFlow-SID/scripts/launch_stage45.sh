#!/bin/bash
# Stage 4+5 watcher: per dataset, wait for Stage2 (T5 teacher) and Stage3 (MTP)
# to finish, then launch SID-MLP (Stage4, frozen teacher) and ExpertScore v3
# (Stage5, MTP init) on the same RQ-VAE SIDs.
set -u

PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
HIF=/data/fszhang/RecBoard-master/HiFlow-SID
SDQ=/data/fszhang/RecBoard-master/SDQ-403

declare -A GPU_SIDMLP=( [Beauty]=0 [Sports]=1 [Toys]=3 )
declare -A GPU_EXPERT=( [Beauty]=1 [Sports]=3 [Toys]=0 )

for DS in Beauty Sports Toys; do
  VOCAB=$HIF/logs/sid_vocab_rqvae_${DS}.json
  T5_BEST=$SDQ/logs/TIGER-T5/Amazon2014${DS}_550_LOU/t5-rq/best.pt
  MTP_BEST=$HIF/logs/HiFlow-SID/Amazon2014${DS}_550_LOU/mtp-rq/best.pt

  # Stage 5 (ExpertScore) only needs the MTP checkpoint; launch as soon as
  # the MTP process for this dataset has exited.
  while pgrep -f "train_mtp.py --category ${DS}" > /dev/null; do sleep 60; done
  if [ -f "$MTP_BEST" ]; then
    echo "[stage5] $DS: launching ExpertScore v3 (RQ) ..."
    setsid bash -c "cd $HIF && CUDA_VISIBLE_DEVICES=${GPU_EXPERT[$DS]} PYTHONUNBUFFERED=1 $PY train_hiflow.py \
      --category $DS --dataset Amazon2014${DS}_550_LOU \
      --sid-vocab-file $VOCAB \
      --mtp-checkpoint $MTP_BEST \
      --id pisid-expertscore-rq-${DS} --epochs 50 --eval-freq 5 --batch-size 512 --seed 2025 \
      --fast-encoder True --fast-hidden 256 --fast-layers 2 --fast-max-seq-len 128 \
      --expert-score True --expert-rank 8 --expert-negatives 2048 \
      --lam-expert-rank 1.0 --lam-parallel 1.0 --lam-prefix 1.0 \
      --lam-align 0.2 --tau-align 0.1 \
      --distill-encoder True --lam-distill 0.25 --lam-logit-distill 0.5 \
      --kd-temperature 2.0 --device 0 \
      > $HIF/logs/stage5_expertscore_rq_${DS}.log 2>&1" < /dev/null & disown
  fi

  # Stage 4 (SID-MLP) needs the T5 teacher checkpoint.
  while [ ! -f "$T5_BEST" ] || pgrep -f "train_t5.py --config configs/t5/Amazon2014${DS}" > /dev/null; do sleep 300; done

  echo "[stage4] $DS: launching SID-MLP (RQ) ..."
  setsid bash -c "cd $HIF && CUDA_VISIBLE_DEVICES=${GPU_SIDMLP[$DS]} PYTHONUNBUFFERED=1 $PY train_sidmlp.py \
    --category $DS --dataset Amazon2014${DS}_550_LOU \
    --sid-vocab-file $VOCAB \
    --teacher-checkpoint $T5_BEST \
    --id sidmlp-rq-${DS} --epochs 200 --early-stop-patience 15 --seed 2025 --device 0 \
    > $HIF/logs/stage4_sidmlp_rq_${DS}.log 2>&1" < /dev/null & disown
done
echo "[stage45] watcher finished"
