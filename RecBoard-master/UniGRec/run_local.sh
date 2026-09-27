#!/bin/bash
# UniGRec local-protocol runner (Amazon-2014 Beauty/Sports/Toys).
#
# Stage 1: train the soft RQVAE on the item text embeddings.
# Stage 2: end-to-end joint training with the local train-only SASRec CF teacher.
#
# Usage: bash run_local.sh <Dataset> <gpu> [rqvae_epochs] [joint_epochs]
set -euo pipefail

ROOT="/data/fszhang/RecBoard-master"
UNI="$ROOT/UniGRec"
RUN="$ROOT/baseline_runs/UniGRec"
PY="${PY:-/data/fszhang/anaconda/envs/myenv/bin/python}"

DATASET="${1:?dataset}"
GPU="${2:-1}"
RQVAE_EPOCHS="${3:-3000}"
JOINT_EPOCHS="${4:-200}"
DATASET_DIR="dataset/${DATASET}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  DATASET_DIR="dataset/smoke/${DATASET}"
fi
MAX_COLLISION_ARG=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  MAX_COLLISION_ARG=(--max_collision 1000)
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export SWANLAB_MODE=disabled

cd "$UNI"

echo "[UniGRec] stage 1 RQVAE: dataset=$DATASET gpu=$GPU epochs=$RQVAE_EPOCHS"
(
  cd rqvae
  "$PY" main.py \
    --data_path "../${DATASET_DIR}/item_emb_td.parquet" \
    --ckpt_dir "./ckpt/${DATASET}" \
    --epochs "$RQVAE_EPOCHS" \
    --batch_size 1024 \
    --eval_step 50 \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --device cuda:0 \
    --gpu_ids "$GPU" \
    --vq_type soft \
    --distance l2 \
    --temperature 0.01 \
    --temperature_schedule linear \
    --temperature_final 0.001 \
    --temperature_anneal_start_epoch 0 \
    --temperature_anneal_end_epoch "$RQVAE_EPOCHS" \
    --kmeans_init True \
    --kmeans_iters 100 \
    --norm_type none \
    --diversity_scale 1e-4 \
    2>&1 | tee "$RUN/logs/rqvae_${DATASET}.log"
)
RQVAE_MODEL="rqvae/ckpt/${DATASET}/latest/best_loss_model.pth"

echo "[UniGRec] stage 2 joint training: dataset=$DATASET epochs=$JOINT_EPOCHS"
"$PY" model/train.py \
  --train_data "${DATASET_DIR}/train.parquet" \
  --valid_data "${DATASET_DIR}/valid.parquet" \
  --test_data "${DATASET_DIR}/test.parquet" \
  --item_emb_path "${DATASET_DIR}/item_emb_td.parquet" \
  --rqvae_model_path "$RQVAE_MODEL" \
  --save_dir "$RUN/ckpt/${DATASET}" \
  --batch_size 512 \
  --num_epochs "$JOINT_EPOCHS" \
  --t5_lr 5e-3 \
  --rqvae_lr 2e-7 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --scheduler_type constant_with_warmup \
  --alpha 0.5 \
  --beta 1.0 \
  --gamma 0.01 \
  --delta 0.1 \
  --diversity_weight 0.0 \
  --tau 0.001 \
  --use_sasrec_teacher \
  --sasrec_emb_path "${DATASET_DIR}/cf_emb_sasrec256.parquet" \
  --max_len 20 \
  --beam_size 30 \
  --num_layers 6 \
  --num_decoder_layers 6 \
  --d_model 128 \
  --d_ff 512 \
  --num_heads 4 \
  --d_kv 64 \
  --dropout_rate 0.1 \
  --patience 15 \
  --early_stop_metric "ndcg@10" \
  --use_code_offset \
  --enable_finetune \
  --finetune_lr 4e-4 \
  --finetune_epochs 100 \
  --finetune_patience 10 \
  --finetune_early_stop_metric "ndcg@10" \
  --finetune_warmup_ratio 0.05 \
  --finetune_scheduler_type constant \
  "${MAX_COLLISION_ARG[@]}" \
  2>&1 | tee "$RUN/logs/joint_${DATASET}.log"
