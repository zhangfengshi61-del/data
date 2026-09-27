#!/usr/bin/env bash
set -u
cd /data/fszhang/RecBoard-master/SDQ-LARS
PY=/data/fszhang/anaconda/envs/myenv_t5/bin/python
launch_one() {
  DS="$1"
  OUT="/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260924_ug_hlars_$DS.launcher.log"
  CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "$PY" -u run_experiment.py --dataset "$DS" --variant utility_hier --gpu 2 --run-name 20260924_ug_hlars \
    > "$OUT" 2>&1
  echo "$DS:$?" >> "/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260924_ug_hlars.exit"
}
mkdir -p "/data/fszhang/RecBoard-master/SDQ-LARS/runs"
: > "/data/fszhang/RecBoard-master/SDQ-LARS/runs/20260924_ug_hlars.exit"
launch_one Beauty &
P1=$!
launch_one Sports &
P2=$!
launch_one Toys &
P3=$!
wait $P1
wait $P2
wait $P3

