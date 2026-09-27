#!/bin/bash
# Launch the three protocol evals and exit.
cd /data/fszhang/RecBoard-master/HiFlow-SID
for pair in "Beauty:0" "Sports:1" "Toys:3"; do
  ds=${pair%%:*}; gpu=${pair##*:}
  setsid bash -c "CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 /data/fszhang/anaconda/envs/myenv_t5/bin/python scripts/eval_t5_protocol.py $ds 0 > logs/eval_t5_protocol_${ds}.log 2>&1" < /dev/null &
done
disown -a
echo "launched protocol evals"
