# RecLARS-SDQ

Recommendation-aware trust-region sparse direction learning for SDQ. The core
remains the LARS inner direction/coefficient path and outer dictionary update;
the candidate version additionally warm-starts from the final SDQ checkpoint,
anchors the encoder/codebook/SID basin, and uses a transition-aware soft SID
loss. The original SDQ hard forward path, 3×256 semantic code budget, and T5
script are reused.

完整中文方案：[方向 5 成熟方案](../我的方向/5/成熟方案.md)。

当前实验、参数、结果和暂停状态：[实验记录_20260924](实验记录_20260924.md)。

First round `20260922_r1` runs Beauty/Sports/Toys main on GPUs 1/2/3,
and Beauty no_bridge on GPU 0. Each worker trains the tokenizer for 100 epochs,
then automatically runs the original T5 for 200 epochs, beam=30, seed=2025.

```bash
cd /data/fszhang/RecBoard-master/SDQ-LARS
/data/fszhang/anaconda/envs/myenv_t5/bin/python status.py --run-name 20260922_r1
```

Run artifacts live in `runs/20260922_r1/`; completed workers write `result.json`.
`status.py` collects completed results into `RESULTS.md`. Models live in `logs/`.
Baseline sources and existing runs are not overwritten.

The formal trust-region candidate is launched with:

```bash
/data/fszhang/anaconda/envs/myenv_t5/bin/python run_experiment.py \
  --dataset Toys --variant trust_region --gpu 3 --run-name 20260923_trust
```

It uses the original SDQ final `model.pt` and `sid_vocab.json` as the anchor,
then keeps the original 100-epoch VAE and 200-epoch T5 protocol.

For a new run, use a new name (existing directories are rejected):

```bash
/data/fszhang/anaconda/envs/myenv_t5/bin/python launch.py --run-name NEW_NAME
```

Verify the numerical implementation:

```bash
OMP_NUM_THREADS=4 /data/fszhang/anaconda/envs/myenv_t5/bin/python -m unittest test_lars -v
```

This is truncated **LAR**, not a full LARS-Lasso implementation. No sparse
coefficients are transmitted to T5. Additional collision tokens follow the
baseline converter exactly. Final effectiveness is unknown until full runs finish.
