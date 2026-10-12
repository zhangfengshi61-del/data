# ServeSID: Hierarchy-Aligned and Decision-Attributed Distillation for Semantic-ID Generative Recommendation

This repository contains the training and inference code for **ServeSID**,
a structure-aware knowledge distillation framework for Semantic-ID (SID)
generative recommendation. ServeSID distills a fixed TIGER teacher into
lightweight students (MLP / GRU / CNN) with two complementary,
training-only objectives. The student encoder family is selected with
`--arch {mlp,gru,cnn}`:

- **HServe (Hierarchical Serving-Distribution Alignment)**: aligns teacher
  and student preference distributions over a shared complete-SID candidate
  support and their hierarchically consistent prefix marginals through
  multi-depth KL.
- **DA-CDRS (Decision-Attributed Cross-Depth Relative-Score Distillation)**:
  matches depth-wise relative-score increments with a scale-normalized Huber
  loss. Depth weights are detached and computed from the hypothetical reduction
  in `softplus(mu - G_S)` under single-depth teacher-increment replacement.
  **Full ServeSID does not directly optimize Pair Loss.** Only the separate
  `HServe + Plain Pair` ablation optimizes this Softplus as a direct loss.

Both objectives operate only during training; the deployed student keeps the
original verify-free inference graph (one history-encoding pass, depth-wise
SID prediction heads, valid-prefix masking, and constrained beam search).

## Layout

- `sid_mlp/` — the ServeSID implementation (losses, models, KD baselines,
  beam decoding, training/inference entry points). Built on top of the
  SID-MLP student architecture.
- `genrec/` — the TIGER teacher pipeline (based on GenRec/SID-MLP teacher
  code) used to generate teacher candidates and scores.
- `configs/` — YAML configurations:
  - `sidmlp_gru.yaml` / `sidmlp_cnn.yaml` — base-KD runs for the GRU
    (`gru_hidden=384`) and CNN (`conv_dim=256`) student encoders.
  - `servesid_final.yaml` — full ServeSID (HServe + DA-CDRS).
  - `abl_*.yaml` — ablation arms (d4-only KL, DA-CDRS only, HServe only,
    HServe + plain pair, HServe + uniform attribution).
  - `sidmlp.yaml` / `sidmlp_amazon2014_baseline.yaml` — token-distilled base-KD baselines.
  - `verify_*.yaml` — held-out diagnostic configurations (H1/D2 statistics).
  - `infer.yaml` — beam-search evaluation.
- `scripts/` — shell wrappers for teacher building, distillation training,
  inference, and valid-SID-set construction.
- `main.py` — entry point.

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
```

## Usage

```bash
# 1. build the valid-SID set / teacher cache
bash scripts/build_valid_set.sh
bash scripts/extract.sh

# 2a. train the full ServeSID student (MLP encoder, default)
bash scripts/train.sh.servesid

# 2b. or train a GRU / CNN student encoder
python -m sid_mlp.cli.train sidmlp --config configs/sidmlp_gru.yaml --arch gru \
    --checkpoint $MODEL_DIR/<category>/checkpoint.pth --category <category> \
    --graph_dir $GRAPH_DIR --sem_ids $SEM_IDS
python -m sid_mlp.cli.train sidmlp --config configs/sidmlp_cnn.yaml --arch cnn \
    --checkpoint $MODEL_DIR/<category>/checkpoint.pth --category <category> \
    --graph_dir $GRAPH_DIR --sem_ids $SEM_IDS

# 3. evaluate with constrained beam search
bash scripts/infer.sh
```

See `scripts/*.sh` and `configs/*.yaml` for dataset paths and hyperparameters
(candidate support `M`, training cutoff `K_tr`, pair range `B`, loss weights
`lambda_H`, `lambda_D`, `lambda_corr`, learning rate, and seeds). The GRU
(`gru_hidden`, `gru_layers`) and CNN (`conv_dim`, `conv_kernel`,
`conv_layers`) student sizes are also configurable; with the shared
512-wide prediction heads the defaults give roughly 2.8M and 2.3M
parameters, and smaller `head_hidden` values reduce the total budget.

## Paper-aligned loss configuration

The full paper objective is:

```text
L_full = L_digit_CE + lambda_H * L_HServe
                    + lambda_D * lambda_corr * L_DA_weighted_Huber
```

Use `configs/servesid_final.yaml`:
`loss: ce`, `d4_teacher: ce`, `serve_kl: 4`, `pair_w: 0`,
`da_w: 1`, `da_corr_w: 31.42`, `da_margin: 0.05`,
`corr_mode: cf`, and `list_topm: 64`. This additive margin yields
`softplus(0.05 - G)` **only when calculating detached attribution
weights**; it is not an extra term added to `L_full`.

The separate `configs/abl_hserve_pair.yaml` sets `pair_w: 1`,
`pair_margin: 0.05`, `da_w: 0`. DA-only and uniform-depth ablations
also set `pair_w: 0`. Training rejects a configuration that combines
positive `pair_w` with DA-CDRS; it also rejects token KD in Full.

The same loss configuration works with all three student encoders:
`--arch mlp`, `--arch gru`, or `--arch cnn`. The standalone
`sidmlp_gru.yaml` and `sidmlp_cnn.yaml` are **base-KD examples**,
not Full-loss recipes. Select the Full config and override `--arch`
and corresponding encoder hyperparameters to train Full with GRU/CNN.
The Full YAML uses Amazon Beauty 2014 as an example; other domains need
their own matching dataset tags, semantic IDs, teacher checkpoints and
training-only beam dumps.

**Reproducibility:** correcting the loss does not validate previously
reported checkpoint results or regenerate experiments. Re-evaluation
requires the actual training data, seeds and corresponding checkpoints.

## Cost and M/B audit (J4)

The current Full objective has **M=64 cached candidate slots**,
`K_train=10`, **B=40 outsider cutoff**, and **W=50 online beam**.
The training loader now refuses a teacher dump whose candidate width
is smaller than M: an M=40 cached dump cannot be reported as
paper-configured M=64 merely because `list_topm=64` is set.
For training timers, use `BENCH_COST=1`; logs distinguish
`train_step_sec` (gradient updates only) and
`epoch_total_sec` (includes validation). This permits matched-cost
comparison across Full and Base checkpoints.

To measure the **full student serving pipeline**, with and without
synchronized stage profiling:

```bash
python scripts/bench_infer_cost.py \
  --checkpoint "$MODEL_DIR/Beauty/checkpoint.pth" \
  --lens_path "$STUDENT_CKPT" \
  --dataset Amazon2014 --category Beauty \
  --dataset_tag Amazon2014_Beauty \
  --sem_ids "$SEM_IDS" --graph_dir "$GRAPH_DIR" \
  --batch_size 32 --beam_size 50 --repeats 3 \
  --output_dir "$RESULTS_DIR"
```

The output `benchmark_summary.json` reports uninstrumented
amortized requests/second **separately** from the slower synchronized
per-phase timing. GPU identity and total device memory are read from
CUDA at runtime. `full_pipeline_peak_allocated_gib` includes all
resident model tensors in the current inference process, including the
frozen T5 encoder; it is not a decoder-only memory increment.
The benchmark does not measure teacher autoregressive decoding,
does not infer batch-one p95 latency from throughput, and does not
claim that correcting Full Loss changes its inference graph.

The author's J4 historical training results use **M=40** and therefore
are not treated as M=64 Full measurements. The previous anonymous
Scientific (2023) HServe-only M-sweep and HServe+Plain-Pair B-sweep
used a different objective. Numerical results and their scopes are
preserved in `verification/J4_COST_AND_BM_PROVENANCE.md`; they
must not be relabeled as present-day Full-loss sensitivity tests.

## Acknowledgments

This code is built on top of the open-source
[SID-MLP](https://github.com/ztguo715/SID-MLP) repository and the
GenRec (TIGER) teacher pipeline.
