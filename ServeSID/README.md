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

# 2. run Full ServeSID (the SAME sid_mlp implementation for all backbones)
# Set GRAPH_DIR, MODEL_DIR, BEAM_DUMP to your real, corresponding assets.
# Default SEEDS="42 43 44". The launcher reads this checkout's
# configs/servesid_final.yaml and saves best-Val checkpoints.
GRAPH_DIR="$GRAPH_DIR" MODEL_DIR="$MODEL_DIR" BEAM_DUMP="$BEAM_DUMP" \
    bash scripts/train.sh.servesid sidmlp Beauty 0
ARCH=gru GRAPH_DIR="$GRAPH_DIR" MODEL_DIR="$MODEL_DIR" BEAM_DUMP="$BEAM_DUMP" \
    bash scripts/train.sh.servesid sidmlp Beauty 0
ARCH=cnn GRAPH_DIR="$GRAPH_DIR" MODEL_DIR="$MODEL_DIR" BEAM_DUMP="$BEAM_DUMP" \
    bash scripts/train.sh.servesid sidmlp Beauty 0

# Base-KD examples use configs/sidmlp_gru.yaml or sidmlp_cnn.yaml instead
# of the Full loss. They are experiment arms, not separate code versions.

# 3. evaluate the best-Val checkpoint with constrained beam search
bash scripts/infer.sh
```

See `scripts/*.sh` and `configs/*.yaml` for dataset paths and hyperparameters
(candidate support `M`, training cutoff `K_tr`, pair range `B`, loss weights
`lambda_H`, `lambda_D`, `lambda_corr`, learning rate, and seeds).
The paper-aligned model definitions have **1,663,744 MLP**,
**2,304,000 GRU**, and **1,744,896 CNN** trainable parameters, excluding
the shared frozen encoder and SID embedding. Re-check these counts with
`PYTHONPATH=. python scripts/count_student_parameters.py` from `ServeSID/`.
Different `--arch` values select backbones within the **same** codebase,
not separate implementation versions.

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

## Reproducible validation checkpointing and multi-backbone KD baselines

Both `sid_mlp.cli.train` and `sid_mlp.kd_baselines` now select the
`*_best_val_ndcg10.pth` checkpoint from **complete validation** ranking
evaluations (every five epochs plus the last epoch). The validation loader
**never falls back to test**. If a validation split is absent or empty,
training fails instead of reporting a test-selected result. If
`--beam_val_max_batches` is supplied outside the debug `--test` mode,
training also fails; debug-selected artifacts are explicitly marked
`selection_scope=debug_subset` and stored with a `_debug` suffix.
Use the Best-Val checkpoint, not `*_final.pth`, to evaluate a
reportable result. Each best checkpoint records its epoch, NDCG@10
value, validation request count, selected split, seed, backbone
architecture and corresponding hyperparameters.

The external KD runner supports all three backbone families for
`rd`, `rrd`, `rcek`, `lohrec`, `gkd`, `distillm2`, and `smartgr`.
Its `--arch` choice uses the same `SIDMLP`, `SIDGRU` and
`SIDCNN` classes as Full training, and its checkpoints load via
`sid_mlp.inference.build_sidmlp_from_checkpoint`. GKD and
DistiLLM-2 share the same on-policy context-caching interface across
the three backbones.

To run a single baseline:

```bash
python -m sid_mlp.kd_baselines --method rrd --arch gru \
  --checkpoint "$TEACHER_CKPT" --sem_ids "$SEM_IDS" \
  --dataset Amazon2014 --category Beauty \
  --dataset_tag Amazon2014_Beauty \
  --val_dataset_tag Amazon2014_Beauty_val \
  --graph_dir "$GRAPH_DIR" --beam_dump "$BEAM_DUMP" \
  --seed 42 --epochs 200 --beam_size 50
```

To sequentially run seven baseline methods on all architectures and
seeds 42/43/44, set the above path variables plus `CATEGORY`,
`DATASET_TAG`, and `VAL_DATASET_TAG`, then use
`bash scripts/train_kd_baselines_all_arches.sh`. The script uses
consistent beam and validation settings and names every checkpoint
by method, architecture, hyperparameters, and seed. It does not
change any previously reported metrics.

The PyTorch CPU CI smoke test trains a synthetic batch through each
architecture and verifies strict, identical checkpoint reload by the
real inference loader. Full-scale reproduction of the published
tables still requires the authors' datasets, teacher dumps, actual
configurations, original seeds, and GPU checkpoints; these tests do
not establish that historical paper scores were generated with the
newly corrected Full Loss or selected checkpoints. In particular,
published GRU parameter counts must be re-checked against the exact
training configuration/checkpoint.

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

The author clarified that the J4 Beauty training-cost audit used
**M=64**, not M=40; the earlier 40-candidate statement was a reporting
error. The same report independently specifies **B=40** and an
historical **cache-export beam width of 40**. No timing or memory
numbers were modified. Original logs and cache-width metadata have
not been independently audited, so reproducibility still requires
checking the `BENCH_COST=1` run against the intended M=64 dump.
The older Scientific (2023) HServe-only M-sweep and
HServe+Plain-Pair B-sweep used different objectives; those results
must not be labeled as the present Full-loss sensitivity test.
The full provenance is in `verification/J4_COST_AND_BM_PROVENANCE.md`.

## Acknowledgments

This code is built on top of the open-source
[SID-MLP](https://github.com/ztguo715/SID-MLP) repository and the
GenRec (TIGER) teacher pipeline.


### Implementation integrity checks

The maintained implementation is on **`main`**.
`configs/servesid_final.yaml` sets the paper Full loss to
digit CE + 4 HServe + 31.42 DA-CDRS, with no separate gold-prefix
token KL or direct Pair Loss. The detached CF evaluator is
`softplus(0.05 - G)`. HServe + Plain Pair requires
`--da_w 0 --pair_w 1 --pair_margin 0.05`.

DistiLLM-2 student candidate scores now stay in the autograd graph:
only sampled SID identities, teacher scoring and comparison labels are
detached. `tests/test_distillm2_cpu.py` checks that the contrastive
term actually changes student parameter gradients. On-policy
teacher conditionals for GKD and DistiLLM-2 are normalized over their
corresponding 256-token SID codebooks; the test also checks the
distribution sums and zero JSD under identical codebook conditionals.

**Empirical scope:** these are real implementation corrections.
Historical GKD/DistiLLM-2 table entries have not been re-evaluated on
the corrected implementation. If those entries were generated from
the prior buggy functions, rerun the affected baselines (using
`scripts/train_kd_baselines_all_arches.sh`) before treating those
figures as matched comparisons. No other methods require a rerun
*solely* due to this baseline-specific change.

The beam-dump loader now refuses `list_topm` larger than the
physical dump width, checks valid counts and `step_logp` shape, and
reports the candidate-dependent remainder between cached complete
teacher scores and the sum of four SID log-probabilities. This
remainder may include EOS. The loader does not silently reinterpret
historical scores. True request identity alignment and actual
candidate availability still require the original cache metadata.
