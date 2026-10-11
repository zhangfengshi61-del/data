# ServeSID: Hierarchy-Aligned and Direction-Aware Distillation for Semantic-ID Generative Recommendation

This repository contains the training and inference code for **ServeSID**,
a structure-aware knowledge distillation framework for Semantic-ID (SID)
generative recommendation. ServeSID distills a fixed TIGER teacher into
lightweight students (MLP / GRU / CNN) with two complementary,
training-only objectives:

- **HServe (Hierarchical Serving-Distribution Alignment)**: aligns teacher
  and student preference distributions over a shared complete-SID candidate
  support and their hierarchically consistent prefix marginals through
  multi-depth KL.
- **DA-CDRS (Decision-Attributed Cross-Depth Relative-Score Distillation)**:
  distills teacher-ordered cross-cutoff candidate pairs by decomposing their
  score gaps into depth-wise relative-score increments and weighting
  scale-normalized Huber correction by detached, counterfactual decision-impact
  weights computed with softplus(mu - G), mu=0.05. Full ServeSID does **not**
  optimize an additional pairwise ranking loss; that objective exists only
  as the separate HServe + Plain Pair ablation.

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

# 2. train the full ServeSID student
# supply GRAPH_DIR, MODEL_DIR and a matching TRAIN-only BEAM_DUMP
GRAPH_DIR=/data/graph MODEL_DIR=/data/models BEAM_DUMP=/data/train_beam.pt \\
  bash scripts/train.sh.servesid sidmlp Beauty 0
# Defaults to seeds 42, 43, 44; one seed: SEEDS="43" bash ...

# 3. evaluate with constrained beam search
bash scripts/infer.sh
```

See `scripts/*.sh` and `configs/*.yaml` for dataset paths and hyperparameters
(candidate support `M`, training cutoff `K_tr`, pair range `B`, loss weights
`lambda_H`, `lambda_D`, `lambda_corr`, learning rate, and seeds).

## Acknowledgments

This code is built on top of the open-source
[SID-MLP](https://github.com/ztguo715/SID-MLP) repository and the
GenRec (TIGER) teacher pipeline.

## Paper-code alignment and reproducibility (October 2026)

The public paper-matching **MLP** objective is

```
L = L_digit_CE + lambda_H * L_HServe
               + lambda_D * lambda_corr * L_DA_weighted_Huber
lambda_H=4; lambda_D=1; lambda_corr=31.42
```

- `configs/servesid_final.yaml` is the WWW Full MLP example for **Beauty (Amazon2014)**. Set `DATASET`, `CATEGORY`, `DATASET_TAG`, and `VAL_DATASET_TAG` for the other domains; Video Games uses AmazonReviews2023.
- `loss: ce`, `d4_teacher: ce`, `pair_w: 0`, `da_margin: 0.05`, `da_w: 1`, `da_corr_w: 31.42`, and `serve_kl: 4` are intentional. `pair_gamma`/`da_gamma` are legacy temperature-mode controls; the new margin flags choose the paper's **additive** Softplus.
- Structural ablations are paper-aligned: DA-only / HServe / d4-only / HServe+Uniform / HServe+Plain Pair. The Plain Pair arm uses `pair_w: 1` and `pair_margin: 0.05`; all DA arms use `pair_w: 0`. `corr_mode` now reaches DA-CDRS (CF, uniform_valid, posres).
- All paper and diagnostic configs specify `seed: 42` for a single direct CLI run and `seeds: [42, 43, 44]` for the three intended repetitions; `seeds` is **metadata ignored by the individual CLI**. The launcher `scripts/run_paper_seeds.sh` **actually executes** 42, 43, 44 sequentially, with separate run tags.
- Every five epochs, training evaluates validation **NDCG@10** and writes `*_best_val_ndcg10.pth`; the original `*_final.pth` is still saved for diagnostics. Evaluate the **best validation checkpoint** on test exactly once. A missing validation split now raises an error instead of silently using test.
- Teacher beam candidates are sorted/truncated consistently with their per-depth `step_logp`, preventing candidate-to-depth-label mismatch when M=64.
- `python scripts/check_paper_consistency.py` statically verifies the source/config contract; the GitHub workflow also checks Python and shell syntax.

**Research-integrity limitation:** This change alters Full ServeSID's optimization compared with previous runs. Existing checkpoints and previously reported experiment tables are **not evidence of the corrected objective**. Re-run every affected ablation with actual training-only teacher caches, all seeds, and recorded validation selection. The repository does **not** currently contain the published GRU/CNN student architectures or their weights; it is therefore not sufficient by itself to reproduce the paper's MLP/GRU/CNN table. The teacher's T5 encoder is still used as a feature encoder at inference; 'no online teacher' means no online teacher **decoder/scorer**, not that the complete TIGER encoder has been removed. Teacher-cache row identity is not yet independently verified by a per-request hash.

To run the **MLP** paper domains with their own matching assets:

```bash
# 2014 Beauty (seeds 42/43/44)
GRAPH_DIR=... MODEL_DIR=... BEAM_DUMP=... bash scripts/run_paper_seeds.sh sidmlp Beauty 0
# 2014 Toys (provide Toys-specific graph/model/cache)
GRAPH_DIR=... MODEL_DIR=... BEAM_DUMP=... bash scripts/run_paper_seeds.sh sidmlp Toys 0
# 2023 Video Games (provide Video_Games-specific graph/model/cache)
GRAPH_DIR=... MODEL_DIR=... BEAM_DUMP=... bash scripts/run_paper_seeds.sh sidmlp Video_Games 0
```

The dataset preparation and offline beam dump must refer to exactly the same **training-request order**. Do not reuse the original checkpoints/metrics as if they were produced by these new loss settings.
