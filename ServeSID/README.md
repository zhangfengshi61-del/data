# ServeSID: Hierarchy-Aligned and Direction-Aware Distillation for Semantic-ID Generative Recommendation

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
- **DA-CDRS (Direction-Aware Cross-Depth Relative-Score Distillation)**:
  distills teacher-ordered cross-cutoff candidate pairs by decomposing their
  score gaps into depth-wise relative-score increments and weighting
  scale-normalized Huber correction by the direction-aware (positive-residual)
  attribution of each depth. No pairwise ranking loss is added to the training
  objective.

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

## Acknowledgments

This code is built on top of the open-source
[SID-MLP](https://github.com/ztguo715/SID-MLP) repository and the
GenRec (TIGER) teacher pipeline.
