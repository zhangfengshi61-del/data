# SeRveSID — canonical repository

**The `main` branch is the single maintained source of truth for the WWW 2027
manuscript, student models, training objectives, baselines, and tests.**
Earlier review branches are historical snapshots, not supported implementation
variants. Open this repository on `main` for all current work.

## Repository map

| Location | Purpose |
| --- | --- |
| [`ServeSID/`](ServeSID/) | Reproducible training, inference, teacher/cache pipeline and tests |
| [`ServeSID/sid_mlp/`](ServeSID/sid_mlp/) | Canonical student MLP/GRU/CNN, HServe, DA-CDRS and adapted KD losses |
| [`ServeSID/configs/servesid_final.yaml`](ServeSID/configs/servesid_final.yaml) | **Full model** hyperparameters |
| [`ServeSID/configs/abl_*.yaml`](ServeSID/configs/) | Named ablations, not competing Full implementations |
| [`ServeSID/scripts/train_kd_baselines_all_arches.sh`](ServeSID/scripts/train_kd_baselines_all_arches.sh) | Adapted baseline launcher across backbones and seeds |
| [`ServeSID-www/submission.tex`](ServeSID-www/submission.tex) | **Only official submission PDF entry point** |
| [`ServeSID-www/main.tex`](ServeSID-www/main.tex) | Main manuscript source, included by `submission.tex` |
| [`ServeSID-www/supplement.tex`](ServeSID-www/supplement.tex) | Expanded supplementary source, not a second submission |
| [`.github/workflows/`](.github/workflows/) | On-push code checks and single-PDF ACM publication build |

Other top-level directories are upstream/reference assets; they are not
alternative ServeSID implementations.

## One objective, three backbones

The paper's Full training loss is

```text
L_full = L_digit_CE + 4 * L_HServe + 31.42 * L_DA_weighted_Huber
```

Full **does not** include a separate gold-prefix token KL or directly
optimized pairwise Softplus. The detached DA attribution *evaluates*
`softplus(0.05 - G)`. The `HServe + Plain Pair` ablation has its own
explicit pairwise loss and is not the Full objective. MLP/GRU/CNN share this
objective and the frozen T5 history encoder, but have different student
context modules. See [`ServeSID/README.md`](ServeSID/README.md) for
setup, input data, commands, and parameter counts.

## Building the manuscript

From the repository root, generate tables and the integrated appendix:

```bash
python3 ServeSID-www/scripts/gen_tables.py
python3 ServeSID-www/scripts/check_sensitivity_references.py
python3 ServeSID-www/scripts/check_rank_metric_invariants.py
python3 ServeSID-www/scripts/gen_submission_appendix.py
```

Then use the ACM LaTeX build on
[`ServeSID-www/submission.tex`](ServeSID-www/submission.tex).
The [**Build ServeSID paper** GitHub Actions workflow](.github/workflows/build-servesid.yml)
builds and page-audits `submission.pdf`, and makes it available under the
run's downloadable **ServeSID-submission-PDF** artifact. The PDF is a
generated release artifact, not a separately maintained copy of source data.
Standalone `main.tex` and `supplement.tex` remain available as
working/diagnostic inputs; submit only `submission.pdf`.

## Validation and experimental scope

The [code CI workflow](.github/workflows/verify-servesid-paper-loss.yml)
checks all three backbones, DA weighting, baseline gradient flow,
candidate-cache shapes, Best-Val checkpoint policy, and parameter counts.
The paper CI audits all 63 reported ablation rows for the necessary
single-positive Recall/NDCG inequalities and checks layout. These software
tests do not replace real GPU experiment logs. In particular, GKD and
DistiLLM-2 historical results should only be attributed to their corrected
implementations when the corresponding training records establish that
lineage.

**Maintenance rule:** make further manuscript and implementation changes on
`main`, keep named ablation configs under `ServeSID/configs/`, and do not
treat dated review branches or generated PDF files as alternative maintained
versions.
