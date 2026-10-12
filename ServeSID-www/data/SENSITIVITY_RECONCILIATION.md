# Sensitivity-reference reconciliation (2026-10-12)

**Decision:** For a sensitivity row corresponding to the same named
arm and default parameter setting as Table 2, the author instructed
that the main ablation mean is authoritative. The 15 aligned rows
are explicitly marked with a dagger in `supplement.tex`. They are
**reference anchors copied from Table 2**, not additional measurements
at those sweep settings. The original values are recorded in the
historical discrepancy section below; no prior reported measurements
are silently deleted.

| Table | Anchored setting | Source (MLP main ablation) |
|---|---|---|
| M sensitivity | M=64 | HServe only |
| B sensitivity | B=40 | HServe + Plain Pair |
| beta sensitivity | beta=0 | HServe only, because no direct Pair Loss |
| beta sensitivity | beta=31.42 | Full ServeSID |
| tau sensitivity | tau=1 | HServe only |

The M, B, beta and tau results for all other settings retain the
independently supplied values. This means these comparisons are
**reference-anchored summaries**, not matched-seed controlled
hyperparameter scans. Original training/evaluation logs are needed
before presenting differences as rigorous sensitivity effects.

## Original provided reference-row values (before paper precedence)

All vectors below are in dataset order Beauty / Toys / Video Games.
M and B vectors have metric order R@5, NDCG@5, R@10, NDCG@10;
beta and tau have metric order R@10, NDCG@10.

| Parameter | Original separately supplied row | Authoritative paper row |
|---|---|---|
| M=64 Beauty | .0437 .0288 .0670 .0364 | .0436 .0287 .0670 .0363 |
| M=64 Toys | .0393 .0246 .0637 .0324 | .0390 .0245 .0633 .0323 |
| M=64 Video Games | .0598 .0391 .0932 .0498 | .0597 .0390 .0935 .0499 |
| B=40 Beauty | .0440 .0290 .0675 .0366 | .0435 .0288 .0668 .0363 |
| B=40 Toys | .0400 .0250 .0640 .0328 | .0388 .0246 .0631 .0322 |
| B=40 Video Games | .0601 .0393 .0940 .0501 | .0595 .0389 .0936 .0497 |
| beta=0 Beauty | .0675 .0366 | .0670 .0363 |
| beta=0 Toys | .0640 .0328 | .0633 .0323 |
| beta=0 Video Games | .0940 .0501 | .0935 .0499 |
| tau=1 Beauty | .0670 .0364 | .0670 .0363 |
| tau=1 Toys | .0637 .0324 | .0633 .0323 |
| tau=1 Video Games | .0932 .0498 | .0935 .0499 |

The beta=31.42 row matched Full before reconciliation (no numerical
change). The cross-table trend narratives were revised to avoid
unsupported monotonicity or 'optimal B' claims.

---

# Sensitivity vs. Main Ablation: Protocol Reconciliation

This is an audit note for the author's 2026-10-12 four sensitivity tables.
The supplied values were copied unchanged into `supplement.tex`.
Their sensitivity-specific per-seed checkpoints/configuration IDs are not
in the provided record; **do not silently replace the main ablation results**.
The comparison below is against the currently reported MLP means for
the same named arm and dataset.

| Sensitivity condition | Main ablation counterpart | Different point-estimate cells |
|---|---|---:|
| M=64, HServe-only | HServe only | **11/12** (R@5, N@5, R@10, N@10 x 3) |
| B=40, HServe + Plain Pair | HServe + Plain Pair | **12/12** |
| beta=0 of no-direct-pair Full | HServe only | **6/6** (R@10, N@10 x 3) |
| beta=31.42 | Full ServeSID | **0/6** |
| tau=1 | M=64 sensitivity within the same supplied tables | **0/6** |

## Critical equivalence check: beta=0

The paper's corrected Full objective is
`Digit CE + 4*HServe + beta*DA-weighted-Huber`, **without a separately
optimized pairwise Softplus anchor**. Consequently beta=0 switches off
DA-CDRS correction and should be identical to HServe-only *under exactly
matched initial seeds, optimizer, data splits, cache, epochs, and checkpoint
selection*. Different runs may produce different statistics, but that
requires explicit protocol identification.

The author's beta=0 sweep reports:
- Beauty: R@10 **0.0675**, N@10 **0.0366**
- Toys: R@10 **0.0640**, N@10 **0.0328**
- Video Games: R@10 **0.0940**, N@10 **0.0501**

The current HServe-only ablation mean reports respectively
**0.0670 / 0.0363**, **0.0633 / 0.0323**, and
**0.0935 / 0.0499**. All six entries differ.

In addition, the beta=0 entries match the **separate supplied B=40
Plain Pair sensitivity row** (R@10 and N@10) exactly for all three
datasets. This is not proof that a pairwise anchor was optimized, but
it is a **specific source/configuration identity question**. Do not
claim beta=0 is a matched loss-ablation control until logs confirm it.

## M=64 HServe-only vs main table (illustrative mismatches)

| Metric | Sensitivity M=64 | Main HServe only |
|---|---:|---:|
| Beauty R@5 | 0.0437 | 0.0436 |
| Beauty NDCG@10 | 0.0364 | 0.0363 |
| Toys R@10 | 0.0637 | 0.0633 |
| Video Games R@10 | 0.0932 | 0.0935 |

The single matching cell is Beauty R@10 (both 0.0670).

## B=40 Plain Pair vs main table (all twelve differ)

| Dataset | Sweep R@5 / N@5 / R@10 / N@10 | Main ablation same order |
|---|---|---|
| Beauty | .0440 / .0290 / .0675 / .0366 | .0435 / .0288 / .0668 / .0363 |
| Toys | .0400 / .0250 / .0640 / .0328 | .0388 / .0246 / .0631 / .0322 |
| Video Games | .0601 / .0393 / .0940 / .0501 | .0595 / .0389 / .0936 / .0497 |

## Before final publication

1. Identify the exact code commit, loss variant (in particular no-direct
   pair anchor), seed, dataset split, teacher dump width, validation
   checkpoint selection and metric evaluator for beta=0 and the other sweeps.
2. If these are independent sweep runs, label the appendix explicitly as
   independent runs and avoid the impression of cell identity with Table 2.
3. If these are intended to be *identical* runs, correct the underlying
   source tables based on original evaluation records; **do not select
   whichever estimate is larger**.
4. The 42 sensitivity rows have point estimates only, not reported
   three-seed SDs; no significance conclusion can be drawn from small
   (0.0001--0.0004) changes.

No values were invented or modified by this reconciliation.
