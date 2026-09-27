# RecBoard-master baseline results

Recorded: 2026-09-17 (Asia/Shanghai)
Workspace: `/data/fszhang/RecBoard-master`
Python env: `/data/fszhang/anaconda/envs/myenv_t5/bin/python`

## Comparison contract

From `BASELINE_MIGRATION.md` and `baseline_inputs/PROTOCOL_MANIFEST.json`:

- Raw data: Amazon Review 2014 5-core
- Split: per-user chronological leave-two-out
- History: right-aligned, max length 20
- Evaluation: full-ranking, seen items masked
- Metrics: HitRate/Recall@5/10/20, NDCG@5/10/20, MRR (where available)
- Beam: 30 (generative methods)
- Canonical data: `data/processed/{Beauty,Sports,Toys}`
  (Beauty 22363 users / 12101 items / 198502 interactions, etc.)

All results below are **test-set** numbers under this contract, unless noted.

## Status

| Method | Status | Test results |
|---|---|---|
| CoFiRec | done | Beauty / Sports / Toys |
| Latte | done | Beauty / Sports / Toys |
| UniGRec | done | Beauty / Sports / Toys |
| Pctx | done | Beauty / Sports / Toys |
| SDQ-403 (SDQ-VAE, T5) | done | Beauty / Sports / Toys |

## CoFiRec

Source: `baseline_runs/CoFiRec/logs/{Beauty,Sports,Toys}_full_test.json`
(single prompt, `test_prompt_ids="0"`).

| Dataset | Hit@5 | Hit@10 | Hit@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---|---|---|---|---|---|---|
| Beauty | 0.03796 | 0.06265 | 0.09189 | 0.02416 | 0.03206 | 0.03942 | 0.02489 |
| Sports | 0.10484 | 0.20254 | 0.35786 | 0.06266 | 0.09406 | 0.13345 | 0.07265 |
| Toys | 0.02107 | 0.03261 | 0.05115 | 0.01306 | 0.01679 | 0.02147 | 0.01327 |

## Latte

Source: `baseline_runs/Latte/orchestrator_{Beauty,Sports,Toys}.out` (`RESULT` line,
best checkpoint selected on validation NDCG@10).

| Dataset | Best epoch | Best val NDCG@10 | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| Beauty | 81 | 0.04906 | 0.04387 | 0.05397 | 0.05464 | 0.02952 | 0.03288 | 0.03306 | 0.02626 |
| Sports | 113 | 0.02850 | 0.02854 | 0.03514 | 0.03520 | 0.01890 | 0.02113 | 0.02115 | 0.01672 |
| Toys | 102 | 0.04450 | 0.04544 | 0.05610 | 0.05625 | 0.02930 | 0.03293 | 0.03297 | 0.02563 |

## UniGRec

Source: `baseline_runs/UniGRec/logs/joint_{Beauty,Sports,Toys}.log`
(joint training + finetune, best finetune model on test set).
UniGRec does not report HitRate@1 or MRR in its finetune summary.

| Dataset | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Beauty | 0.0510 | 0.0747 | 0.1081 | 0.0346 | 0.0423 | 0.0506 |
| Sports | 0.0302 | 0.0455 | 0.0681 | 0.0193 | 0.0242 | 0.0299 |
| Toys | 0.0461 | 0.0751 | 0.1097 | 0.0297 | 0.0390 | 0.0477 |

## Pctx

Source: `baseline_runs/Pctx/logs/{duorec,upstream,gr}_{Beauty,Sports,Toys}.log`

Three stages: DuoRec auxiliary model -> upstream personalized clustering -> GR
training/test. Pctx only reports NDCG/Recall@5/10 (no @20, no MRR).

| Dataset | Best epoch | Best val NDCG@10 | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| Beauty | 121 | 0.05653 | 0.04843 | 0.07369 | 0.03244 | 0.04053 |
| Sports | 57 | 0.02401 | 0.02073 | 0.03489 | 0.01303 | 0.01758 |
| Toys | 111 | 0.04840 | 0.04188 | 0.06908 | 0.02669 | 0.03547 |

## Cross-method summary (test NDCG@10)

| Dataset | CoFiRec | Latte | UniGRec | Pctx | SDQ-VAE |
|---|---|---|---|---|---|
| Beauty | 0.0321 | 0.0329 | **0.0423** | 0.0405 | 0.0411 |
| Sports | 0.0941* | 0.0211 | 0.0242 | 0.0176 | **0.0267** |
| Toys | 0.0168 | 0.0329 | **0.0390** | 0.0355 | 0.0389 |

\* CoFiRec Sports is anomalously high and unverified (see notes).

## SDQ-403 (SDQ-VAE + T5)

Source: `baseline_runs/SDQ-403/logs/t5_vae_{Beauty,Sports,Toys}.log`
(best checkpoint by validation NDCG@10, then evaluated on the test set).

| Dataset | Best epoch | Best val NDCG@10 | Hit@5 | Hit@10 | Hit@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---|---|---|---|---|---|---|---|---|
| Beauty | 105 | 0.0584 | 0.0504 | 0.0751 | 0.1105 | 0.0331 | 0.0411 | 0.0500 | 0.0307 |
| Sports | 80 | 0.0352 | 0.0319 | 0.0494 | 0.0719 | 0.0211 | 0.0267 | 0.0323 | 0.0198 |
| Toys | 90 | 0.0537 | 0.0460 | 0.0701 | 0.1026 | 0.0311 | 0.0389 | 0.0471 | 0.0294 |

Runner: `SDQ-403/run_local.sh <Dataset> <gpu> [vae|kmeans|both] [sdq_epochs] [t5_epochs]`
Queue watcher: `run_sdq_after.sh` (started after `run_all_serial.sh` finished,
3 datasets in parallel on GPUs 1/2/3).

Pipeline: canonical parquet -> freerec format (`SDQ-403/prepare_local_data.py`)
-> sentence-t5-xl item features -> SDQ-VAE quantization -> `sid_vocab.json`
-> T5 next-item training/test (`num_beams=30`, `seed=2025`).

Only the main **SDQ-VAE** variant is reproduced. The repository's
non-trainable `SDQ-KMeans` ablation was dropped by request to save compute.

## SID-MLP-RecBoard (2026-09-24)

Our own SID-MLP on the protocol: frozen SDQ-T5 teacher encoder + conditional
MLP heads (head_hidden=512), full-catalog scoring, 200 epochs + early stop
(patience 15 evals, eval every 5 epochs), best by val NDCG@10, seed 2025.

> **SID tokenizer note: SID-MLP-RecBoard uses the same SDQ-VAE SID as the
> TIGER-style T5 baseline above** (identical `sid_vocab.json` from
> `SDQ-403/logs/SDQ/Amazon2014{DS}_550_LOU/vae/`). No method in this
> comparison uses the original TIGER paper's RQ-VAE codes. The quality gap
> between the beam-search T5 baseline and the parallel scorers is therefore
> attributable to the decoding/scoring method, not to the tokenizer.

Source: `HiFlow-SID/logs/sidmlp_{Beauty,Sports,Toys}_v2.log`
(`TEST @Epoch` line at best epoch).
Code: `HiFlow-SID/train_sidmlp.py`, `HiFlow-SID/hiflow_lib/model_sidmlp.py`.

| Dataset | Early stop | Best epoch | Hit@5 | Hit@10 | Hit@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|---|---|
| Beauty | 90 | 10 | 0.0282 | 0.0457 | 0.0723 | 0.0191 | 0.0246 | 0.0315 |
| Sports | 90 | 10 | 0.0147 | 0.0250 | 0.0396 | 0.0095 | 0.0131 | 0.0169 |
| Toys | 95 | 15 | 0.0269 | 0.0432 | 0.0675 | 0.0181 | 0.0241 | 0.0305 |

(Recall == HitRate per cut in the full-ranking setting; no MRR in this run's
monitor list.)

Speed (SID-MLP rules: batch32 / beam50 / bf16 / TF32 off / test set):
TIGER 69.69/113.14/61.96 s vs SID-MLP-RecBoard 20.13/33.04/16.85 s
→ **3.46× / 3.42× / 3.68×** (see `tiger并行加速优化/速度测试结果.md` §8).

Notes:

- Best val arrives very early (epoch 10/10/15); the rest of the run is
  plateau/overfit, so early-stop patience in *evals* (15 × eval_freq 5 = 75
  epochs after best) is what makes the runs stop at 90/90/95.
- Quality is below the TIGER teacher (test NDCG@10 0.0411/0.0267/0.0389) and
  below MTP on Beauty (0.0251). Treat as a strong parallel-scoring baseline,
  not a quality upper bound. (Full metric rows: Hit@5/10/20 and NDCG@5/10/20
  above; Recall == HitRate per cut under full ranking.)

## RQ-VAE SID re-run (2026-09-24)

Same protocol, splits and seed 2025 — only the SID tokenizer changes from
SDQ-VAE to a freshly trained **RQ-VAE** (original TIGER's residual
quantization; 3 x 256 codebooks; collision 6.6%/6.8%).

RQ-VAE trained with `TIGER/train_rqvae.py` on protocol data; SIDs at
`HiFlow-SID/logs/sid_vocab_rqvae_{Beauty,Sports,Toys}.json`. Downstream runs
(all on RQ SIDs): T5 teacher (`train_t5_rq.py`, 200 epochs, fast candidate
eval, checkpoints at `SDQ-403/logs/TIGER-T5/Amazon2014{DS}_550_LOU/t5-rq-fast/`),
MTP (`--id mtp-rq`), SID-MLP (`--id sidmlp-rq-{DS}`), ExpertScore v3
(`--id pisid-expertscore-rq-{DS}`). Logs: `HiFlow-SID/logs/stage{2,3,4,5}_*_rq*.log`.

Test NDCG@10 / Hit@10 (best by val NDCG@10):

| Method | Beauty | Sports | Toys |
|---|---:|---:|---:|
| TIGER-T5 (trie protocol eval) | pending | pending | pending |
| TIGER-T5 (unconstrained ref) | 0.0367 / 0.0669 | 0.0223 / 0.0421 | 0.0353 / 0.0624 |
| MTP | 0.0255 / 0.0455 | 0.0099 / 0.0192 | 0.0276 / 0.0487 |
| SID-MLP-RecBoard | 0.0300 / 0.0568 | 0.0133 / 0.0246 | 0.0282 / 0.0509 |
| ExpertScore v3 | 0.0275 / 0.0511 | 0.0130 / 0.0254 | 0.0273 / 0.0480 |

Beauty cross-tokenizer (test NDCG@10, SDQ-VAE SID -> RQ-VAE SID):
MTP 0.0251 -> 0.0255; SID-MLP 0.0246 -> **0.0300 (+22%)**; ExpertScore
0.0260 -> 0.0275. RQ-VAE SIDs are friendlier to parallel scorers.
Protocol (trie-constrained) TIGER baseline evaluation in progress:
`HiFlow-SID/scripts/eval_t5_protocol.py`.

## Notes and caveats

- **SDQ-VAE is competitive with UniGRec**: best on Sports (0.0267 vs 0.0242),
  essentially tied on Toys (0.0389 vs 0.0390), and a close second on Beauty
  (0.0411 vs 0.0423).
- **UniGRec** is the strongest on Beauty (0.0423) and Toys (0.0390).
- **Pctx** is second best on Beauty (NDCG@10 0.0405) and Toys (0.0355),
  behind UniGRec, and lowest on Sports (0.0176). It only reports @5/@10.
- **CoFiRec Sports is anomalously high** (NDCG@10 0.0941, ~3x Beauty and ~6x
  Toys). This is not comparable to the other datasets without checking the
  index/candidate set used for Sports (local tokenizer vs official assets).
  Treat it as suspicious until verified.
- **Latte @20 == @10** exactly (Recall and NDCG) for all three datasets, i.e.
  its candidate list is effectively capped at ~10 (`num_return_sequences`).
  Its @20 numbers carry no extra information.
- Generative SID methods (Latte, UniGRec, SDQ) use constrained beam search and
  are **not** directly comparable to full-catalog discriminative scores; per
  `REPRODUCTION_MATRIX.md` fairness notes, the two families must not be mixed
  in one claim.
- CoFiRec and UniGRec were not run with an explicit `seed=2025`; Latte/UniGRec
  use their own default seeds. SDQ uses `seed=2025` per the protocol manifest.

## How to refresh

```bash
# Completed methods: read the sources listed above.
# SDQ-VAE (final test metrics + best checkpoint):
tr '\r' '\n' < baseline_runs/SDQ-403/logs/t5_vae_Beauty.log | grep "TEST @Epoch"
```
