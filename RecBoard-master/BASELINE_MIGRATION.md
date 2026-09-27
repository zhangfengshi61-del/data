# Baselines migrated from tiger协同语义训练

The following directories are complete source copies from the local
reproduction workspace:

| Baseline | Directory | Source |
|---|---|---|
| CoFiRec | `CoFiRec/` | `experiments/baselines/CoFiRec/` |
| UniGRec | `UniGRec/` | `experiments/baselines/UniGRec/` |
| Latte | `Latte/` | `experiments/baselines/Latte/` |
| Pctx | `Pctx/` | `experiments/baselines/Pctx/` |
All migrated runs must use the RecBoard comparison contract: Amazon Review
2014 5-core, chronological leave-two-out, max history 20, and full-ranking
evaluation with HitRate@1/5/10/20 and NDCG@5/10/20. The canonical `data/` and
`baseline_inputs/` directories are shared with the audited tiger workspace,
so migration does not create a second split or item mapping.
