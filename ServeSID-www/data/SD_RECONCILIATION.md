# Main–Ablation Standard-Deviation Reconciliation

The author confirmed both main-result and ablation aggregate tables as real experimental results on 2026-10-11. Their means coincide on all 72 duplicated **Base KD** and **Full ServeSID** measurements, but their standard deviations do not coincide on **67/72** entries.

Two distinct tables can legitimately reflect different repetitions, checkpoint choices, or reporting protocols. However, if they claim the same three seeds (42, 43, 44) under the same method, dataset, architecture, and checkpoint-selection rule, their SDs should be identical. Neither source is silently overwritten here.

**Action before publication:** compare the seed-level metric files and checkpoint IDs; determine whether the main- and ablation-table arms represent precisely the same runs. If yes, replace inconsistent SDs with values recomputed from those runs and regenerate both LaTeX tables. If not, clearly disclose differing protocols.

| Dataset | Student | Shared method | Metric | Mean | Main SD | Ablation SD |
|---|---|---|---|---:|---:|---:|
| Beauty | MLP | Base KD | R@5 | 0.0429 | 0.00020 | 0.00017 |
| Beauty | MLP | Base KD | NDCG@5 | 0.0282 | 0.00018 | 0.00006 |
| Beauty | MLP | Base KD | R@10 | 0.0660 | 0.00023 | 0.00022 |
| Beauty | MLP | Base KD | NDCG@10 | 0.0356 | 0.00024 | 0.00020 |
| Beauty | MLP | Full ServeSID | R@5 | 0.0442 | 0.00023 | 0.00011 |
| Beauty | MLP | Full ServeSID | NDCG@5 | 0.0292 | 0.00014 | 0.00011 |
| Beauty | MLP | Full ServeSID | R@10 | 0.0679 | 0.00029 | 0.00025 |
| Beauty | MLP | Full ServeSID | NDCG@10 | 0.0369 | 0.00027 | 0.00022 |
| Beauty | GRU | Base KD | R@5 | 0.0416 | 0.00029 | 0.00028 |
| Beauty | GRU | Base KD | NDCG@5 | 0.0281 | 0.00008 | 0.00014 |
| Beauty | GRU | Base KD | R@10 | 0.0656 | 0.00025 | 0.00039 |
| Beauty | GRU | Base KD | NDCG@10 | 0.0358 | 0.00025 | 0.00020 |
| Beauty | GRU | Full ServeSID | R@5 | 0.0441 | 0.00019 | 0.00017 |
| Beauty | GRU | Full ServeSID | NDCG@5 | 0.0291 | 0.00020 | 0.00011 |
| Beauty | GRU | Full ServeSID | R@10 | 0.0682 | 0.00044 | 0.00045 |
| Beauty | GRU | Full ServeSID | NDCG@10 | 0.0368 | 0.00016 | 0.00017 |
| Beauty | CNN | Base KD | R@5 | 0.0380 | 0.00028 | 0.00011 |
| Beauty | CNN | Base KD | NDCG@5 | 0.0254 | 0.00015 | 0.00017 |
| Beauty | CNN | Base KD | R@10 | 0.0580 | 0.00033 | 0.00036 |
| Beauty | CNN | Base KD | NDCG@10 | 0.0319 | 0.00012 | 0.00020 |
| Beauty | CNN | Full ServeSID | R@5 | 0.0437 | 0.00017 | 0.00020 |
| Beauty | CNN | Full ServeSID | NDCG@5 | 0.0288 | 0.00008 | 0.00011 |
| Beauty | CNN | Full ServeSID | R@10 | 0.0686 | 0.00024 | 0.00034 |
| Beauty | CNN | Full ServeSID | NDCG@10 | 0.0368 | 0.00021 | 0.00011 |
| Toys | MLP | Base KD | R@5 | 0.0374 | 0.00015 | 0.00017 |
| Toys | MLP | Base KD | NDCG@5 | 0.0235 | 0.00011 | 0.00006 |
| Toys | MLP | Base KD | R@10 | 0.0626 | 0.00045 | 0.00031 |
| Toys | MLP | Base KD | NDCG@10 | 0.0315 | 0.00012 | 0.00017 |
| Toys | MLP | Full ServeSID | R@5 | 0.0408 | 0.00023 | 0.00028 |
| Toys | MLP | Full ServeSID | NDCG@10 | 0.0330 | 0.00022 | 0.00008 |
| Toys | GRU | Base KD | NDCG@5 | 0.0233 | 0.00015 | 0.00014 |
| Toys | GRU | Base KD | R@10 | 0.0565 | 0.00030 | 0.00036 |
| Toys | GRU | Base KD | NDCG@10 | 0.0296 | 0.00012 | 0.00014 |
| Toys | GRU | Full ServeSID | R@5 | 0.0388 | 0.00017 | 0.00011 |
| Toys | GRU | Full ServeSID | NDCG@5 | 0.0250 | 0.00007 | 0.00008 |
| Toys | GRU | Full ServeSID | NDCG@10 | 0.0334 | 0.00020 | 0.00017 |
| Toys | CNN | Base KD | R@5 | 0.0328 | 0.00024 | 0.00008 |
| Toys | CNN | Base KD | NDCG@5 | 0.0210 | 0.00010 | 0.00008 |
| Toys | CNN | Base KD | R@10 | 0.0533 | 0.00037 | 0.00017 |
| Toys | CNN | Base KD | NDCG@10 | 0.0276 | 0.00018 | 0.00008 |
| Toys | CNN | Full ServeSID | NDCG@5 | 0.0250 | 0.00016 | 0.00008 |
| Toys | CNN | Full ServeSID | R@10 | 0.0639 | 0.00042 | 0.00022 |
| Toys | CNN | Full ServeSID | NDCG@10 | 0.0328 | 0.00012 | 0.00014 |
| VG | MLP | Base KD | R@5 | 0.0588 | 0.00030 | 0.00034 |
| VG | MLP | Base KD | NDCG@5 | 0.0383 | 0.00017 | 0.00014 |
| VG | MLP | Base KD | R@10 | 0.0922 | 0.00049 | 0.00022 |
| VG | MLP | Base KD | NDCG@10 | 0.0488 | 0.00027 | 0.00011 |
| VG | MLP | Full ServeSID | R@5 | 0.0604 | 0.00028 | 0.00011 |
| VG | MLP | Full ServeSID | NDCG@5 | 0.0396 | 0.00021 | 0.00014 |
| VG | MLP | Full ServeSID | R@10 | 0.0944 | 0.00067 | 0.00034 |
| VG | MLP | Full ServeSID | NDCG@10 | 0.0505 | 0.00031 | 0.00011 |
| VG | GRU | Base KD | R@5 | 0.0591 | 0.00030 | 0.00022 |
| VG | GRU | Base KD | NDCG@5 | 0.0389 | 0.00013 | 0.00017 |
| VG | GRU | Base KD | R@10 | 0.0935 | 0.00068 | 0.00067 |
| VG | GRU | Base KD | NDCG@10 | 0.0499 | 0.00025 | 0.00028 |
| VG | GRU | Full ServeSID | R@5 | 0.0607 | 0.00036 | 0.00034 |
| VG | GRU | Full ServeSID | NDCG@5 | 0.0400 | 0.00019 | 0.00017 |
| VG | GRU | Full ServeSID | R@10 | 0.0954 | 0.00029 | 0.00064 |
| VG | GRU | Full ServeSID | NDCG@10 | 0.0512 | 0.00037 | 0.00025 |
| VG | CNN | Base KD | R@5 | 0.0553 | 0.00028 | 0.00031 |
| VG | CNN | Base KD | NDCG@5 | 0.0358 | 0.00021 | 0.00017 |
| VG | CNN | Base KD | R@10 | 0.0870 | 0.00046 | 0.00034 |
| VG | CNN | Base KD | NDCG@10 | 0.0460 | 0.00015 | 0.00020 |
| VG | CNN | Full ServeSID | R@5 | 0.0605 | 0.00022 | 0.00031 |
| VG | CNN | Full ServeSID | NDCG@5 | 0.0399 | 0.00021 | 0.00025 |
| VG | CNN | Full ServeSID | R@10 | 0.0944 | 0.00059 | 0.00045 |
| VG | CNN | Full ServeSID | NDCG@10 | 0.0508 | 0.00016 | 0.00031 |
