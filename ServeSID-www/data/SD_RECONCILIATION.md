# Main–Ablation SD Source Reconciliation

The author specified on 2026-10-11 that the **second (ablation) summary** is the preferred reporting source for shared Base KD and Full ServeSID statistics. This is source precedence, **not** a rule to select the smaller SD and not independent validation of seed-level measurements.

| Check | Result |
|---|---:|
| Shared metric cells | 72 |
| Shared means preserved | 72 / 72 |
| Previously inconsistent SDs replaced | 67 |
| Already matching SDs retained | 5 |
| New main–ablation SD discrepancies | 0 |
| Selected SD higher than previous main SD | 26 |
| Selected SD lower than previous main SD | 41 |

The 288 main metrics, 252 ablation metrics and all means are otherwise unchanged. Data source: `ablation_user_supplied_mean_sd.csv` for shared values; the other main-table methods continue to use `main_user_supplied_mean_sd.csv`. A fail-fast consistency check is included in `scripts/gen_tables.py`.

**Provenance limitation:** aggregate SDs have not been independently recalculated from per-seed logs/checkpoints. Per-seed values need not appear in the paper, but statistical significance and full run-level reproducibility cannot be inferred from these summaries alone.

## Historical differences (before reconciliation)

| Dataset / Student / Method / Metric | Former Main SD | Selected Ablation SD |
|---|---:|---:|
| Beauty / MLP / SID-MLP (base KD) / R@5 | 0.00020 | 0.00017 |
| Beauty / MLP / SID-MLP (base KD) / NDCG@5 | 0.00018 | 0.00006 |
| Beauty / MLP / SID-MLP (base KD) / R@10 | 0.00023 | 0.00022 |
| Beauty / MLP / SID-MLP (base KD) / NDCG@10 | 0.00024 | 0.00020 |
| Beauty / MLP / ServeSID (ours) / R@5 | 0.00023 | 0.00011 |
| Beauty / MLP / ServeSID (ours) / NDCG@5 | 0.00014 | 0.00011 |
| Beauty / MLP / ServeSID (ours) / R@10 | 0.00029 | 0.00025 |
| Beauty / MLP / ServeSID (ours) / NDCG@10 | 0.00027 | 0.00022 |
| Beauty / GRU / SID-MLP (base KD) / R@5 | 0.00029 | 0.00028 |
| Beauty / GRU / SID-MLP (base KD) / NDCG@5 | 0.00008 | 0.00014 |
| Beauty / GRU / SID-MLP (base KD) / R@10 | 0.00025 | 0.00039 |
| Beauty / GRU / SID-MLP (base KD) / NDCG@10 | 0.00025 | 0.00020 |
| Beauty / GRU / ServeSID (ours) / R@5 | 0.00019 | 0.00017 |
| Beauty / GRU / ServeSID (ours) / NDCG@5 | 0.00020 | 0.00011 |
| Beauty / GRU / ServeSID (ours) / R@10 | 0.00044 | 0.00045 |
| Beauty / GRU / ServeSID (ours) / NDCG@10 | 0.00016 | 0.00017 |
| Beauty / CNN / SID-MLP (base KD) / R@5 | 0.00028 | 0.00011 |
| Beauty / CNN / SID-MLP (base KD) / NDCG@5 | 0.00015 | 0.00017 |
| Beauty / CNN / SID-MLP (base KD) / R@10 | 0.00033 | 0.00036 |
| Beauty / CNN / SID-MLP (base KD) / NDCG@10 | 0.00012 | 0.00020 |
| Beauty / CNN / ServeSID (ours) / R@5 | 0.00017 | 0.00020 |
| Beauty / CNN / ServeSID (ours) / NDCG@5 | 0.00008 | 0.00011 |
| Beauty / CNN / ServeSID (ours) / R@10 | 0.00024 | 0.00034 |
| Beauty / CNN / ServeSID (ours) / NDCG@10 | 0.00021 | 0.00011 |
| Toys / MLP / SID-MLP (base KD) / R@5 | 0.00015 | 0.00017 |
| Toys / MLP / SID-MLP (base KD) / NDCG@5 | 0.00011 | 0.00006 |
| Toys / MLP / SID-MLP (base KD) / R@10 | 0.00045 | 0.00031 |
| Toys / MLP / SID-MLP (base KD) / NDCG@10 | 0.00012 | 0.00017 |
| Toys / MLP / ServeSID (ours) / R@5 | 0.00023 | 0.00028 |
| Toys / MLP / ServeSID (ours) / NDCG@10 | 0.00022 | 0.00008 |
| Toys / GRU / SID-MLP (base KD) / NDCG@5 | 0.00015 | 0.00014 |
| Toys / GRU / SID-MLP (base KD) / R@10 | 0.00030 | 0.00036 |
| Toys / GRU / SID-MLP (base KD) / NDCG@10 | 0.00012 | 0.00014 |
| Toys / GRU / ServeSID (ours) / R@5 | 0.00017 | 0.00011 |
| Toys / GRU / ServeSID (ours) / NDCG@5 | 0.00007 | 0.00008 |
| Toys / GRU / ServeSID (ours) / NDCG@10 | 0.00020 | 0.00017 |
| Toys / CNN / SID-MLP (base KD) / R@5 | 0.00024 | 0.00008 |
| Toys / CNN / SID-MLP (base KD) / NDCG@5 | 0.00010 | 0.00008 |
| Toys / CNN / SID-MLP (base KD) / R@10 | 0.00037 | 0.00017 |
| Toys / CNN / SID-MLP (base KD) / NDCG@10 | 0.00018 | 0.00008 |
| Toys / CNN / ServeSID (ours) / NDCG@5 | 0.00016 | 0.00008 |
| Toys / CNN / ServeSID (ours) / R@10 | 0.00042 | 0.00022 |
| Toys / CNN / ServeSID (ours) / NDCG@10 | 0.00012 | 0.00014 |
| VG / MLP / SID-MLP (base KD) / R@5 | 0.00030 | 0.00034 |
| VG / MLP / SID-MLP (base KD) / NDCG@5 | 0.00017 | 0.00014 |
| VG / MLP / SID-MLP (base KD) / R@10 | 0.00049 | 0.00022 |
| VG / MLP / SID-MLP (base KD) / NDCG@10 | 0.00027 | 0.00011 |
| VG / MLP / ServeSID (ours) / R@5 | 0.00028 | 0.00011 |
| VG / MLP / ServeSID (ours) / NDCG@5 | 0.00021 | 0.00014 |
| VG / MLP / ServeSID (ours) / R@10 | 0.00067 | 0.00034 |
| VG / MLP / ServeSID (ours) / NDCG@10 | 0.00031 | 0.00011 |
| VG / GRU / SID-MLP (base KD) / R@5 | 0.00030 | 0.00022 |
| VG / GRU / SID-MLP (base KD) / NDCG@5 | 0.00013 | 0.00017 |
| VG / GRU / SID-MLP (base KD) / R@10 | 0.00068 | 0.00067 |
| VG / GRU / SID-MLP (base KD) / NDCG@10 | 0.00025 | 0.00028 |
| VG / GRU / ServeSID (ours) / R@5 | 0.00036 | 0.00034 |
| VG / GRU / ServeSID (ours) / NDCG@5 | 0.00019 | 0.00017 |
| VG / GRU / ServeSID (ours) / R@10 | 0.00029 | 0.00064 |
| VG / GRU / ServeSID (ours) / NDCG@10 | 0.00037 | 0.00025 |
| VG / CNN / SID-MLP (base KD) / R@5 | 0.00028 | 0.00031 |
| VG / CNN / SID-MLP (base KD) / NDCG@5 | 0.00021 | 0.00017 |
| VG / CNN / SID-MLP (base KD) / R@10 | 0.00046 | 0.00034 |
| VG / CNN / SID-MLP (base KD) / NDCG@10 | 0.00015 | 0.00020 |
| VG / CNN / ServeSID (ours) / R@5 | 0.00022 | 0.00031 |
| VG / CNN / ServeSID (ours) / NDCG@5 | 0.00021 | 0.00025 |
| VG / CNN / ServeSID (ours) / R@10 | 0.00059 | 0.00045 |
| VG / CNN / ServeSID (ours) / NDCG@10 | 0.00016 | 0.00031 |
