# SID-RAD paper ↔ ServeSID code alignment (2026-10-21)

This report records **implementation changes, not new experiment results**.
Names in the manuscript: SID-RAD / HCDA / RRDA. Maintained code equivalents:
ServeSID / HServe / DA-CDRS. The existing Full mode remains `loss=ce`,
`serve_kl=4`, `da_w=1`, `da_corr_w=31.42`, `corr_mode=cf`,
`pair_w=0`, `M=64`, `K=10`, `B=40`, `W=50`.

## Four extra variants

The distinct recipes are in `configs/abl_hcda_{absres,posuni,shuffled,collapsed}.yaml`.
They use the same real-interaction CE, HCDA, architectures, validation
selection and RRDA loss coefficient as Full, varying only the named control:

| Paper label | Code route | Exact semantics |
|---|---|---|
| HCDA+AbsRes | `corr_mode=absres` | Pair gated by at least one positive correction, then normalized **absolute** residual magnitude across all eligible depths, including negative corrections |
| HCDA+PosUni | `corr_mode=posuni` | Equal weights on eligible depths with strictly positive teacher–student corrections, zero otherwise |
| HCDA+Shuffled | `hcda_shuffle=true` | Same per-request random permutation of teacher/student **candidate score slots** for HCDA only; SID leaf positions/prefix grouping fixed; RRDA remains on unshuffled teacher/student increments |
| HCDA+Collapsed | `corr_mode=collapsed` | Sum eligible raw depth-gap residuals, then apply one unit-threshold Huber penalty after normalizing by detached current-minibatch std (floor 1e-4); changes granularity **and** normalization |

To run one arm with the existing launcher:
```bash
cd ServeSID
export GRAPH_DIR=/path/to/processed-cache MODEL_DIR=/path/to/teacher BEAM_DUMP=/path/to/train-dump.pt
CATEGORY=Beauty ARCH=mlp SEEDS="42 43 44" bash scripts/train_paper_ablation.sh shuffled
```
Select `*_best_val_ndcg10.pth` for reported test evaluation. These new
implementations have **not** been established as the exact historical
implementations used to produce Table 2/S2/S3. Verify original checkpoint,
optimizer config, code revision and seed run logs **before** attributing old
numbers to this branch; rerun the affected arms if that evidence is missing.
Shuffled is stochastic only during training and uses the global seeded PyTorch
RNG; independent request draws are within valid candidate slots.

## Steam-100K

`genrec.datasets.Steam100K` is now registered. It consumes author-preprocessed
`user_item_sequences.jsonl` with lines
`{"user":"u1","items":["10","20","30","40"]}` where items are already in
chronological order, and (for sentence metadata)
`item_text.jsonl` with `{"item":"10","title":"...","description":"..."}`.
Optional `id_mapping.json` supplies exact `id2item` and `id2user` arrays,
both beginning with `"[PAD]"`. Without that file it deterministically maps
IDs; this **must not** be assumed compatible with old Steam checkpoints.

Set `DATASET=Steam100K CATEGORY=Steam100K`, provide a processed Steam root
via `STEAM100K_ROOT` or `steam100k_root`, and supply the matching teacher
checkpoint, SID mapping, valid-SID set, train beam dump and cached encoder
features. The adapter uses `last_out` chronological leave-one-out.
**No raw Steam converter, original processed data or historical checkpoint
is present in this repository.** This adapter is infrastructure, **not a
reproduction of Table S1/S2**; confirm the original Steam evaluation split.

## Teacher scoring convention

HCDA uses cached teacher totals `bm_sc`; RRDA uses four cached teacher
SID-path increments `step_logp`. An EOS or other candidate-specific
remainder may explain why their sum and the total are unequal.

```bash
python scripts/audit_beam_score_convention.py /path/to/train_beam_dump.pt \
  --tolerance 1e-5 --output-json /path/to/score_audit.json
```
The report measures **within-request spread** of the difference, not just
average difference. Constant per-request offsets cancel in ranking; varying
offsets do not. Do not overwrite teacher scores or silently substitute
the four-step sums. Use `--fail-on-nonconstant` when your intended scoring
contract demands constant offsets.

## Mechanism metrics and experiment provenance

Recompute the nine retained diagnostics, including Base-KD-*fixed* Huber-D/I
replacement-gain weights, from **aligned** exact arrays:

```bash
python scripts/recompute_paper_mechanisms.py /path/to/top40_aligned.npz \
  --output-json /path/to/mechanism_verified.json
```

See the script docstring for required arrays, units and shape. It fails on
unsorted teacher support, duplicate SIDs, missing Base-KD reference increments,
or missing training-derived Huber scales; it never inserts synthetic metric
values. A matching NPZ export needs the actual teacher cache, evaluated
student and Base-KD checkpoints, and their common support. Figure 1's
gold-prefix interpolation still needs the original per-prefix conditionals and
the experimental interpolation/evaluation script.

Use the manifest verifier to pin five **existing real files** to each
dataset–architecture–method–seed result:

```bash
python scripts/verify_paper_artifacts.py /path/to/paper_runs.json
```
Example `paper_runs.json` schema (values below are PLACEHOLDERS, not data):
```json
{
  "runs": [{
    "dataset": "Beauty", "arch": "mlp", "method": "GKD", "seed": 42,
    "code_commit": "0000000000000000000000000000000000000000",
    "checkpoint": {"path": "best_val.pth", "sha256": "REPLACE"},
    "eval_json": {"path": "evaluation.json", "sha256": "REPLACE"},
    "teacher_dump": {"path": "teacher_train.pt", "sha256": "REPLACE"},
    "semantic_ids": {"path": "sem_ids.pt", "sha256": "REPLACE"},
    "train_manifest": {"path": "data_protocol.json", "sha256": "REPLACE"}
  }]
}
```
The historical GKD/DistiLLM-2 results and multi-seed aggregate results
have **not been verified** against post-fix runs; the existing
`scripts/train_kd_baselines_all_arches.sh` supports rerunning both.
The manuscript's mechanism figure values and CSVs should not be treated
as verified solely because an evaluation function now exists.
