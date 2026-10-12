# J4 cost report and historical M/B sweeps — provenance and consistency

This is a **source ledger**, not a rerun or independently verified GPU result.
It records (1) the user's 2026-10-12 J4 cost report and (2) the uploaded
`20971_ServeSID_Serving_Decisio_Technical Supplement.pdf` (4 pages).
The latter documents a **different older submission** ("Fidelity-Constrained
Acceleration for Verify-Free Semantic-ID Recommendation"); it is not evidence
that the current Beauty/Toys/VG no-direct-pair Full objective was retrained.

## Current paper's four independent dimensions

| Symbol | Meaning | Current paper |
|---|---|---:|
| M | cached complete teacher-candidate support for HServe | 64 |
| K_train | cutoff for teacher-preferred paired items | 10 |
| B | last allowed outsider rank in DA-CDRS pair formation | 40 |
| W | online student beam width | 50 |

**Code invariant:** Full cannot launch when the teacher dump has fewer than M
candidate slots or B > M. A dump with width 40 does not become M=64
just because `list_topm: 64` is set.

## Historical exploratory scans (older objective and Sci domain)

Uploaded technical supplement p.3 Table 6, Scientific (Sci), seed 42,
**HServe-only** (not modern Full):

| M | Jac@10 | R@10 | N@10 |
|---:|---:|---:|---:|
| 20 | 0.6451 | 0.0457 | 0.0242 |
| 40 | 0.6668 | 0.0461 | 0.0244 |
| 64 | 0.6731 | 0.0459 | 0.0244 |

Uploaded p.3 Table 7, Sci, seed 42, **HServe + Plain Pair** with 10
fixed negative samples (not modern no-direct-pair Full):

| B | 20 | 30 | 40 | 50 | 60 |
|---|---:|---:|---:|---:|---:|
| Jac@10 | 0.6633 | 0.6715 | 0.6739 | 0.6755 | 0.6757 |

These scans motivated the working settings M=64 and B=40.
They must **not** be described as current Full-loss robustness results
or combined numerically with Beauty/Toys/VG results.

## User-reported J4 cost audit, Beauty 2014

- Training: 131,413 packed rows, 131,221 mapped rows; test 22,363
  packed / 22,171 mapped rows.
- J4 training probe uses **40 teacher candidates x 4 SID depths**, while
  current Full paper specifies **M=64**; B remains 40 in both. As a
  result, J4 training time/memory are illustrative costs for M=40,
  not the confirmed per-epoch overhead of M=64.
- Shared cache: Beauty teacher export 26.2 min, beam-dump 105.4 MB,
  teacher-top-K 8.5 MB, fp32 k0 hidden-state cache 5.7 GB.
  This is shared **in the J4 paired-run protocol**, not required for
  every possible gold-prefix KD implementation.
- Controlled 3-epoch train-step timing (beam validation excluded):
  Base MLP 4.6 s/epoch, Full MLP 11.5 s/epoch; Base GRU 5.5,
  Full GRU 11.9, Full CNN 14.5. These are train-step measures, not
  total epochs including validation.
- Historical 200-epoch complete run wall time: Base MLP 27 min,
  Full MLP 49 min, Full CNN 65 min. Different measurement boundary
  from the 3-epoch train-step probe.
- Peak reported training CUDA allocation: Base 7.4 GB,
  Full 9.2–9.5 GB (same 2014 Beauty setup).
- Student inference includes frozen T5 encoder before student decoding.
  For independent runs at batch 32, W=50:
  Base MLP 2,769 req/s, Full MLP 3,036;
  Base GRU 3,445, Full GRU 3,423;
  Base CNN 3,028, Full CNN 3,024.
  Differences are expected measurement noise, not a distillation speedup.
- TIGER reference 1,346 req/s was measured at **batch 64, beam 50**.
  Do not claim an iso-batch teacher/student speedup or mix it with
  NDCG of the beam-20 teacher reference.
- J4 reported Base-MLP NDCG@10 = 0.0365 while the current paper's
  main table reports 0.0356. Therefore **do not use J4 NDCG
  values for the accuracy table**; it is a checkpoint/protocol audit item.
- J4 reported inference peak allocations of 0.07 GB for student
  MLP/GRU, 0.11 GB CNN, versus 0.65 GB teacher while the student
  inference path still loads and runs the T5 encoder. These figures
  cannot represent **full-pipeline resident GPU peak allocation**.
  They are retained here as originally reported but **excluded from
  publication tables** until the measurement scope is established.
- The report's label "4 x RTX 4090 (48 GB)" is not enough to verify
  actual GPU device memory; the new benchmark records both the
  runtime GPU name and its reported total physical memory. No 48 GB
  claim should appear in the paper without that device record.

## Reproduction

The previous cost scripts described as
`ServeSID-Amazon2014/scripts/bench_infer_cost.py` and raw
`verification/bench_*.json` / `logs/bench_*.log` were not present
in the inspected `ServeSID/` GitHub path. The newly added
`ServeSID/scripts/bench_infer_cost.py` is a separate **reproducible
student path** using `sid_mlp.cli.infer`: for each repetition it saves
one ordinary inference throughput run and one synchronized,
post-warmup per-phase and full-resident-memory audit. It does not
automatically reproduce historical teacher timing.

For paper-aligned **M=64** cost measurements, first regenerate the
full-width teacher beam dump and run Full (without direct Pair Loss)
with `BENCH_COST=1`. Compare training with the exact same dataset,
epoch and validation schedule, rather than comparing 3-epoch
train-step probes to total 200-epoch walls.
