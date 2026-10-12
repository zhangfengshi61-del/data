#!/usr/bin/env python3
"""Audit a single-positive Recall/NDCG report for rank-discount invariants.

Single held-out positive, identical ranked list and denominator at K=5/10:
(R10-R5)/log2(11) <= N10-N5 <= (R10-R5)/log2(7).
The mean of seed metrics obeys the same inequalities.
Allow independent four-decimal rounding of each reported mean.

Usage:
  python scripts/check_rank_metric_invariants.py --report-only
  python scripts/check_rank_metric_invariants.py  # strict: fail on inconsistency
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "data" / "ablation_user_supplied_mean_sd.csv"
OUT = HERE / "metric_invariant_audit.txt"
DISC_MIN = 1.0 / math.log2(11.0)
DISC_MAX = 1.0 / math.log2(7.0)


def audit(path=SOURCE):
    entries = defaultdict(dict)
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(line for line in stream if not line.lstrip().startswith("#")):
            key = row["dataset"], row["student"], row["method"]
            entries[key][row["metric"]] = float(row["mean"])
    problems = []
    inspected = 0
    for (dataset, student, method), scores in sorted(entries.items()):
        try:
            r5, r10 = scores["R@5"], scores["R@10"]
            n5, n10 = scores["NDCG@5"], scores["NDCG@10"]
        except KeyError as exc:
            problems.append((dataset, student, method, f"Missing {exc}"))
            continue
        inspected += 1
        dr, dn = r10-r5, n10-n5
        # Each of four rounded values contributes up to 5e-5.
        # Thus dr, dn have a worst-case ±1e-4 uncertainty.
        if dn-1e-4 > DISC_MAX*(dr+1e-4)+1e-12:
            problems.append((dataset, student, method,
                f"upper bound violated: ΔR={dr:.4f}, ΔN={dn:.4f}, "
                f"max_disc={DISC_MAX:.9f}"))
        elif dn+1e-4 < DISC_MIN*(dr-1e-4)-1e-12:
            problems.append((dataset, student, method,
                f"lower bound violated: ΔR={dr:.4f}, ΔN={dn:.4f}, "
                f"min_disc={DISC_MIN:.9f}"))
        elif r10+1e-4 < r5 or n10+1e-4 < n5:
            problems.append((dataset, student, method,
                             "R@10 or NDCG@10 < @5"))
    return inspected, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
        help="Report inconsistencies without failing the PDF builder")
    args = parser.parse_args()
    checked, problems = audit()
    lines = [
        "Single-positive Recall/NDCG@5/10 necessary-condition audit",
        f"Checked: {checked} dataset/student/ablation rows.",
        f"Violations: {len(problems)}",
    ]
    for ds, st, arm, detail in problems:
        lines.append(f"  {ds} / {st} / {arm}: {detail}")
    if problems:
        lines.append("Resolution requires evaluator ranks for the same "
                     "split and checkpoint; aggregate means cannot "
                     "determine which input cell to correct.")
    report = "\n".join(lines) + "\n"
    OUT.write_text(report, encoding="utf-8")
    print(report)
    if problems and not args.report_only:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
