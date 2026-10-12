#!/usr/bin/env python3
"""Fail if sensitivity reference anchors drift from the published ablation means."""
import csv
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
source = PAPER / "data" / "ablation_user_supplied_mean_sd.csv"
tex = PAPER / "supplement.tex"

with source.open(encoding="utf-8", newline="") as stream:
    main = {}
    for r in csv.DictReader(line for line in stream if not line.lstrip().startswith("#")):
        if r["student"] == "MLP":
            main[(r["dataset"], r["method"], r["metric"])] = float(r["mean"])

tables = [
    ("tab:sup-sensitive-m", "64", "HServe only",
     ("R@5", "NDCG@5", "R@10", "NDCG@10")),
    ("tab:sup-sensitive-b", "40", "HServe + Plain Pair",
     ("R@5", "NDCG@5", "R@10", "NDCG@10")),
    ("tab:sup-sensitive-beta", "0", "HServe only",
     ("R@10", "NDCG@10")),
    ("tab:sup-sensitive-beta", "31.42", "Full ServeSID",
     ("R@10", "NDCG@10")),
    ("tab:sup-sensitive-tau", "1", "HServe only",
     ("R@10", "NDCG@10")),
]
domains = {"Beauty": "Beauty", "Toys": "Toys", "Video Games": "VG"}
text = tex.read_text(encoding="utf-8")
checked = 0
for label, value, method, metrics in tables:
    start = text.index(r"\label{" + label + "}")
    end = text.index(r"\end{table}", start)
    current = ""
    seen = set()
    for line in text[start:end].splitlines():
        if "&" not in line or r"\textsuperscript" not in line:
            continue
        cells = [c.strip().removesuffix("\\").strip()
                 for c in line.split("&")]
        if len(cells) < len(metrics) + 2:
            continue
        if cells[0] in domains:
            current = cells[0]
        elif cells[0] != "":
            continue
        param = cells[1].split("\\", 1)[0].strip()
        if param != value:
            continue
        if not current:
            raise AssertionError(f"{label}: missing dataset")
        name = domains[current]
        if current in seen:
            raise AssertionError(f"{label}: duplicate {current} anchor")
        seen.add(current)
        for metric, cell in zip(metrics, cells[2:]):
            expected = main[(name, method, metric)]
            actual = float(cell)
            if round(actual, 4) != round(expected, 4):
                raise AssertionError(
                    f"{label} {current} {metric}: {actual:.4f} "
                    f"does not match main {method} {expected:.4f}")
            checked += 1
    if seen != set(domains):
        raise AssertionError(f"{label}: anchored domains {seen}")
if checked != 42:
    raise AssertionError(f"Expected 42 repeated mean values, got {checked}")
print(f"PASS: {checked} anchor mean values across 15 rows match main ablations.")
