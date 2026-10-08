#!/usr/bin/env python
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent / "runs" / "20260928_lars_grid_st"
SDQ = {"HIT@1": 0.010113, "HIT@5": 0.031940, "HIT@10": 0.049385, "HIT@20": 0.071942,
       "NDCG@5": 0.021054, "NDCG@10": 0.026651, "NDCG@20": 0.032320,
       "MRR@5": 0.017495, "MRR@10": 0.019779, "MRR@20": 0.021317}
MAP = {"HITRATE@1": "HIT@1", "HITRATE@5": "HIT@5", "HITRATE@10": "HIT@10", "HITRATE@20": "HIT@20",
       "NDCG@5": "NDCG@5", "NDCG@10": "NDCG@10", "NDCG@20": "NDCG@20",
       "MRR@5": "MRR@5", "MRR@10": "MRR@10", "MRR@20": "MRR@20"}

rows = []
for d in sorted(BASE.iterdir()):
    r = d / "Sports" / "result.json"
    if r.exists():
        s = json.loads(r.read_text())["test_selected"]
        rows.append((d.name, s))
    else:
        rows.append((d.name, None))

print(f"{'candidate':<26} {'NDCG@10':>9} {'vs SDQ':>8}  status")
for name, s in rows:
    if s is None:
        print(f"{name:<26} {'-':>9} {'-':>8}  PENDING")
    else:
        ndcg = s["NDCG@10"]
        rel = ndcg / SDQ["NDCG@10"] - 1
        print(f"{name:<26} {ndcg:>9.6f} {rel*100:>+7.2f}%  DONE")
