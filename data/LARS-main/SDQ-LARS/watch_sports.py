#!/usr/bin/env python
import json, time
from pathlib import Path

BASE = Path(__file__).resolve().parent / "runs" / "20260928_lars_grid_st"
SDQ_NDCG10 = 0.026651
CAND = [c for c in sorted(d.name for d in BASE.iterdir()) if c.startswith("c")]

def results():
    out = {}
    for c in CAND:
        r = BASE / c / "Sports" / "result.json"
        if r.exists():
            out[c] = json.loads(r.read_text())["test_selected"]["NDCG@10"]
    return out

seen = set()
log = open("/tmp/opencode/sports_watch.log", "a")
while True:
    res = results()
    for c, v in res.items():
        if c not in seen:
            seen.add(c)
            line = f"{time.strftime('%H:%M:%S')} {c:<26} NDCG@10={v:.6f} rel={((v/SDQ_NDCG10)-1)*100:+.2f}%"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    if len(seen) == len(CAND):
        print("ALL DONE", flush=True)
        log.write("ALL DONE\n")
        break
    time.sleep(600)
log.close()
