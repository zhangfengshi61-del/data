#!/usr/bin/env python
"""Keep the local result table current until every submitted worker terminates."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
args = parser.parse_args()
root = HERE / "runs" / args.run_name
jobs = json.loads((root / "launch.json").read_text())
while True:
    subprocess.run([sys.executable, str(HERE / "status.py"), "--run-name", args.run_name],
                   cwd=HERE, check=True)
    terminal = []
    for job in jobs:
        path = root / job["variant"] / job["dataset"] / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        done = state.get("status") in {"complete", "failed"}
        if not done and not Path(f"/proc/{job['pid']}").exists():
            print(f"Worker disappeared: {job['dataset']}/{job['variant']}", flush=True)
            done = True
        terminal.append(done)
    if all(terminal):
        break
    time.sleep(60)
print("All workers have terminated; RESULTS.md includes completed results only.", flush=True)
