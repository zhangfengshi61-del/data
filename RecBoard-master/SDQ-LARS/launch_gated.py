#!/usr/bin/env python
"""Launch the margin-gated robustness round on selected free GPUs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--jobs", nargs="+", default=["Beauty:0", "Toys:3"],
                   help="dataset:gpu pairs; Sports can be added after its current run ends")
    args = p.parse_args()
    root = HERE / "runs" / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    for spec in args.jobs:
        ds, gpu = spec.split(":", 1)
        if ds not in {"Beauty", "Sports", "Toys"}:
            p.error(f"unknown dataset: {ds}")
        cmd = [sys.executable, "-u", str(HERE / "run_experiment.py"),
               "--dataset", ds, "--variant", "gated_margin", "--gpu", gpu,
               "--run-name", args.run_name]
        log_path = root / f"worker_gated_margin_{ds}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"dataset": ds, "variant": "gated_margin", "gpu": int(gpu),
                     "pid": process.pid, "command": cmd, "worker_log": str(log_path)})
    (root / "launch.json").write_text(json.dumps(jobs, indent=2))
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
