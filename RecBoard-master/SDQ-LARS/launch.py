#!/usr/bin/env python
"""Detach the explicitly configured first round, preserving logs and exit status."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3, 0],
                   help="GPUs for the submitted jobs")
    args = p.parse_args()
    if len(set(args.gpus)) != 4:
        p.error("each worker requires a distinct GPU")
    root = HERE / "runs" / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    jobs = []
    requested = [("Beauty", "main"), ("Sports", "main"),
                 ("Toys", "main"), ("Beauty", "no_bridge")]
    if len(args.gpus) != len(requested):
        p.error("the original launcher expects four GPUs")
    for (ds, variant), gpu in zip(requested, args.gpus):
        cmd = [sys.executable, "-u", str(HERE / "run_experiment.py"),
               "--dataset", ds, "--variant", variant, "--gpu", str(gpu), "--run-name", args.run_name]
        log_path = root / f"worker_{variant}_{ds}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(cmd, cwd=HERE, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        jobs.append({"dataset": ds, "variant": variant, "gpu": gpu,
                     "pid": process.pid, "command": cmd, "worker_log": str(log_path)})
    (root / "launch.json").write_text(json.dumps(jobs, indent=2))
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
