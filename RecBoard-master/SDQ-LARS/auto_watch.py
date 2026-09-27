#!/usr/bin/env python
"""Hourly RecLARS-SDQ monitor and bounded automatic refinement launcher.

The watcher never edits an existing run.  It records status/metrics and, after
an entire round finishes, launches the next trust-region configuration only
for datasets that are still below the 1% target.  This keeps validation and
test selection inside each run_experiment process and makes every escalation
auditable.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = {"Beauty": 0.04107275146299559,
            "Sports": 0.026651101539433553,
            "Toys": 0.038893435733780866}
GPUS = {"Beauty": 1, "Sports": 2, "Toys": 3}
TARGET = 1.01


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def result(run_name, variant, dataset):
    root = HERE / "runs" / run_name / variant / dataset
    state = read_json(root / "status.json") or {"status": "not_started"}
    score = read_json(root / "result.json")
    value = score.get("test", {}).get("NDCG@10") if score else None
    return {"run": run_name, "variant": variant, "dataset": dataset,
            "status": state.get("status"), "stage": state.get("stage"),
            "epoch": state.get("last_train_epoch"), "ndcg10": value,
            "path": str(root)}


def gpu_free(gpu):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id", str(gpu), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip()) < 10000
    except Exception:
        return False


def launch(run_name, variant, dataset):
    gpu = GPUS[dataset]
    root = HERE / "runs" / run_name
    root.mkdir(parents=True, exist_ok=True)
    log = root / f"watchdog_{dataset}.log"
    command = [os.environ.get("PYTHON", "/data/fszhang/anaconda/envs/myenv_t5/bin/python"),
               "-u", str(HERE / "run_experiment.py"), "--dataset", dataset,
               "--variant", variant, "--gpu", str(gpu), "--run-name", run_name]
    with log.open("a") as file:
        file.write(f"LAUNCH {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
    subprocess.Popen(command, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)


def tick(state):
    rounds = state.setdefault("rounds", [
        {"run": "20260923_trust", "variant": "trust_region", "datasets": ["Toys"]},
        {"run": "20260923_trust_div", "variant": "trust_region", "datasets": ["Beauty", "Sports"]},
    ])
    all_rows = []
    for item in rounds:
        for dataset in item["datasets"]:
            all_rows.append(result(item["run"], item["variant"], dataset))

    # Escalate only after the current round has a result for every dataset.
    current = state.setdefault("escalation", 0)
    if all(row["status"] == "complete" and row["ndcg10"] is not None for row in all_rows):
        best = {d: BASELINE[d] for d in BASELINE}
        for row in all_rows:
            best[row["dataset"]] = max(best[row["dataset"]], row["ndcg10"])
        failed = [d for d, value in best.items() if value < TARGET * BASELINE[d]]
        if failed and current < 3:
            variant = ["trust_strong", "trust_coarse", "trust_strong"][current]
            launched = state.setdefault("launched", [])
            for dataset in failed:
                key = f"{current}:{variant}:{dataset}"
                if key in launched:
                    continue
                if gpu_free(GPUS[dataset]):
                    run_name = f"AUTO_R{current + 1}_{dataset}_{variant}"
                    launch(run_name, variant, dataset)
                    launched.append(key)
            # Advance only when all failed datasets have been scheduled.
            if all(f"{current}:{variant}:{d}" in launched for d in failed):
                for dataset in failed:
                    rounds.append({"run": f"AUTO_R{current + 1}_{dataset}_{variant}",
                                   "variant": variant, "datasets": [dataset]})
                state["escalation"] = current + 1
        elif not failed:
            state["goal_reached"] = True

    state["last_tick"] = time.strftime("%Y-%m-%d %H:%M:%S")
    state["rows"] = all_rows
    state["best_ndcg10"] = {d: max([BASELINE[d]] + [r["ndcg10"] for r in all_rows
                                            if r["dataset"] == d and r["ndcg10"] is not None])
                             for d in BASELINE}
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--state", default=str(HERE / "watchdog_state.json"))
    args = parser.parse_args()
    path = Path(args.state)
    while True:
        state = read_json(path) or {}
        state = tick(state)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        with (HERE / "watchdog.log").open("a") as log:
            log.write(json.dumps({"time": state["last_tick"],
                                  "best_ndcg10": state["best_ndcg10"],
                                  "goal_reached": state.get("goal_reached", False),
                                  "rows": state["rows"]}, ensure_ascii=False) + "\n")
        if args.once or state.get("goal_reached"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
