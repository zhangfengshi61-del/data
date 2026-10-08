#!/usr/bin/env python
"""Show current stages and collect validation-selected final test metrics."""
import argparse
import json
import pickle
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "SDQ-403"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    root = HERE / "runs" / args.run_name
    rows = []
    for path in sorted(root.glob("*/*/status.json")):
        state = json.loads(path.read_text())
        folder = path.parent
        stage = state.get("stage", "vae")
        log = folder / f"{stage}.log"
        epochs = re.findall(r"TRAIN @Epoch:\s*(\d+)", log.read_text(errors="replace")) if log.exists() else []
        state["last_train_epoch"] = int(epochs[-1]) if epochs else 0
        if state["status"].startswith("training"):
            pid = state.get("child_pid")
            if pid and not Path(f"/proc/{pid}").exists():
                state["process_note"] = "child missing; check worker status/log"
        rows.append(state)
        print(f"{state['variant']:10} {state['dataset']:7} GPU={state['gpu']} "
              f"{state['status']:14} epoch={state['last_train_epoch']} pid={state.get('child_pid', '-')}")
        if "error" in state:
            print("  ERROR:", state["error"])
    table = ["# LARS-SDQ first-round results", "", "Only completed runs enter this table.", "",
             "| Method | Dataset | Test NDCG@10 | Test Hit@10 | Test NDCG@20 |",
             "|---|---|---:|---:|---:|"]
    for ds in ("Beauty", "Sports", "Toys"):
        with (BASE / "logs/SDQ-T5-VAE" / f"Amazon2014{ds}_550_LOU/t5vae/data/best.pkl").open("rb") as f:
            values = pickle.load(f)["best"]
        table.append(f"| SDQ baseline | {ds} | {values['NDCG@10']:.6f} | {values['HITRATE@10']:.6f} | {values['NDCG@20']:.6f} |")
    for path in sorted(root.glob("*/*/result.json")):
        r = json.loads(path.read_text())
        v = r["test"]
        table.append(f"| LARS {r['variant']} | {r['dataset']} | {v['NDCG@10']:.6f} | {v['HITRATE@10']:.6f} | {v['NDCG@20']:.6f} |")
    (root / "RESULTS.md").write_text("\n".join(table) + "\n")
    (root / "status_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"Result table: {root / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
