"""Generate a validation-only diagnosis and a baseline-preserving selector."""
import json
import pickle
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "SDQ-LARS"


def best_valid(log):
    vals = []
    for line in log.read_text(errors="replace").splitlines():
        m = re.search(r"VALID @Epoch:\s*(\d+).*?NDCG@10 Avg: ([0-9.]+)", line)
        if m:
            vals.append((int(m.group(1)), float(m.group(2))))
    return max(vals, key=lambda x: x[1]) if vals else None


def selected_test(path):
    with path.open("rb") as f:
        return pickle.load(f)["best"]


def main():
    rows = []
    for ds in ("Beauty", "Sports", "Toys"):
        base_dir = ROOT / "SDQ-403/logs/SDQ-T5-VAE" / f"Amazon2014{ds}_550_LOU/t5vae"
        base_scores = selected_test(base_dir / "data/best.pkl")
        base_valid = best_valid(base_dir / "log.txt")
        candidates = [{"method": "SDQ", "valid": base_valid, "test": base_scores,
                       "source": str(base_dir)}]
        for variant, desc in (("main", "LARS-20260922_r1-main-T5"),
                              ("gated_margin", "LARS-20260923_gated-gated_margin-T5")):
            d = HERE / "logs" / desc / f"Amazon2014{ds}_550_LOU/t5"
            if (d / "data/best.pkl").exists():
                candidates.append({"method": f"LARS-{variant}", "valid": best_valid(d / "log.txt"),
                                   "test": selected_test(d / "data/best.pkl"), "source": str(d)})
        # This is decided by validation NDCG@10 only; test is never inspected here.
        candidates = [c for c in candidates if c["valid"]]
        selected = max(candidates, key=lambda c: c["valid"][1])
        rows.append({"dataset": ds, "candidates": candidates, "selected_by_valid_ndcg10": selected})
    out = HERE / "round_analysis.json"
    out.write_text(json.dumps(rows, indent=2))
    md = ["# Robust LARS-SDQ round analysis", "", "Selection uses validation NDCG@10 only.", "",
          "| Dataset | SDQ valid | LARS valid | Gated valid | Selected | Selected test NDCG@10 |",
          "|---|---:|---:|---:|---|---:|"]
    for row in rows:
        vals = {c["method"]: c["valid"][1] for c in row["candidates"]}
        selected = row["selected_by_valid_ndcg10"]
        md.append(f"| {row['dataset']} | {vals.get('SDQ', float('nan')):.6f} | "
                  f"{vals.get('LARS-main', float('nan')):.6f} | "
                  f"{vals.get('LARS-gated_margin', float('nan')):.6f} | "
                  f"{selected['method']} | {selected['test']['NDCG@10']:.6f} |")
    (HERE / "ROUND_ANALYSIS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
