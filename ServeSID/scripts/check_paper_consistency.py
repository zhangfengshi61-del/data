#!/usr/bin/env python3
"""Static, dependency-free checks for the WWW ServeSID objective/config contract.

This verifies source invariants; it does NOT validate checkpoints, datasets,
full numerical training, or the manuscript's reported results.
"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]

def config(name: str) -> dict:
    result = {}
    for line in (ROOT / "configs" / name).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if not k or k[0].isspace():
            continue
        v = v.strip().split(" #", 1)[0]
        try:
            result[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            result[k] = v
    return result

def require_equal(label: str, value, wanted):
    assert value == wanted, f"{label}: got {value!r}, expected {wanted!r}"

def check():
    files = [ROOT / "sid_mlp" / n for n in ("losses.py", "train_decoder.py", "kd_baselines.py", "dataset.py", "cli/train.py")]
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    configs = [
        "servesid_final.yaml", "abl_d4kl.yaml", "abl_daonly.yaml",
        "abl_hserve.yaml", "abl_hserve_pair.yaml", "abl_hserve_attruni.yaml",
        "sidmlp.yaml", "sidmlp_amazon2014_baseline.yaml",
        "verify_a0.yaml", "verify_a2.yaml", "verify_a5.yaml", "verify_a7.yaml", "verify_a8.yaml"
    ]
    for n in configs:
        c = config(n)
        require_equal(n + ".seed", c["seed"], 42)
        require_equal(n + ".seeds", c["seeds"], SEEDS)
    structural = ["servesid_final.yaml", "abl_d4kl.yaml", "abl_daonly.yaml", "abl_hserve.yaml", "abl_hserve_pair.yaml", "abl_hserve_attruni.yaml", "verify_a7.yaml", "verify_a8.yaml"]
    for n in structural:
        c = config(n)
        require_equal(n + ".loss", c["loss"], "ce")
        require_equal(n + ".d4_teacher", c["d4_teacher"], "ce")
    full = config("servesid_final.yaml")
    for key, wanted in {"pair_w": 0.0, "da_w": 1.0, "da_corr_w": 31.42,
                        "da_margin": 0.05, "serve_kl": 4, "list_topm": 64,
                        "corr_mode": "cf"}.items():
        require_equal("Full." + key, full[key], wanted)
    for n in ["abl_daonly.yaml", "abl_hserve_attruni.yaml", "verify_a7.yaml", "verify_a8.yaml"]:
        c = config(n)
        require_equal(n + ".pair_w", c["pair_w"], 0.0)
        require_equal(n + ".da_corr_w", c["da_corr_w"], 31.42)
        require_equal(n + ".da_w", c["da_w"], 1.0)
    pair = config("abl_hserve_pair.yaml")
    for key, wanted in {"pair_w": 1.0, "pair_margin": 0.05, "da_w": 0.0}.items():
        require_equal("PlainPair." + key, pair[key], wanted)
    require_equal("Uniform mode", config("abl_hserve_attruni.yaml")["corr_mode"], "uniform_valid")
    require_equal("R+ mode", config("verify_a8.yaml")["corr_mode"], "posres")
    require_equal("Baseline preserves token KD", config("sidmlp_amazon2014_baseline.yaml")["loss"], "combined")
    loss = (ROOT / "sid_mlp" / "losses.py").read_text()
    train = (ROOT / "sid_mlp" / "train_decoder.py").read_text()
    assert "F.softplus(margin - G_S)" in loss
    assert "eligible.to(dG_S.dtype)" in loss
    assert "weight_mode=weight_mode" in train
    assert "lam_da * corr_w * l_da" in train
    assert "best_val_ndcg10.pth" in train
    assert "refusing test fallback" in train
    dataset = (ROOT / "sid_mlp" / "dataset.py").read_text()
    assert "d['step_logp'] = d['step_logp'].gather(" in dataset
    launcher = (ROOT / "scripts" / "run_paper_seeds.sh").read_text()
    assert 'SEEDS="${SEEDS:-42 43 44}"' in launcher
    print(f"PASS: {len(files)} Python source files parsed; {len(configs)} configs audited; seeds={SEEDS}.")
    print("NOTE: No training, checkpoint selection, or metric replication was executed.")

if __name__ == "__main__":
    check()
