#!/usr/bin/env python
"""CLI for the PRQ-SID pipeline.

Examples
--------
python build_prqsid.py teacher  --category Beauty --device cuda:3
python build_prqsid.py profiles --category Beauty --device cuda:3
python build_prqsid.py quantize --category Beauty --mode pred_rq --sem-weight 0.5
python build_prqsid.py diagnose --category Beauty --sid pred_rq
python build_prqsid.py correct  --category Beauty --sid pred_rq
python build_prqsid.py export   --category Beauty --sid pred_rq_corrected
python build_prqsid.py all      --category Beauty --device cuda:3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import prq_lib as L

ART = Path(__file__).resolve().parent / "artifacts"


def art(category: str) -> Path:
    p = ART / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_teacher(args) -> None:
    L.train_teacher(
        args.category, art(args.category), max_len=args.max_len,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dropout=args.dropout, lr=args.lr, batch_size=args.batch_size,
        epochs=args.teacher_epochs, seed=args.seed, device=args.device,
        log_every=args.log_every,
    )


def _load_teacher(category: str, device: str):
    ckpt = torch.load(art(category) / "teacher.pt", map_location=device,
                      weights_only=False)
    cfg = ckpt["config"]
    model = L.SASRecTeacher(cfg["n_items"], cfg["max_len"], cfg["d_model"],
                            cfg["n_layers"], cfg["n_heads"], cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()


def cmd_profiles(args) -> None:
    teacher = _load_teacher(args.category, args.device)
    L.compute_profiles(
        args.category, teacher, art(args.category),
        n_anchors=args.n_anchors, max_len=args.max_len, seed=args.seed,
        device=args.device, batch_size=args.batch_size,
    )


def _load_profiles(category: str) -> np.ndarray:
    data = np.load(art(category) / "profiles.npz")
    return data["P"]


def cmd_quantize(args) -> None:
    P = _load_profiles(args.category)
    emb = L.load_sent_emb(args.category)
    result = L.predictive_residual_quantization(
        P, emb, n_levels=args.n_levels, n_codes=args.n_codes,
        sem_weight=args.sem_weight, niter=args.kmeans_iters, seed=args.seed,
        mode=args.mode,
    )
    name = args.mode if args.sem_weight == 0 else f"{args.mode}_sem{args.sem_weight:g}"
    np.save(art(args.category) / f"sid_{name}.npy", result["sid"])
    np.save(art(args.category) / f"codebooks_{name}.npy", result["codebooks"])
    np.save(art(args.category) / "rho.npy", result["rho"])
    print(f"[quantize] {args.category}/{name}: saved SID {result['sid'].shape}")


def _load_sid(category: str, sid_name: str) -> np.ndarray:
    return np.load(art(category) / f"sid_{sid_name}.npy")


def cmd_diagnose(args) -> None:
    P = _load_profiles(args.category)
    sid = _load_sid(args.category, args.sid)
    report = {"category": args.category, "sid": args.sid,
              "n_codes": int(sid.max() + 1)}
    for k in range(sid.shape[1]):
        report[f"D_{k + 1}"] = L.prefix_distortion(P, sid, k)
    for beam in (10, 20, 30):
        report[f"R_{beam},1"] = L.beam_regret(P, sid, beam=beam, top_r=1)
        report[f"R_{beam},5"] = L.beam_regret(P, sid, beam=beam, top_r=5)
    report["collision_rate"] = 1.0 - len(set(map(tuple, sid.tolist()))) / len(sid)
    out = art(args.category) / f"diagnostics_{args.sid}.json"
    out.write_text(json.dumps(report, indent=2))
    print("[diagnose]", json.dumps(report, indent=2))


def cmd_correct(args) -> None:
    P = _load_profiles(args.category)
    sid = _load_sid(args.category, args.sid)
    corrected = L.hard_sid_correction(
        P, sid, lambda_r=args.lambda_r, beam=args.beam, top_r=args.top_r,
        n_anchors_diag=args.diag_anchors, max_swaps=args.max_swaps,
        max_evals=args.max_evals, pool_size=args.pool_size, seed=args.seed,
    )
    np.save(art(args.category) / f"sid_{args.sid}_corrected.npy", corrected)
    print(f"[correct] saved sid_{args.sid}_corrected.npy")


def cmd_export(args) -> None:
    items = L.load_items(args.category)
    sid = _load_sid(args.category, args.sid)
    out = Path(args.out) if args.out else (
        L.SENT_EMB_ROOT / args.category / "processed"
        / f"sentence-t5-base_{args.vq_method}_{sid.shape[1]}x256.sem_ids"
    )
    L.export_sem_ids(sid, items, out)
    print(f"[export] wrote {out} ({len(items)} items)")


def cmd_all(args) -> None:
    cmd_teacher(args)
    cmd_profiles(args)
    args.mode = "pred_rq"
    args.sem_weight = args.sem_weight
    args.n_levels = 3
    args.n_codes = 256
    args.kmeans_iters = 30
    cmd_quantize(args)
    name = "pred_rq" if args.sem_weight == 0 else f"pred_rq_sem{args.sem_weight:g}"
    args.sid = name
    cmd_diagnose(args)
    if args.hard_correct:
        cmd_correct(args)
        args.sid = f"{name}_corrected"
        cmd_diagnose(args)
    args.vq_method = "prq"
    cmd_export(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["teacher", "profiles", "quantize",
                                       "diagnose", "correct", "export", "all"])
    p.add_argument("--category", default="Beauty")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2025)
    # teacher
    p.add_argument("--max-len", type=int, default=20)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--teacher-epochs", type=int, default=60)
    p.add_argument("--log-every", type=int, default=10)
    # profiles
    p.add_argument("--n-anchors", type=int, default=512)
    # quantize
    p.add_argument("--mode", default="pred_rq", choices=["pred_rq", "plain_rq"])
    p.add_argument("--sem-weight", type=float, default=0.0)
    p.add_argument("--n-levels", type=int, default=3)
    p.add_argument("--n-codes", type=int, default=256)
    p.add_argument("--kmeans-iters", type=int, default=30)
    # diagnose / correct / export
    p.add_argument("--sid", default="pred_rq")
    p.add_argument("--lambda-r", type=float, default=1.0)
    p.add_argument("--beam", type=int, default=20)
    p.add_argument("--top-r", type=int, default=1)
    p.add_argument("--diag-anchors", type=int, default=128)
    p.add_argument("--max-swaps", type=int, default=300)
    p.add_argument("--max-evals", type=int, default=4000)
    p.add_argument("--pool-size", type=int, default=2000)
    p.add_argument("--hard-correct", action="store_true")
    p.add_argument("--vq-method", default="prq")
    p.add_argument("--out", default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    {"teacher": cmd_teacher, "profiles": cmd_profiles,
     "quantize": cmd_quantize, "diagnose": cmd_diagnose,
     "correct": cmd_correct, "export": cmd_export,
     "all": cmd_all}[args.command](args)
