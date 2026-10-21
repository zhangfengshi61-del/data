#!/usr/bin/env python3
"""Audit whether cached teacher total scores equal four SID log increments.

A constant per-request remainder does not affect rankings; candidate-dependent
remainders (e.g. EOS) can change score comparisons. This script only measures
the distinction and never changes old caches or reported metrics.

Only load author-trusted torch checkpoint/dump files.
"""
import argparse
import json
from pathlib import Path

import torch


def audit(data, tolerance=1e-5):
    for key in ("codes", "scores", "nvalid", "step_logp"):
        if key not in data:
            raise ValueError("Beam dump is missing " + key)
    codes = data["codes"]
    scores = data["scores"].detach().float().cpu()
    step = data["step_logp"].detach().float().cpu()
    count = data["nvalid"].detach().long().cpu()
    if codes.ndim != 3 or codes.shape[-1] != 4:
        raise ValueError("codes must be (requests, width, 4)")
    N, M, _ = codes.shape
    if scores.shape != (N, M) or step.shape != (N, M, 4) or count.shape != (N,):
        raise ValueError("Beam dump dimensions disagree")
    if N < 1 or M < 1 or bool((count < 1).any()) or bool((count > M).any()):
        raise ValueError("Invalid beam dump row counts or widths")
    valid = torch.arange(M).unsqueeze(0) < count.unsqueeze(1)
    remainder = scores - step.sum(-1)
    if not torch.isfinite(remainder[valid]).all():
        raise ValueError("Nonfinite teacher score or increment on a valid item")
    high = remainder.masked_fill(~valid, -float("inf")).max(-1).values
    low = remainder.masked_fill(~valid, float("inf")).min(-1).values
    per_request_spread = high - low
    candidate_dependent = per_request_spread > tolerance
    return {
        "requests": N,
        "candidate_slots": M,
        "mean_valid_candidates": round(count.float().mean().item(), 5),
        "tolerance_nats": tolerance,
        "nonconstant_remainder_requests": int(candidate_dependent.sum().item()),
        "nonconstant_remainder_fraction": round(candidate_dependent.float().mean().item(), 8),
        "max_within_request_remainder_spread_nats": float(per_request_spread.max().item()),
        "median_within_request_remainder_spread_nats": float(per_request_spread.median().item()),
        "p95_within_request_remainder_spread_nats": float(torch.quantile(per_request_spread, .95).item()),
        "mean_remainder_nats": float(remainder[valid].mean().item()),
        "conclusion": ("Candidate-dependent teacher cache remainder observed"
                       if candidate_dependent.any() else
                       "Four-step and complete scores differ at most by a per-request constant"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("beam_dump", type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--fail-on-nonconstant", action="store_true")
    args = parser.parse_args()
    if args.tolerance < 0:
        parser.error("--tolerance must be nonnegative")
    data = torch.load(args.beam_dump, map_location="cpu", weights_only=False)
    report = audit(data, tolerance=args.tolerance)
    report["source_file"] = str(args.beam_dump)
    print(json.dumps(report, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.fail_on_nonconstant and report["nonconstant_remainder_requests"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
