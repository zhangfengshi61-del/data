#!/usr/bin/env python3
"""Recompute Figure 1's gold-prefix interpolation on fixed candidate supports.

Required aligned npz:
  sid (N,M,4), gold_sid (N,4), nvalid (N,)
  teacher_scores (N,M), student_steps (N,M,4)
  teacher_gold_probs, student_gold_probs (N,4,256)

The *_gold_probs must be conditional next-token probability distributions
at the *ground-truth prefix* for each depth, with the same valid continuation
normalization used by the student candidate scorer. Student path increments
for all candidate SIDs must be scored by the actual inference graph.

For alpha, ONLY the conditional distribution at each ground-truth prefix is
mixed: (1-alpha)*student + alpha*teacher. Candidate paths that have already
diverged retain their student conditional score. No teacher probabilities are
mixed into other prefixes. This is post-hoc analysis, not training.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def evaluate(data, alphas=(0.0, 0.5), topk=10, outsider=40):
    keys = ("sid", "gold_sid", "nvalid", "teacher_scores",
            "student_steps", "teacher_gold_probs", "student_gold_probs")
    for key in keys:
        if key not in data:
            raise ValueError("Missing aligned input " + key)
    sid = np.asarray(data["sid"])
    gold = np.asarray(data["gold_sid"])
    counts = np.asarray(data["nvalid"])
    teacher = np.asarray(data["teacher_scores"], dtype=np.float64)
    steps = np.asarray(data["student_steps"], dtype=np.float64)
    pt = np.asarray(data["teacher_gold_probs"], dtype=np.float64)
    ps = np.asarray(data["student_gold_probs"], dtype=np.float64)
    if sid.ndim != 3 or sid.shape[-1] != 4:
        raise ValueError("sid must have four token positions")
    n, m, _ = sid.shape
    if (n < 1 or gold.shape != (n, 4) or counts.shape != (n,) or
        teacher.shape != (n, m) or steps.shape != (n, m, 4) or
        pt.shape != (n, 4, 256) or ps.shape != (n, 4, 256)):
        raise ValueError("Figure 1 inputs are not aligned")
    if not 0 < topk < outsider <= m or np.any(counts < 1) or np.any(counts > m):
        raise ValueError("Invalid cutoff or candidate counts")
    if np.any(sid < 0) or np.any(sid >= 256) or np.any(gold < 0) or np.any(gold >= 256):
        raise ValueError("SID must contain codebook-space digits 0..255")
    if not (np.isfinite(pt).all() and np.isfinite(ps).all() and
            (pt >= 0).all() and (ps >= 0).all()):
        raise ValueError("Gold-prefix probabilities must be finite and nonnegative")
    if (np.max(np.abs(pt.sum(-1) - 1)) > 1e-4 or
            np.max(np.abs(ps.sum(-1) - 1)) > 1e-4):
        raise ValueError("Gold-prefix probability rows must sum to one")
    vals = []
    for alpha in alphas:
        alpha = float(alpha)
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0,1]")
        token_kl = 0.
        item_kl = 0.
        violations = 0
        pairs = 0
        mixed = (1 - alpha) * ps + alpha * pt
        token_kl = float(np.mean(np.sum(
            pt * (np.log(np.maximum(pt, 1e-30)) -
                  np.log(np.maximum(mixed, 1e-30))), axis=-1)))
        for i in range(n):
            c = int(counts[i])
            cand = sid[i, :c]
            tscore = teacher[i, :c]
            base = steps[i, :c].sum(-1)
            if not np.isfinite(tscore).all() or not np.isfinite(base).all():
                raise ValueError("Invalid teacher/student candidate scores")
            if np.any(np.diff(tscore) > 1e-7):
                raise ValueError("Teacher candidates must be sorted in descending score order")
            update = np.zeros(c)
            for depth in range(4):
                matches = ((cand[:, :depth] == gold[i, :depth]).all(-1)
                           if depth else np.ones(c, dtype=bool))
                codes = cand[:, depth]
                q0 = ps[i, depth, codes]
                q1 = mixed[i, depth, codes]
                if (q0[matches] <= 0).any() or (q1[matches] <= 0).any():
                    raise ValueError("Zero observed-code probability at an active gold prefix")
                update[matches] += np.log(q1[matches]) - np.log(q0[matches])
            scores = base + update
            p = _softmax(tscore)
            q = _softmax(scores)
            item_kl += float(np.sum(p * (np.log(np.maximum(p, 1e-30)) -
                                           np.log(np.maximum(q, 1e-30)))))
            for k in range(min(topk, c)):
                for j in range(topk, min(outsider, c)):
                    violations += int(scores[k] <= scores[j])
                    pairs += 1
        if not pairs:
            raise ValueError("No boundary pairs for Figure 1 analysis")
        vals.append({"alpha": alpha, "token_kl": token_kl,
                     "candidate_kl": item_kl / n,
                     "boundary_disagreement_pct": 100. * violations / pairs,
                     "requests": n, "boundary_pairs": pairs})
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("aligned_npz", type=Path)
    ap.add_argument("--alphas", default="0,0.25,0.5",
                    help="Comma-separated coefficients in [0,1]")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--outsider", type=int, default=40)
    ap.add_argument("--output-json", type=Path)
    args = ap.parse_args()
    alphas = [float(v.strip()) for v in args.alphas.split(",")]
    with np.load(args.aligned_npz, allow_pickle=False) as d:
        result = evaluate(d, alphas, args.topk, args.outsider)
    output = json.dumps({"results": result}, indent=2) + "\n"
    print(output)
    if args.output_json:
        args.output_json.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
