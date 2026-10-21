#!/usr/bin/env python3
"""Recompute the nine paper diagnostics from a fixed teacher top-40 support.

The .npz must contain aligned arrays (all request and candidate axes identical):
  sid: (N,M,4) integer SID digits
  teacher_scores, student_scores: (N,M) full-item and SID-path scores
  teacher_steps, student_steps, base_steps: (N,M,4) SID log increments
  nvalid: (N,) valid teacher candidate counts
  step_scale: (3,) or (4,) fixed TRAIN-derived teacher increment scales.

base_steps MUST come from the matching frozen Base-KD reference, not from
each evaluated model. The script does not infer scores, seeds or candidates.
It deliberately reports no evaluation numbers without an actual .npz.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    z = np.exp(x - np.max(x))
    return z / z.sum()


def _kl(p, q):
    p, q = np.asarray(p), np.asarray(q)
    active = p > 0
    return float(np.sum(p[active] * (np.log(p[active]) - np.log(np.maximum(q[active], 1e-30)))))


def _groups(codes, probs, depth):
    acc = {}
    for sid, p in zip(codes, probs):
        key = tuple(int(v) for v in sid[:depth])
        acc[key] = acc.get(key, 0.) + float(p)
    keys = sorted(acc)
    mass = np.array([acc[k] for k in keys], dtype=np.float64)
    return keys, mass


def _huber(z):
    z = np.abs(np.asarray(z))
    return np.where(z <= 1, 0.5 * z * z, z - 0.5)


def compute(d, topk=10, outsider=40, margin=.05):
    required = ("sid", "teacher_scores", "student_scores",
                "teacher_steps", "student_steps", "base_steps",
                "nvalid", "step_scale")
    for key in required:
        if key not in d:
            raise ValueError("Missing aligned input " + key)
    sid = np.asarray(d["sid"])
    ts = np.asarray(d["teacher_scores"])
    ss = np.asarray(d["student_scores"])
    tstep = np.asarray(d["teacher_steps"])
    sstep = np.asarray(d["student_steps"])
    bstep = np.asarray(d["base_steps"])
    nvalid = np.asarray(d["nvalid"])
    scale = np.asarray(d["step_scale"], dtype=np.float64)
    if sid.ndim != 3 or sid.shape[2] != 4:
        raise ValueError("sid must be (N,M,4)")
    n, m, _ = sid.shape
    if n < 1 or m < 1 or ts.shape != (n, m) or ss.shape != (n, m):
        raise ValueError("Full scores do not match aligned SID support")
    if (tstep.shape != sid.shape or sstep.shape != sid.shape or
            bstep.shape != sid.shape or nvalid.shape != (n,) or
            scale.shape not in ((3,), (4,))):
        raise ValueError("Step arrays/counts/scale do not match aligned SID support")
    if not 0 < topk < outsider <= m or np.any(nvalid < 1) or np.any(nvalid > m):
        raise ValueError("Invalid K/B or per-request candidate count")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Invalid training-derived fixed step scale")
    if not np.all(np.isfinite(np.concatenate((ts[:, :1], ss[:, :1])))):
        raise ValueError("Nonfinite request scores")
    sigma = np.maximum(scale[:3], 1e-4)
    totals = dict(item_kl=0., hdrift_1=0., hdrift_2=0.,
                  tgd_1=0., tgd_2=0.)
    pairs = dict(pov=0., pair_softplus=0., huber_d=0., huber_i=0.)
    pair_count = 0
    for idx in range(n):
        count = int(nvalid[idx])
        st, sc = ts[idx, :count], ss[idx, :count]
        if not (np.isfinite(st).all() and np.isfinite(sc).all() and
                np.isfinite(tstep[idx, :count]).all() and
                np.isfinite(sstep[idx, :count]).all() and
                np.isfinite(bstep[idx, :count]).all()):
            raise ValueError("Nonfinite active scores at request %d" % idx)
        # Candidate-axis identity is essential. Reordering just one array
        # silently corrupts teacher–student comparisons; refuse unsorted rows.
        if np.any(np.diff(st) > 1e-7):
            raise ValueError("Teacher candidates must be sorted by descending cached score: request %d" % idx)
        if len(set(map(tuple, sid[idx, :count].tolist()))) != count:
            raise ValueError("Duplicate valid SID in request %d" % idx)
        pt, ps = _softmax(st), _softmax(sc)
        totals["item_kl"] += _kl(pt, ps)
        for depth in (1, 2):
            kt, gt = _groups(sid[idx, :count], pt, depth)
            ks, gs = _groups(sid[idx, :count], ps, depth)
            if kt != ks:
                raise ValueError("Prefix grouping differs despite same SIDs")
            totals["hdrift_%d" % depth] += _kl(gt, gs)
            # Keys are sorted lexicographically, so argmax breaks ties by SID.
            totals["tgd_%d" % depth] += int(kt[int(np.argmax(gt))] !=
                                             kt[int(np.argmax(gs))])
        # All eligible teacher-ordered cross-cutoff pairs.
        for i in range(min(topk, count)):
            for j in range(topk, min(outsider, count)):
                diff = np.flatnonzero(sid[idx, i] != sid[idx, j])
                if diff.size == 0:
                    continue
                first = int(diff[0])
                g_s = sstep[idx, i] - sstep[idx, j]
                g_t = tstep[idx, i] - tstep[idx, j]
                g_b = bstep[idx, i] - bstep[idx, j]
                gap_s = float(g_s.sum())
                gap_b = float(g_b.sum())
                pairs["pov"] += float(gap_s <= 0)
                pairs["pair_softplus"] += float(np.logaddexp(0., margin - gap_s))
                eligible = np.zeros(3, dtype=np.float64)
                eligible[first:3] = 1.
                r_base = (g_t[:3] - g_b[:3]) * eligible
                z = (g_s[:3] - g_t[:3]) / sigma
                hz = _huber(z)
                for mode in ("d", "i"):
                    if mode == "d":
                        before = np.logaddexp(0., margin - gap_b)
                        after = np.logaddexp(0., margin - (gap_b + r_base))
                    else:
                        before = np.logaddexp(0., -gap_b / margin)
                        after = np.logaddexp(0., -(gap_b + r_base) / margin)
                    improvement = np.maximum(0., before - after) * eligible
                    weights = improvement / max(float(improvement.sum()), 1e-9)
                    pairs["huber_" + mode] += float(np.dot(weights, hz))
                pair_count += 1
    if pair_count == 0:
        raise ValueError("No valid teacher boundary pairs; cannot report pair diagnostics")
    result = {k: v / n for k, v in totals.items()}
    result.update({k: v / pair_count for k, v in pairs.items()})
    for k in ("tgd_1", "tgd_2", "pov"):
        result[k] *= 100.
    result["requests"] = int(n)
    result["pairs"] = int(pair_count)
    result["support_m"] = int(m)
    result["cutoff_k"] = int(topk)
    result["outsider_b"] = int(outsider)
    result["notes"] = "TGD and POV are percentages; all other metrics unscaled"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned_npz", type=Path)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--outsider", type=int, default=40)
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    with np.load(args.aligned_npz, allow_pickle=False) as d:
        result = compute(d, topk=args.topk, outsider=args.outsider, margin=args.margin)
    output = json.dumps(result, indent=2) + "\n"
    print(output)
    if args.output_json:
        args.output_json.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
