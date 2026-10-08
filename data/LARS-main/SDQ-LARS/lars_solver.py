"""Batched, budget-truncated least-angle regression (LAR, not Lasso).

The dictionary rows are unit directions. This follows the equiangular LAR
path, including joint updates to ALL active coefficients, with no Lasso
coefficient-drop events. Sparse inference is detached for alternating
dictionary learning. See Efron et al. (2004), section 2.
"""
import torch


@torch.no_grad()
def lar_code(target, directions, steps=3, ridge=1e-7):
    if target.ndim != 2 or directions.ndim != 2:
        raise ValueError("target and directions must be matrices")
    if target.shape[1] != directions.shape[1]:
        raise ValueError("feature dimensions differ")
    if not 1 <= steps <= min(directions.shape):
        raise ValueError("steps must be in 1..min(dictionary size, dimension)")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative")
    # Keep sensitive active-set solves out of mixed precision.
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    y, d = target.to(dtype), directions.to(dtype)
    n, k = y.shape[0], d.shape[0]
    coef = y.new_zeros(n, k)
    residual = y.clone()
    active = torch.zeros(n, k, dtype=torch.bool, device=y.device)
    rows = torch.arange(n, device=y.device)
    indices, signs = [], []
    eps = torch.finfo(dtype).eps * 8
    for _ in range(steps):
        corr = residual @ d.T
        idx = corr.abs().masked_fill(active, -1).argmax(dim=1)
        sign = torch.where(corr[rows, idx] >= 0, 1.0, -1.0)
        indices.append(idx)
        signs.append(sign)
        active[rows, idx] = True
        chosen = torch.stack(indices, dim=1)
        s = torch.stack(signs, dim=1).to(dtype)
        atoms = d[chosen] * s.unsqueeze(-1)
        gram = atoms @ atoms.transpose(1, 2)
        eye = torch.eye(len(indices), dtype=dtype, device=y.device)
        ones = y.new_ones(n, len(indices), 1)
        system = gram + ridge * eye
        try:
            w = torch.linalg.solve(system, ones).squeeze(-1)
        except RuntimeError as exc:
            # Large catalogs can create nearly dependent active directions.
            # Keep the normal path unchanged; only singular systems use a
            # numerically safe jittered solve, then least-squares fallback.
            if "singular" not in str(exc).lower():
                raise
            jitter = max(float(ridge), 1e-5)
            solved = False
            for _ in range(4):
                try:
                    w = torch.linalg.solve(system + jitter * eye, ones).squeeze(-1)
                    solved = True
                    break
                except RuntimeError:
                    jitter *= 10.0
            if not solved:
                w = torch.linalg.lstsq(system + jitter * eye, ones).solution.squeeze(-1)
        angle = w.sum(dim=1).clamp_min(eps).rsqrt()
        w = w * angle.unsqueeze(1)
        u = (atoms * w.unsqueeze(-1)).sum(dim=1)
        a = u @ d.T
        cmax = (corr.gather(1, chosen) * s).mean(dim=1).clamp_min(0)
        # Stop at either the next knot or the active-set least-squares end.
        gamma_end = cmax / angle.clamp_min(eps)
        den1, den2 = angle[:, None] - a, angle[:, None] + a
        g1 = (cmax[:, None] - corr) / den1.clamp_min(eps)
        g2 = (cmax[:, None] + corr) / den2.clamp_min(eps)
        valid1 = (~active) & (den1 > eps) & (g1 >= -eps)
        valid2 = (~active) & (den2 > eps) & (g2 >= -eps)
        g1 = g1.clamp_min(0).masked_fill(~valid1, float("inf"))
        g2 = g2.clamp_min(0).masked_fill(~valid2, float("inf"))
        gamma = torch.minimum(gamma_end, torch.minimum(g1.amin(1), g2.amin(1)))
        coef.scatter_add_(1, chosen, gamma[:, None] * w * s)
        residual = y - coef @ d
    return coef.to(target.dtype)
