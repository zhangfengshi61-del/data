"""PRQ-SID: Predictive Residual Quantization for Generative Recommendation.

Standalone research implementation that lives inside the RecBoard-master
comparison contract (Amazon-2014 5-core, chronological leave-two-out,
max_history=20, Recall/NDCG/MRR@5/10/20).

Pipeline
--------
1. Train a collaborative sequential teacher on the canonical training prefixes.
2. Sample history anchors and compute each item's conditional prediction
   profile pi_i(m) = P_T(I=i | H=h_m), normalised over anchors.
3. Predictive residual quantization (log-space residual k-means) with an
   optional semantic-coherence constraint, producing a fixed conflict-free SID.
4. Diagnostics: prefix predictive distortion D_k and ideal beam regret R_{B,r}.
5. Hard SID correction by full-SID swaps.
6. Export a ``sem_ids`` JSON consumable by the shared genrec PSID/T5 pipeline.

Nothing here changes the downstream TIGER/T5 architecture, the data split or
the item mapping.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/fszhang/RecBoard-master")
DATA_DIR = ROOT / "data" / "processed"
SENT_EMB_ROOT = ROOT / "Latte" / "cache" / "Amazon2014"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_items(category: str) -> List[int]:
    """Sorted item ids, matching genrec's ``id2item`` ordering (1-based)."""
    proc = DATA_DIR / category
    items: List[int] = []
    with open(proc / "item_text.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                items.append(int(json.loads(line)["item"]))
    items = sorted(items)
    return items


def load_sequences(category: str) -> List[List[int]]:
    """Chronological per-user full item sequences."""
    proc = DATA_DIR / category
    seqs: List[List[int]] = []
    with open(proc / "user_item_sequences.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            seqs.append([int(x) for x in row["items"]])
    return seqs


def load_sent_emb(category: str, dim: int = 768) -> np.ndarray:
    """Sentence embeddings in sorted-item order (row r -> item index r)."""
    path = SENT_EMB_ROOT / category / "processed" / "sentence-t5-base.sent_emb"
    emb = np.fromfile(path, dtype=np.float32).reshape(-1, dim)
    return emb


def item_id_to_index(items: Sequence[int]) -> Dict[int, int]:
    return {it: idx for idx, it in enumerate(items)}


def build_transitions(
    seqs: Sequence[Sequence[int]],
    max_len: int = 20,
) -> Tuple[List[List[int]], List[int]]:
    """Training prefixes/targets under leave-two-out.

    For a full sequence ``s`` the trainable next-item targets are
    ``s[1], ..., s[L-3]`` (indices ``L-2``/``L-1`` are val/test).
    """
    inputs: List[List[int]] = []
    targets: List[int] = []
    for seq in seqs:
        L = len(seq)
        for k in range(1, L - 2):
            hist = list(seq[max(0, k - max_len):k])
            inputs.append(hist)
            targets.append(seq[k])
    return inputs, targets


# ---------------------------------------------------------------------------
# Collaborative sequential teacher (SASRec-style)
# ---------------------------------------------------------------------------
class CausalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


class SASRecTeacher(nn.Module):
    """Next-item teacher over the full item catalogue."""

    def __init__(self, n_items: int, max_len: int = 20, d_model: int = 64,
                 n_layers: int = 2, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [CausalBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self._causal = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        if self._causal is None or self._causal.shape[0] != length \
                or self._causal.device != device:
            mask = torch.full((length, length), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._causal = mask
        return self._causal

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` (B, L) padded with 0 (left padding). Returns hidden (B, d)."""
        b, L = seq.shape
        pad_mask = seq == 0
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(b, L)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        attn_mask = self._causal_mask(L, seq.device)
        for block in self.blocks:
            x = block(x, pad_mask, attn_mask)
        x = self.ln_final(x)
        return x[:, -1, :]

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item_emb.weight.t()


def pad_histories(histories: Sequence[Sequence[int]], max_len: int,
                  device: torch.device) -> torch.Tensor:
    out = torch.zeros(len(histories), max_len, dtype=torch.long)
    for r, hist in enumerate(histories):
        h = list(hist)[-max_len:]
        out[r, -len(h):] = torch.tensor(h, dtype=torch.long)
    return out.to(device)


def train_teacher(
    category: str,
    out_dir: Path,
    max_len: int = 20,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 60,
    seed: int = 2025,
    device: str = "cuda:0",
    log_every: int = 10,
) -> SASRecTeacher:
    """Train the sequential teacher on the canonical train prefixes."""
    seed_all(seed)
    dev = torch.device(device)
    items = load_items(category)
    idx = item_id_to_index(items)
    seqs = load_sequences(category)
    inputs, targets = build_transitions(seqs, max_len=max_len)
    n_items = len(items)
    print(f"[teacher] {category}: {n_items} items, {len(inputs)} train transitions")

    inputs_idx = [[idx[i] + 1 for i in h] for h in inputs]
    targets_idx = torch.tensor([idx[t] + 1 for t in targets], dtype=torch.long)

    model = SASRecTeacher(n_items, max_len, d_model, n_layers, n_heads,
                          dropout).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(inputs_idx)
    steps_per_epoch = max(1, n // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch, eta_min=lr * 0.05
    )
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        total = 0.0
        nb = 0
        for start in range(0, n - batch_size + 1, batch_size):
            sel = order[start:start + batch_size]
            hist = [inputs_idx[i] for i in sel]
            seq = pad_histories(hist, max_len, dev)
            y = targets_idx[sel].to(dev)
            h = model.encode(seq)
            logits = model.logits(h)[:, 1:]
            loss = F.cross_entropy(logits, y - 1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"[teacher] epoch {epoch:3d} loss {total / max(nb, 1):.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "config": {"n_items": n_items, "max_len": max_len,
                           "d_model": d_model, "n_layers": n_layers,
                           "n_heads": n_heads, "dropout": dropout}},
               out_dir / "teacher.pt")
    return model


@torch.no_grad()
def compute_profiles(
    category: str,
    teacher: SASRecTeacher,
    out_dir: Path,
    n_anchors: int = 512,
    max_len: int = 20,
    seed: int = 2025,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample anchors and return the teacher probability matrix.

    Returns ``P`` of shape (n_anchors, n_items): ``P[m, i] = p_T(i | h_m)``.
    """
    dev = torch.device(device)
    teacher.eval().to(dev)
    items = load_items(category)
    idx = item_id_to_index(items)
    seqs = load_sequences(category)
    inputs, _ = build_transitions(seqs, max_len=max_len)
    rng = np.random.default_rng(seed)
    n = len(inputs)
    sel = rng.choice(n, size=min(n_anchors, n), replace=False)
    anchors = [[idx[i] + 1 for i in inputs[j]] for j in sel]
    probs = np.zeros((len(anchors), len(items)), dtype=np.float32)
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start:start + batch_size]
        seq = pad_histories(chunk, max_len, dev)
        h = teacher.encode(seq)
        logits = teacher.logits(h)[:, 1:]
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        probs[start:start + len(chunk)] = p
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "profiles.npz", P=probs,
                        anchor_rows=sel.astype(np.int64))
    return probs, sel


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def _group_prefix_codes(sid: np.ndarray, level: int) -> np.ndarray:
    """Integer prefix code for levels ``0..level`` (inclusive)."""
    code = np.zeros(sid.shape[0], dtype=np.int64)
    for l in range(level + 1):
        code = code * 256 + sid[:, l]
    return code


def prefix_distortion(P: np.ndarray, sid: np.ndarray, level: int,
                      eps: float = 1e-12) -> float:
    """``D_k = sum_i rho_i KL(pi_i || pi_{C_k(i)})`` for ``k = level + 1``.

    ``P`` is (M, N) teacher probabilities over anchors; profiles are formed
    with uniform anchor weights.
    """
    rho = P.mean(axis=0)                       # (N,)
    pi = (P / np.maximum(P.sum(axis=0, keepdims=True), eps)).T  # (N, M)
    pi = pi / np.maximum(pi.sum(axis=1, keepdims=True), eps)
    code = _group_prefix_codes(sid, level)
    n_codes = code.max() + 1
    wsum = np.bincount(code, weights=rho, minlength=n_codes)          # (C,)
    psum = np.zeros((n_codes, P.shape[0]), dtype=np.float64)
    np.add.at(psum, code, rho[:, None] * pi)                          # (C, M)
    pi_c = psum / np.maximum(wsum[:, None], eps)
    log_pi = np.log(np.maximum(pi, eps))
    log_pi_c = np.log(np.maximum(pi_c, eps))
    kl = (pi * (log_pi - log_pi_c[code])).sum(axis=1)                 # (N,)
    return float((rho * kl).sum())


def beam_regret(P: np.ndarray, sid: np.ndarray, beam: int = 20,
                top_r: int = 1, n_anchors: Optional[int] = None,
                seed: int = 2025) -> float:
    """Ideal-probability structural beam regret ``R_{B,r}``.

    For each anchor history the exact prefix probability masses are computed
    and a global top-B beam is simulated level by level. The metric is the
    teacher mass of the ideal top-``r`` items that the beam can no longer
    reach.
    """
    rng = np.random.default_rng(seed)
    M, N = P.shape
    if n_anchors is not None and n_anchors < M:
        rows = rng.choice(M, size=n_anchors, replace=False)
        P = P[rows]
        M = P.shape[0]
    L = sid.shape[1]
    codes = []
    c = np.zeros(N, dtype=np.int64)
    for l in range(L):
        c = c * 256 + sid[:, l]
        codes.append(c)
    total = 0.0
    for m in range(M):
        p = P[m]
        masses = np.bincount(sid[:, 0], weights=p, minlength=256)
        keep = np.argsort(-masses)[:beam]
        for l in range(1, L):
            parent_ok = np.isin(codes[l - 1], keep)
            if not parent_ok.any():
                keep = np.array([], dtype=np.int64)
                break
            child = codes[l][parent_ok]
            pw = p[parent_ok]
            uniq, inv = np.unique(child, return_inverse=True)
            mass = np.bincount(inv, weights=pw)
            order = np.argsort(-mass)[:beam]
            keep = uniq[order]
        reachable = np.isin(codes[-1], keep) if len(keep) else np.zeros(N, dtype=bool)
        ideal = np.argsort(-p)[:top_r]
        total += float(p[ideal][~reachable[ideal]].sum())
    return total / max(M, 1)


# ---------------------------------------------------------------------------
# Predictive residual quantization
# ---------------------------------------------------------------------------
def _kmeans(features: np.ndarray, k: int, seed: int, niter: int = 30):
    try:
        import faiss  # type: ignore
        d = features.shape[1]
        km = faiss.Kmeans(d, k, niter=niter, verbose=False, seed=seed)
        km.train(features.astype(np.float32))
        centroids = km.centroids
        index = faiss.IndexFlatL2(d)
        index.add(centroids)
        _, assign = index.search(features.astype(np.float32), 1)
        return centroids.astype(np.float64), assign[:, 0].astype(np.int64)
    except Exception:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, random_state=seed,
                             batch_size=4096, n_init=10, max_iter=300)
        assign = km.fit_predict(features)
        return km.cluster_centers_.astype(np.float64), assign.astype(np.int64)


def _resolve_sid_conflicts(base: np.ndarray, sid: np.ndarray,
                           codebooks: np.ndarray, seed: int = 2025) -> np.ndarray:
    """Reassign colliding SIDs to the nearest unused code (PSID-style).

    ``base`` is the (N, M) representation being reconstructed by the sum of
    codebooks. Collisions are rare for this quantizer but must be removed
    because the downstream tokenizer assumes conflict-free semantic IDs.
    """
    rng = np.random.default_rng(seed)
    N, L = sid.shape
    resolved = sid.copy()
    used: Dict[Tuple[int, ...], List[int]] = {}
    for i in range(N):
        used.setdefault(tuple(int(x) for x in sid[i]), []).append(i)
    used_set = set(used.keys())

    def recon_err(i: int, code: Tuple[int, ...]) -> float:
        acc = np.zeros(base.shape[1], dtype=np.float64)
        for l in range(L):
            acc += codebooks[l][code[l]]
        diff = base[i] - acc
        return float(diff @ diff)

    n_resolved = 0
    for code, items in used.items():
        if len(items) == 1:
            continue
        for i in items[1:]:
            best = None
            best_err = float("inf")
            for l in range(L):
                for v in range(codebooks.shape[1]):
                    cand = list(code)
                    cand[l] = int(v)
                    cand = tuple(cand)
                    if cand in used_set:
                        continue
                    err = recon_err(i, cand)
                    if err < best_err:
                        best_err = err
                        best = cand
            if best is None:
                for _ in range(2000):
                    cand = tuple(int(rng.integers(codebooks.shape[1]))
                                 for _ in range(L))
                    if cand in used_set:
                        continue
                    err = recon_err(i, cand)
                    if err < best_err:
                        best_err = err
                        best = cand
            if best is None:
                raise RuntimeError("could not resolve SID conflict")
            resolved[i] = list(best)
            used_set.add(best)
            n_resolved += 1
    if n_resolved:
        print(f"[quantize] resolved {n_resolved} SID conflicts")
    return resolved


def predictive_residual_quantization(
    P: np.ndarray,
    sent_emb: np.ndarray,
    n_levels: int = 3,
    n_codes: int = 256,
    sem_weight: float = 0.0,
    niter: int = 30,
    seed: int = 2025,
    mode: str = "pred_rq",
) -> Dict[str, np.ndarray]:
    """Quantize conditional prediction profiles.

    ``mode='pred_rq'``: log-space residual quantization. The codeword at each
    level is a log-probability correction; residuals are
    ``log pi_i - log q_{C_{k-1}(i)}``.

    ``mode='plain_rq'``: ordinary Euclidean residual quantization on the raw
    profile vectors (the key ablation that isolates the predictive-residual
    mechanism from the profile representation itself).
    """
    eps = 1e-12
    n_anchors, n_items = P.shape
    rho = P.mean(axis=0)
    pi = (P / np.maximum(P.sum(axis=0, keepdims=True), eps)).T
    pi = pi / np.maximum(pi.sum(axis=1, keepdims=True), eps)
    base = np.log(pi + eps) if mode == "pred_rq" else pi

    X = sent_emb.astype(np.float64)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)

    residual = base.copy()
    sid = np.zeros((n_items, n_levels), dtype=np.int64)
    codebooks: List[np.ndarray] = []
    for l in range(n_levels):
        if sem_weight > 0:
            feats = np.concatenate(
                [residual, sem_weight * X], axis=1
            ).astype(np.float32)
        else:
            feats = residual.astype(np.float32)
        centroids, assign = _kmeans(feats, n_codes, seed + l, niter=niter)
        codebooks.append(centroids[:, :n_anchors])
        sid[:, l] = assign
        # log q after this level
        log_q = np.zeros_like(base)
        for j in range(l + 1):
            log_q += codebooks[j][sid[:, j]]
        residual = base - log_q
    cb = np.stack(codebooks)
    sid = _resolve_sid_conflicts(base, sid, cb, seed=seed)
    return {"sid": sid, "codebooks": cb, "rho": rho, "pi": pi}


def sid_only_semantic_reconstruction(
    sid: np.ndarray, sent_emb: np.ndarray, n_codes: int = 256,
    code_dim: int = 128, epochs: int = 300, lr: float = 1e-3,
    seed: int = 2025, device: str = "cuda:0",
) -> Dict[str, float]:
    """Train a decoder that reads *only the SID* and reconstructs semantics."""
    seed_all(seed)
    dev = torch.device(device)
    X = sent_emb.astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Xt = torch.tensor(X, device=dev)
    L = sid.shape[1]
    codes = [torch.tensor(sid[:, l], dtype=torch.long, device=dev) for l in range(L)]
    emb = nn.ModuleList([nn.Embedding(n_codes, code_dim) for _ in range(L)]).to(dev)
    dec = nn.Sequential(nn.Linear(code_dim, 512), nn.GELU(),
                        nn.Linear(512, X.shape[1])).to(dev)
    params = list(emb.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    for _ in range(epochs):
        h = sum(emb[l](codes[l]) for l in range(L))
        x_hat = dec(h)
        x_hat = x_hat / x_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        loss = F.mse_loss(x_hat, Xt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        h = sum(emb[l](codes[l]) for l in range(L))
        x_hat = dec(h)
        x_hat = x_hat / x_hat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        cos = (x_hat * Xt).sum(dim=-1).mean().item()
        mse = F.mse_loss(x_hat, Xt).item()
    return {"sem_cosine": float(cos), "sem_mse": float(mse)}


# ---------------------------------------------------------------------------
# Hard SID correction
# ---------------------------------------------------------------------------
def hard_sid_correction(
    P: np.ndarray,
    sid: np.ndarray,
    alpha: Optional[Sequence[float]] = None,
    lambda_r: float = 1.0,
    beam: int = 20,
    top_r: int = 1,
    n_anchors_diag: int = 128,
    max_swaps: int = 300,
    max_evals: int = 4000,
    pool_size: int = 2000,
    seed: int = 2025,
    verbose: bool = True,
) -> np.ndarray:
    """Greedy full-SID swaps that reduce ``sum_k alpha_k D_k + lambda_R R``.

    Swapping two complete SIDs preserves SID length, collision-freeness and
    every prefix bucket size, so any gain is attributable to *who shares a
    prefix* rather than to capacity changes.
    """
    rng = np.random.default_rng(seed)
    M, N = P.shape
    L = sid.shape[1]
    if alpha is None:
        alpha = [1.0 / L] * L
    rows = rng.choice(M, size=min(n_anchors_diag, M), replace=False)
    P_diag = P[rows]

    def objective(sid_cur: np.ndarray) -> float:
        val = 0.0
        for k in range(L):
            val += alpha[k] * prefix_distortion(P_diag, sid_cur, k)
        val += lambda_r * beam_regret(P_diag, sid_cur, beam=beam,
                                      top_r=top_r, n_anchors=None)
        return val

    rho = P.mean(axis=0)
    pool = np.argsort(-rho)[:min(pool_size, N)]
    cur = sid.copy()
    cur_obj = objective(cur)
    best = cur.copy()
    best_obj = cur_obj
    if verbose:
        print(f"[hard] start objective {cur_obj:.6f} (pool {len(pool)})", flush=True)
    evals = 0
    accepted = 0
    while accepted < max_swaps and evals < max_evals:
        i = int(pool[rng.integers(len(pool))])
        j = int(pool[rng.integers(len(pool))])
        if i == j:
            continue
        trial = cur.copy()
        trial[i], trial[j] = cur[j].copy(), cur[i].copy()
        evals += 1
        obj = objective(trial)
        if obj < cur_obj - 1e-9:
            cur = trial
            cur_obj = obj
            accepted += 1
            if obj < best_obj:
                best_obj = obj
                best = trial.copy()
            if verbose and accepted % 25 == 0:
                print(f"[hard] swap {accepted} obj {cur_obj:.6f}", flush=True)
    if verbose:
        print(f"[hard] done: {accepted} accepted / {evals} evals, "
              f"obj {best_obj:.6f}", flush=True)
    return best


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_sem_ids(sid: np.ndarray, items: Sequence[int], path: Path) -> None:
    mapping = {str(items[i]): [int(x) for x in sid[i].tolist()]
               for i in range(len(items))}
    assert len(set(tuple(v) for v in mapping.values())) == len(mapping), \
        "SIDs must be conflict-free"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(mapping, handle)
