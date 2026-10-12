"""Unified distillation baselines for the ServeSID A/B study (Amazon2014).

Implements the baseline losses on the SAME chosen SID student architecture
(MLP/GRU/CNN within each comparison) and TIGER teacher / packed cache / beam dump:

  - RD         (Ranking Distillation, Tang & Wang KDD'18)   [ported from RCE-KD repo baseline.py]
  - RRD        (Relaxed Ranking Distillation, CIKM'20)      [ported from RCE-KD repo baseline.py]
  - RCE-KD     (Rejuvenated CE KD, ICLR'26)                 [ported from RCE-KD repo playground.py]
  - LOHRec     (Order + Hierarchy ranking losses, EMNLP'25) [ported from LOHRec repo loss.py]
  - GKD        (On-policy distillation, ICLR'24)            [JSD per digit, student-sampled prefixes]
  - DistiLLM-2 (skew-KL + contrastive, ICML'25)

Every method keeps the standard digit distillation as its base ("combined" loss,
alpha=0.7), mirroring how ServeSID adds its serving-list losses on top.
Checkpoint selection uses complete validation beam NDCG@10 every 5 epochs
and on the final epoch. Test is never a selection fallback.
Each checkpoint stores the architecture and its hyperparameters, and
can be loaded by sid_mlp.inference.build_sidmlp_from_checkpoint.

Usage:
  python -m sid_mlp.kd_baselines --method rd \
      --checkpoint <teacher.pth> --sem_ids <sem_ids.json> \
      --dataset Amazon2014 --category Beauty \
      --graph_dir <GRAPH_DIR> --dataset_tag Amazon2014_Beauty \
      --val_dataset_tag Amazon2014_Beauty_val \
      --beam_dump <dump.pt> --ckpt_dir ... --log_dir ...
"""

import argparse
import datetime as dt
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_DIR", os.path.join(_CHECKOUT_ROOT, "..")))

from sid_mlp.beam import build_valid_tensors, gather_d4_mask
from sid_mlp.constants import CODEBOOK_SIZE, DIGIT_OFFSETS, set_seed
from sid_mlp.dataset import attach_beam_dump, collate_fn_gpu, prepare_common_data
from sid_mlp.inference import (
    build_tiger_pipeline,
    configure_precision,
    evaluate_loader,
    find_valid_set,
)
from sid_mlp.losses import (
    d4_loss_fn,
    loss_fn,
    student_list_scores,
    student_list_scores_steps,
    topk_correct,
)
from sid_mlp.models import SIDMLP, SIDGRU, SIDCNN
from sid_mlp.validation_selection import (
    require_validation_split, validate_selection_settings,
    should_beam_validate, save_if_best_val_ndcg10,
)

OFF = torch.tensor(DIGIT_OFFSETS, dtype=torch.long)

METHODS = ("rd", "rrd", "rcek", "lohrec", "gkd", "distillm2", "smartgr")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def teacher_codebook_logp(logits):
    """Normalize teacher logits on the four 256-token SID codebooks."""
    if logits.ndim != 3 or logits.shape[1] < 4:
        raise ValueError("Expected (batch, >=4 decoder steps, vocabulary) teacher logits")
    parts = []
    for d, offset in enumerate(DIGIT_OFFSETS[:4]):
        chunk = logits[:, d, offset:offset + CODEBOOK_SIZE].float()
        if chunk.shape[-1] != CODEBOOK_SIZE:
            raise ValueError(f"Missing 256-token codebook at SID position {d+1}")
        parts.append(F.log_softmax(chunk, dim=-1))
    return tuple(parts)


@torch.inference_mode()
def teacher_digit_logp(t5, enc, attn, d1, d2, d3):
    """Teacher per-digit log-softmax given teacher-forced prefixes (codebook space).

    Returns (p1, p2, p3, p4): each (B, 256) log-probs at the four digit positions.
    """
    from transformers.modeling_outputs import BaseModelOutput
    dec_in = torch.cat([
        torch.full((d1.shape[0], 1), int(t5.config.decoder_start_token_id),
                   dtype=torch.long, device=d1.device),
        (d1.long() + OFF[0]).unsqueeze(1),
        (d2.long() + OFF[1]).unsqueeze(1),
        (d3.long() + OFF[2]).unsqueeze(1),
    ], 1)
    eo = BaseModelOutput(last_hidden_state=enc)
    out = t5(encoder_outputs=eo, attention_mask=attn, decoder_input_ids=dec_in, return_dict=True)
    # Use the same 256-way conditional space as cached gold-prefix
    # logits; full-vocabulary probabilities sliced afterwards do not
    # sum to one and would invalidate on-policy JSD/skew-KL.
    return teacher_codebook_logp(out.logits)


def sample_valid_quads(valid_quads, batch, k, device):
    """Sample k valid (item) quads per row, uniform over the full catalog. (B,k,4)."""
    idx = torch.randint(0, valid_quads.shape[0], (batch, k), device=device)
    return valid_quads[idx]


def student_quad_score(model, enc_seq, enc_mask, quads_cb, list_masks):
    """Student total score (sum of per-digit logp) for candidate quads. (B,M)."""
    return student_list_scores(model, enc_seq, enc_mask, quads_cb, list_masks)


def pairwise_margin_ranking(score, rank, margin=0.001, scale=1.0, adaptive=False, valid=None):
    """LOHRec-style pairwise ranking over a teacher-ranked list.

    score: (B, n) higher-is-better; rank: (B, n) 1..n teacher ranks (smaller = better).
    Pairs (i<j) with teacher rank_i < rank_j get margin = margin*dist (+log2(dist)*scale if adaptive).
    """
    B, n = score.shape
    idx = torch.triu_indices(n, n, offset=1, device=score.device)
    si = score[:, idx[0]]
    sj = score[:, idx[1]]
    ri = rank[:, idx[0]].float()
    rj = rank[:, idx[1]].float()
    pair_valid = torch.ones_like(si, dtype=torch.bool)
    if valid is not None:
        pair_valid = valid[:, idx[0]] & valid[:, idx[1]]
    pair_valid = pair_valid & torch.isfinite(si) & torch.isfinite(sj)
    dist = (rj - ri).clamp_min(1.0)
    if adaptive:
        m = (margin + torch.log2(dist)) * scale
    else:
        m = margin * dist
    terms = torch.relu(m - (si - sj))
    terms = terms.masked_fill(~pair_valid, 0.0)
    return terms.sum() / pair_valid.sum().clamp_min(1)


# ----------------------------------------------------------------------------
# per-method losses
# ----------------------------------------------------------------------------

def loss_rd(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[3:15]
    bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
    B = enc.shape[0]
    W = bm_cb.shape[1]
    K = min(W, int(getattr(args, "rd_topk", 50)))
    cand = bm_cb[:, :K].long()
    valid = torch.arange(K, device=enc.device).unsqueeze(0) < bm_nv.unsqueeze(1)

    s_S = student_quad_score(model, enc, msk, cand, list_masks)

    rank = torch.arange(1, K + 1, device=enc.device).float().unsqueeze(0)
    static = torch.exp(-rank / float(getattr(args, "rd_T", 2.0)))          # (1,K)
    static = static.expand(B, -1)
    static = static.masked_fill(~valid, 0.0)
    static = static / static.sum(1, keepdim=True).clamp_min(1e-8)

    dyn_cb = sample_valid_quads(valid_quads, B, int(getattr(args, "rd_dyn", 100)), enc.device)
    s_dyn = student_quad_score(model, enc, msk, dyn_cb, list_masks).detach()   # (B,D)
    num_smaller = (s_S.unsqueeze(-1) < s_dyn.unsqueeze(1)).float().sum(-1)      # (B,K)
    rel_rank = num_smaller / max(s_dyn.shape[1], 1)
    appro_rank = torch.floor((valid_quads.shape[0] - 1) * rel_rank) + 1
    dyn_w = torch.tanh(float(getattr(args, "rd_mu", 0.01)) * (appro_rank - rank)).clamp(min=0.0)
    w = static * dyn_w
    w = w.masked_fill(~valid, 0.0)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
    loss = -(w * F.logsigmoid(s_S)).sum(1).mean()
    return loss, {"rd": float(loss)}


def loss_rrd(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
    B = enc.shape[0]
    K = min(bm_cb.shape[1], int(getattr(args, "rrd_K", 10)))
    L = int(getattr(args, "rrd_L", 40))
    cand = bm_cb[:, :K].long()
    valid = torch.arange(K, device=enc.device).unsqueeze(0) < bm_nv.unsqueeze(1)

    s1 = student_quad_score(model, enc, msk, cand, list_masks)          # (B,K) teacher top-K
    un = sample_valid_quads(valid_quads, B, L, enc.device)
    s2 = student_quad_score(model, enc, msk, un, list_masks)            # (B,L)

    s1 = torch.clamp(s1, min=-80.0).masked_fill(~valid, float("-inf"))
    s2 = torch.clamp(s2, min=-80.0)
    above = s1.masked_fill(~valid, 0.0).sum(1, keepdim=True)
    below1 = s1.masked_fill(~valid, float("-inf")).flip(-1).exp().cumsum(1)
    below2 = s2.exp().sum(1, keepdim=True)
    below = (below1 + below2).log().sum(1, keepdim=True)
    loss = -(above - below).sum() / B
    return loss, {"rrd": float(loss)}


def loss_rcek(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
    B = enc.shape[0]
    W = bm_cb.shape[1]
    K = min(W, int(getattr(args, "mkd_K", 50)))
    L = min(W, int(getattr(args, "mkd_L", 50)))
    tau = float(getattr(args, "mkd_tau", 1.0))
    T = float(getattr(args, "mkd_T", 10.0))
    beta = float(getattr(args, "mkd_beta", 5.0))

    pool = bm_cb[:, :K].long()
    valid = torch.arange(K, device=enc.device).unsqueeze(0) < bm_nv.unsqueeze(1)
    s_pool = student_quad_score(model, enc, msk, pool, list_masks)      # (B,K)
    t_pool = bm_sc[:, :K].float()                                        # teacher exact scores

    # student's top-k within the pool
    s_rank = s_pool.masked_fill(~valid, float("-inf")).argsort(dim=1, descending=True)
    itemS = torch.gather(pool, 1, s_rank.unsqueeze(-1).expand(-1, -1, 4))[:, :K, :]
    logit_S_itemS = torch.gather(s_pool, 1, s_rank)[:, :K] / tau
    t_itemS = torch.gather(t_pool, 1, s_rank)[:, :K]
    prob_T_itemS = torch.softmax(t_itemS / tau, dim=1)
    loss_itemS = F.cross_entropy(logit_S_itemS, prob_T_itemS, reduction="none")

    # interesting items: rank-weighted sampling from the pool
    with torch.no_grad():
        w = torch.exp(-(torch.arange(K, device=enc.device).float() + 1) / T).expand(B, -1)
        w = w.masked_fill(~valid, 0.0)
        samp_idx = torch.multinomial(w, L, replacement=False)
        interesting = torch.gather(pool, 1, samp_idx.unsqueeze(-1).expand(-1, -1, 4))
        s_interesting = torch.gather(s_pool, 1, samp_idx)
        t_interesting = torch.gather(t_pool, 1, samp_idx)

    logit_T_interesting = t_interesting / tau
    logit_T_itemT = t_pool / tau
    exp_T_interesting = torch.exp(logit_T_interesting)
    exp_T_itemT = torch.exp(logit_T_itemT)
    mask = (interesting.unsqueeze(2) == pool.unsqueeze(1)).all(-1)       # (B,L,K)
    mask_k = mask.any(1)                                                  # (B,K): pool item in interesting
    mask2_k = (pool.unsqueeze(2) == itemS.unsqueeze(1)).all(-1).any(2)    # (B,K): pool item in itemS
    exp_T_itemT = exp_T_itemT.masked_fill(mask_k, 0.0)
    Z_T = exp_T_interesting.sum(-1, keepdim=True) + exp_T_itemT.sum(-1, keepdim=True)
    prob_T_all = torch.cat([exp_T_interesting, exp_T_itemT], -1) / Z_T.clamp_min(1e-8)

    exp_S_interesting = torch.exp(s_interesting / tau)
    exp_S_itemT = torch.exp(s_pool / tau)
    exp_S_itemT = exp_S_itemT.masked_fill(mask_k | mask2_k, 0.0)
    Z_S = exp_S_interesting.sum(-1, keepdim=True) + exp_S_itemT.sum(-1, keepdim=True)
    logit_S_all = torch.cat([s_interesting / tau, s_pool / tau], -1)
    loss_itemT = -(prob_T_all * (logit_S_all - torch.log(Z_S.clamp_min(1e-8)))).sum(-1)

    overlap = mask_k.float().mean(-1)
    weight = torch.exp(-beta * overlap)
    loss = ((1 - weight) * loss_itemS + weight * loss_itemT).mean()
    return loss, {"rcek": float(loss)}


def loss_lohrec(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    d1, d2, d3, d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[3:15]
    bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
    B = enc.shape[0]
    W = bm_cb.shape[1]
    margin = float(getattr(args, "rank_margin", 0.001))
    rank_weight = float(getattr(args, "rank_weight", 1.0))
    pn_weight = float(getattr(args, "pos_neg_weight", 1.0))
    adaptive = bool(getattr(args, "adaptive", True))

    pool = bm_cb.long()                                                  # teacher-ranked (B,W,4)
    valid = torch.arange(W, device=enc.device).unsqueeze(0) < bm_nv.unsqueeze(1)
    gt = torch.stack([gd1, gd2, gd3, gd4], -1)                           # (B,4)

    is_gt = (pool == gt.unsqueeze(1)).all(-1)
    cand_mask = valid & ~is_gt
    cand_scores_all = student_list_scores_steps(model, enc, msk, pool, list_masks)  # (B,W,4)
    cand_scores = cand_scores_all.sum(-1).masked_fill(~cand_mask, float("-inf"))

    # gold score (teacher-forced student score of the GT quad)
    gt_scores = student_quad_score(model, enc, msk, gt.unsqueeze(1), list_masks)[:, 0]  # (B,)

    rank = torch.arange(1, W + 1, device=enc.device).float().unsqueeze(0).expand(B, -1)
    total = 0.0
    comps = {}
    if rank_weight > 0:
        order = pairwise_margin_ranking(cand_scores, rank, margin=margin, adaptive=adaptive,
                                        valid=cand_mask)
        hier = 0.0
        for lv in range(4):
            hier = hier + pairwise_margin_ranking(
                cand_scores_all[:, :, lv].masked_fill(~cand_mask, float("-inf")),
                rank, margin=margin, adaptive=adaptive, valid=cand_mask)
        total = total + rank_weight * (order + hier)
        comps["order"] = float(order)
        comps["hier"] = float(hier)
    if pn_weight > 0:
        cand_mean = cand_scores.masked_fill(~cand_mask, float("-inf")).logsumexp(1)             - torch.log(cand_mask.sum(1).float().clamp_min(1))
        pos_neg = torch.log(1 + torch.exp(cand_mean - gt_scores)).mean()
        total = total + pn_weight * pos_neg
        comps["pos_neg"] = float(pos_neg)
    return total, comps


def _jsd(p_t_log, p_s_log):
    p_t = torch.exp(p_t_log)
    p_s = torch.exp(p_s_log)
    m = 0.5 * (p_t + p_s)
    kl_t = (p_t * (p_t_log - torch.log(m.clamp_min(1e-10)))).sum(-1)
    kl_s = (p_s * (p_s_log - torch.log(m.clamp_min(1e-10)))).sum(-1)
    return 0.5 * (kl_t + kl_s)


def loss_gkd(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[3:15]
    B = enc.shape[0]
    lam = float(getattr(args, "gkd_lambda", 0.5))

    logits1, logits2, logits3, logits4 = model(enc, msk, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    pS1 = F.log_softmax(logits1, -1)
    pS2 = F.log_softmax(logits2, -1)
    pS3 = F.log_softmax(logits3, -1)
    pS4 = F.log_softmax(logits4, -1)
    pT1 = F.log_softmax(tl1, -1)
    pT2 = F.log_softmax(tl2, -1)
    pT3 = F.log_softmax(tl3, -1)
    pT4 = F.log_softmax(tl4, -1)
    off = (_jsd(pT1, pS1) + _jsd(pT2, pS2) + _jsd(pT3, pS3) + _jsd(pT4, pS4)).mean()

    with torch.no_grad():
        q1 = torch.softmax(logits1.detach(), -1)
        s1 = torch.multinomial(q1, 1).squeeze(-1)
        e1 = model.embedding(s1 + OFF[0])
        l2 = F.log_softmax(model.head2(torch.cat([model._last_ctx, e1], 1)), -1)
        q2 = torch.softmax(l2, -1)
        s2 = torch.multinomial(q2, 1).squeeze(-1)
        e2 = model.embedding(s2 + OFF[1])
        l3 = F.log_softmax(model.head3(torch.cat([model._last_ctx, e1, e2], 1)), -1)
        q3 = torch.softmax(l3, -1)
        s3 = torch.multinomial(q3, 1).squeeze(-1)

    pT1o, pT2o, pT3o, pT4o = teacher_digit_logp(t5, enc.to(t5.dtype), (~msk).long(), s1, s2, s3)
    ctx = model._last_ctx
    e1o = model.embedding(s1 + OFF[0])
    l2o = model.head2(torch.cat([ctx, e1o], 1))
    e2o = model.embedding(s2 + OFF[1])
    l3o = model.head3(torch.cat([ctx, e1o, e2o], 1))
    e3o = model.embedding(s3 + OFF[2])
    l4o = model.head4(torch.cat([ctx, e1o, e2o, e3o], 1))
    pS1o = pS1
    pS2o = F.log_softmax(l2o, -1)
    pS3o = F.log_softmax(l3o, -1)
    pS4o = F.log_softmax(l4o, -1)
    on = (_jsd(pT1o, pS1o) + _jsd(pT2o, pS2o) + _jsd(pT3o, pS3o) + _jsd(pT4o, pS4o)).mean()

    loss = lam * on + (1 - lam) * off
    return loss, {"gkd_off": float(off), "gkd_on": float(on)}


def _skew_kl(p_t_log, p_s_log, alpha=0.9):
    """D_SKL(p_T||p_S) = KL(p_S || alpha*p_T + (1-alpha)*p_S)."""
    p_s = torch.exp(p_s_log)
    m = alpha * torch.exp(p_t_log) + (1 - alpha) * p_s
    return (p_s * (p_s_log - torch.log(m.clamp_min(1e-10)))).sum(-1)


def contrastive_student_sid_loss(s_seq, t_seq, temperature=1.0):
    """Contrast sampled SID sequences against gold with student gradients.

    Teacher scores and membership labels are detached, not student logits.
    """
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    with torch.no_grad():
        phi = (t_seq.detach() - s_seq.detach()) / 4.0
        pos_mask = phi >= phi.median(1, keepdim=True).values
        pos_mask[:, 0] = True  # gold SID is always a positive
    logits = s_seq / temperature
    pos_logits = logits.masked_fill(~pos_mask, float("-inf"))
    return (torch.logsumexp(logits, 1) - torch.logsumexp(pos_logits, 1)).mean()


def loss_distillm2(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch[3:15]
    B = enc.shape[0]
    alpha = float(getattr(args, "d2_alpha", 0.9))
    beta = float(getattr(args, "d2_beta", 0.1))
    M = int(getattr(args, "d2_M", 8))
    lam_on = float(getattr(args, "d2_lambda", 0.5))

    logits1, logits2, logits3, logits4 = model(enc, msk, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    pS1 = F.log_softmax(logits1, -1)
    pS2 = F.log_softmax(logits2, -1)
    pS3 = F.log_softmax(logits3, -1)
    pS4 = F.log_softmax(logits4, -1)
    pT1 = F.log_softmax(tl1, -1)
    pT2 = F.log_softmax(tl2, -1)
    pT3 = F.log_softmax(tl3, -1)
    pT4 = F.log_softmax(tl4, -1)
    skew_off = (_skew_kl(pT1, pS1, alpha) + _skew_kl(pT2, pS2, alpha)
                + _skew_kl(pT3, pS3, alpha) + _skew_kl(pT4, pS4, alpha)).mean()

    # on-policy sampled prefixes
    with torch.no_grad():
        q1 = torch.softmax(logits1.detach(), -1)
        s1 = torch.multinomial(q1, 1).squeeze(-1)
        ctx0 = model._last_ctx
        e1 = model.embedding(s1 + OFF[0])
        l2 = model.head2(torch.cat([ctx0, e1], 1))
        q2 = torch.softmax(l2, -1)
        s2 = torch.multinomial(q2, 1).squeeze(-1)
        e2 = model.embedding(s2 + OFF[1])
        l3 = model.head3(torch.cat([ctx0, e1, e2], 1))
        q3 = torch.softmax(l3, -1)
        s3 = torch.multinomial(q3, 1).squeeze(-1)
    pT1o, pT2o, pT3o, pT4o = teacher_digit_logp(t5, enc.to(t5.dtype), (~msk).long(), s1, s2, s3)
    e1o = model.embedding(s1 + OFF[0])
    e2o = model.embedding(s2 + OFF[1])
    e3o = model.embedding(s3 + OFF[2])
    l2o = model.head2(torch.cat([ctx0, e1o], 1))
    l3o = model.head3(torch.cat([ctx0, e1o, e2o], 1))
    l4o = model.head4(torch.cat([ctx0, e1o, e2o, e3o], 1))
    pS2o = F.log_softmax(l2o, -1)
    pS3o = F.log_softmax(l3o, -1)
    pS4o = F.log_softmax(l4o, -1)
    skew_on = (_jsd(pT1o, pS1.detach()) + _skew_kl(pT2o, pS2o, alpha)
               + _skew_kl(pT3o, pS3o, alpha) + _skew_kl(pT4o, pS4o, alpha)).mean()

    # contrastive over student-sampled sequences vs the GT positive
    with torch.no_grad():
        cand_quads = []
        for _ in range(M):
            q1s = torch.softmax(logits1.detach(), -1)
            a1 = torch.multinomial(q1s, 1).squeeze(-1)
            ea1 = model.embedding(a1 + OFF[0])
            la2 = model.head2(torch.cat([ctx0, ea1], 1))
            a2 = torch.multinomial(torch.softmax(la2, -1), 1).squeeze(-1)
            ea2 = model.embedding(a2 + OFF[1])
            la3 = model.head3(torch.cat([ctx0, ea1, ea2], 1))
            a3 = torch.multinomial(torch.softmax(la3, -1), 1).squeeze(-1)
            ea3 = model.embedding(a3 + OFF[2])
            la4 = model.head4(torch.cat([ctx0, ea1, ea2, ea3], 1))
            a4 = torch.multinomial(torch.softmax(la4, -1), 1).squeeze(-1)
            cand_quads.append(torch.stack([a1, a2, a3, a4], -1))
    quads = torch.stack(cand_quads, 1)                                   # (B,M,4)
    gt = torch.stack([gd1, gd2, gd3, gd4], -1).unsqueeze(1)              # (B,1,4)
    all_q = torch.cat([gt, quads], 1)                                    # (B,1+M,4)
    Bn, Mn = all_q.shape[:2]
    flat = all_q.reshape(Bn * Mn, 4)
    flat_enc = enc.repeat_interleave(Mn, 0)
    flat_msk = msk.repeat_interleave(Mn, 0)
    with torch.no_grad():
        tp1, tp2, tp3, tp4 = teacher_digit_logp(t5, flat_enc.to(t5.dtype), (~flat_msk).long(),
                                                flat[:, 0], flat[:, 1], flat[:, 2])
        ar = torch.arange(Bn * Mn, device=flat.device)
        t_seq = (tp1[ar, flat[:, 0]] + tp2[ar, flat[:, 1]]
                 + tp3[ar, flat[:, 2]] + tp4[ar, flat[:, 3]]).view(Bn, Mn)
    # Student candidate scores MUST remain connected to autograd;
    # computing these under no_grad disables the contrastive training term.
    ctx_r = ctx0.repeat_interleave(Mn, 0)
    ef1 = model.embedding(flat[:, 0] + OFF[0])
    ef2 = model.embedding(flat[:, 1] + OFF[1])
    ef3 = model.embedding(flat[:, 2] + OFF[2])
    lf1 = logits1.repeat_interleave(Mn, 0)
    lf2 = model.head2(torch.cat([ctx_r, ef1], 1))
    lf3 = model.head3(torch.cat([ctx_r, ef1, ef2], 1))
    lf4 = model.head4(torch.cat([ctx_r, ef1, ef2, ef3], 1))
    ar = torch.arange(Bn * Mn, device=flat.device)
    s_seq = (F.log_softmax(lf1, -1)[ar, flat[:, 0]]
             + F.log_softmax(lf2, -1)[ar, flat[:, 1]]
             + F.log_softmax(lf3, -1)[ar, flat[:, 2]]
             + F.log_softmax(lf4, -1)[ar, flat[:, 3]]).view(Bn, Mn)
    ctr = contrastive_student_sid_loss(
        s_seq, t_seq, temperature=float(getattr(args, "d2_ctr_temp", 1.0))
    )

    loss = skew_off + lam_on * (skew_on - skew_off) + beta * ctr
    return loss, {"skew_off": float(skew_off), "skew_on": float(skew_on), "ctr": float(ctr)}



def loss_smartgr(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads, theta_raw=None):
    """SmartGR: Hierarchy-Aware SID distillation + Beam-Aware Ranking distillation (MLP student)."""
    gd1, gd2, gd3, gd4 = batch[11], batch[12], batch[13], batch[14]
    bm_cb, bm_sc, bm_nv = extras[0], extras[1], extras[2]
    bm_slp = extras[3] if len(extras) > 3 else None
    B = enc.shape[0]
    K_beam = int(getattr(args, "sg_beam", 16))
    K = min(bm_cb.shape[1], K_beam)
    tau = float(getattr(args, "sg_tau", 1.0))
    tau_lev = float(getattr(args, "sg_tau_lev", 1.0))
    theta_max = float(getattr(args, "sg_theta_max", 5.0))
    lam_sid = float(getattr(args, "sg_lam_sid", 0.6))
    lam_beam = float(getattr(args, "sg_lam_beam", 0.3))

    cb = bm_cb[:, :K].long()
    slp = bm_slp[:, :K].float() if bm_slp is not None else None
    nv = bm_nv.clamp_max(K)
    valid = torch.arange(K, device=enc.device).unsqueeze(0) < nv.unsqueeze(1)
    gt = torch.stack([gd1, gd2, gd3, gd4], -1)

    match = (cb == gt.unsqueeze(1)).float()
    Lstar = torch.cumprod(match, dim=2).sum(-1).long()
    key = Lstar.float() - torch.arange(K, device=enc.device).float().unsqueeze(0) * 1e-6
    kstar = key.argmax(1)
    Ls = Lstar[torch.arange(B, device=enc.device), kstar]
    idx = torch.arange(B, device=enc.device)

    cb_s = cb[idx, kstar]
    c1, c2, c3 = cb_s[:, 0], cb_s[:, 1], cb_s[:, 2]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qT1, qT2, qT3, qT4 = teacher_digit_logp(t5, enc.to(t5.dtype), (~msk).long(), c1, c2, c3)
        l1s, l2s, l3s, l4s = model(enc, msk, d1_ids=c1 + DIGIT_OFFSETS[0],
                                   d2_ids=c2 + DIGIT_OFFSETS[1], d3_ids=c3 + DIGIT_OFFSETS[2])
    pS1 = F.log_softmax(l1s, -1)
    pS2 = F.log_softmax(l2s, -1)
    pS3 = F.log_softmax(l3s, -1)
    pS4 = F.log_softmax(l4s, -1)

    theta = theta_max * torch.sigmoid(theta_raw)
    depths = (torch.arange(1, 5, device=enc.device).float() / 4.0)
    a = torch.tanh(theta * depths / tau_lev) / torch.tanh(theta / tau_lev).clamp_min(1e-4)
    a_exp = torch.exp(a)
    cs = torch.cumsum(a_exp, 0)
    denom = torch.where(Ls > 0, cs[(Ls - 1).clamp(0, 3)], torch.ones_like(Ls.float()))
    w = (a_exp.unsqueeze(0).expand(B, 4) / denom.unsqueeze(1).clamp_min(1e-8))
    lvl = torch.arange(1, 5, device=enc.device).unsqueeze(0)
    w = w.masked_fill(~((lvl <= Ls.unsqueeze(1)) & (Ls.unsqueeze(1) > 0)), 0.0)

    kl = lambda qt, ps: (qt.exp() * (qt - ps)).sum(-1)
    kls = torch.stack([kl(qT1, pS1), kl(qT2, pS2), kl(qT3, pS3), kl(qT4, pS4)], -1)
    L_SID = (w * kls).sum(1).mean()

    kneg = kstar + 1
    ok = (Ls > 0) & (kneg < K) & valid[idx, kneg.clamp_max(K - 1)]
    L_BEAM = torch.zeros((), device=enc.device)
    if ok.any():
        cb_neg = cb[idx, kneg.clamp_max(K - 1)]
        two = torch.stack([cb_s, cb_neg], 1)
        s_steps = student_list_scores_steps(model, enc, msk, two, list_masks)
        S_S = s_steps.cumsum(-1)
        S_T = torch.stack([slp[idx, kstar], slp[idx, kneg.clamp_max(K - 1)]], 1).cumsum(-1)
        term = 0.0
        cnt = 0
        for lev in range(1, 5):
            m = ok & (Ls >= lev)
            if not m.any():
                continue
            pi_T = torch.softmax(S_T[:, :, lev - 1][m] / (lev * tau), -1)
            pi_S = torch.softmax(S_S[:, :, lev - 1][m] / (lev * tau), -1)
            term = term + (pi_T * (torch.log(pi_T.clamp_min(1e-10)) - torch.log(pi_S.clamp_min(1e-10)))).sum(-1).mean()
            cnt += 1
        if cnt > 0:
            L_BEAM = tau * tau * term / cnt

    loss = lam_sid * L_SID + lam_beam * L_BEAM
    return loss, {"sg_sid": float(L_SID), "sg_beam": float(L_BEAM),
                  "sg_Ls_mean": float(Ls.float().mean())}


LOSS_FN = {
    "rd": loss_rd,
    "rrd": loss_rrd,
    "rcek": loss_rcek,
    "lohrec": loss_lohrec,
    "gkd": loss_gkd,
    "distillm2": loss_distillm2,
    "smartgr": loss_smartgr,
}

NEEDS_BEAM_DUMP = {"rd", "rrd", "rcek", "lohrec", "smartgr"}
NEEDS_TEACHER = {"gkd", "distillm2", "smartgr"}


# ----------------------------------------------------------------------------
# training loop
# ----------------------------------------------------------------------------

def _fmt_float(value):
    text = f"{value:e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exponent)
    return f"{mantissa}e{exp}" if exp >= 0 else f"{mantissa}e-{abs(exp)}"


def _save_prefix(args):
    arch = str(getattr(args, "arch", "mlp")).lower()
    fields = [args.category, args.method, arch, "k0"]
    if arch == "mlp":
        fields.extend((f"h{args.num_heads}", f"f{args.ffn_dim}", f"ad{args.attn_dim}"))
    elif arch == "gru":
        fields.extend((f"gh{args.gru_hidden}", f"gl{args.gru_layers}"))
    elif arch == "cnn":
        fields.extend((f"cd{args.conv_dim}", f"ck{args.conv_kernel}", f"cl{args.conv_layers}"))
    else:
        raise ValueError(f"Unsupported student architecture: {arch}")
    fields.extend((f"hhid{args.head_hidden}", f"hl{args.head_layers}",
                   f"lr{_fmt_float(args.lr)}", f"seed{args.seed}"))
    if args.test:
        fields.append("debug")
    if args.run_tag:
        fields.append(args.run_tag)
    return "_".join(str(x) for x in fields)


def _checkpoint_args(args):
    return {k: v for k, v in vars(args).items()
            if not k.startswith("_") and not callable(v)}


def build_baseline_student(args, embedding_weight, device):
    """Build identical student backbones across external KD baselines."""
    arch = str(getattr(args, "arch", "mlp")).lower()
    if arch == "gru":
        model = SIDGRU(
            embedding_weight, embed_dim=embedding_weight.shape[1],
            gru_hidden=args.gru_hidden, gru_layers=args.gru_layers,
            dropout=args.dropout, head_hidden=args.head_hidden,
            head_layers=args.head_layers,
        ).to(device)
    elif arch == "cnn":
        model = SIDCNN(
            embedding_weight, embed_dim=embedding_weight.shape[1],
            conv_dim=args.conv_dim, kernel_size=args.conv_kernel,
            conv_layers=args.conv_layers, dropout=args.dropout,
            head_hidden=args.head_hidden, head_layers=args.head_layers,
        ).to(device)
    elif arch == "mlp":
        model = SIDMLP(
            embedding_weight, embed_dim=embedding_weight.shape[1],
            d_model=128, num_heads=args.num_heads, ffn_dim=args.ffn_dim,
            dropout=args.dropout, head_hidden=args.head_hidden,
            head_layers=args.head_layers, attn_dim=args.attn_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported student architecture: {arch}")
    return model


def install_on_policy_context_cache(model):
    """Expose prefix-scoring context to GKD/DistiLLM-2 for every backbone."""
    def _fwd_with_ctx(enc_seq, enc_mask, d1_ids=None, d2_ids=None, d3_ids=None):
        model._last_ctx = model.encode(enc_seq, enc_mask)
        ctx = model._last_ctx
        logits1 = model.head1(ctx)
        if d1_ids is None:
            d1_ids = logits1.argmax(-1) + DIGIT_OFFSETS[0]
        with torch.no_grad():
            e1 = model.embedding(d1_ids)
        logits2 = model.head2(torch.cat([ctx, e1], 1))
        if d2_ids is None:
            d2_ids = logits2.argmax(-1) + DIGIT_OFFSETS[1]
        with torch.no_grad():
            e2 = model.embedding(d2_ids)
        logits3 = model.head3(torch.cat([ctx, e1, e2], 1))
        if d3_ids is None:
            d3_ids = logits3.argmax(-1) + DIGIT_OFFSETS[2]
        with torch.no_grad():
            e3 = model.embedding(d3_ids)
        logits4 = model.head4(torch.cat([ctx, e1, e2, e3], 1))
        return logits1, logits2, logits3, logits4
    model.forward = _fwd_with_ctx
    return model


@torch.no_grad()
def _validate_digits(args, model, loader):
    model.eval()
    losses = []
    acc1 = np.zeros(4, dtype=np.int64)
    acc5 = np.zeros(4, dtype=np.int64)
    total = 0
    for batch in tqdm(loader, desc="Val digits", ncols=100):
        enc, msk = batch[0], batch[1]
        d1, d2, d3, _ = batch[3], batch[4], batch[5], batch[6]
        tl1, tl2, tl3, tl4 = batch[7], batch[8], batch[9], batch[10]
        gd1, gd2, gd3, gd4 = batch[11], batch[12], batch[13], batch[14]
        logits1, logits2, logits3, logits4 = model(enc, msk, d1_ids=d1, d2_ids=d2, d3_ids=d3)
        loss = (loss_fn(args, logits1, tl1, gd1) + loss_fn(args, logits2, tl2, gd2)
                + loss_fn(args, logits3, tl3, gd3) + d4_loss_fn(args, logits4, tl4, gd4))
        losses.append(float(loss.item()))
        total += gd4.shape[0]
        for i, (lg, lb) in enumerate(zip((logits1, logits2, logits3, logits4), (gd1, gd2, gd3, gd4))):
            c1, c5 = topk_correct(lg, lb, k=5)
            acc1[i] += c1
            acc5[i] += c5
    return {"loss": float(np.mean(losses)) if losses else float("inf"),
            "acc1": acc1 / max(total, 1), "acc5": acc5 / max(total, 1)}


def train_baseline(args):
    validate_selection_settings(args)
    if not getattr(args, "graph_dir", None):
        raise SystemExit("--graph_dir is required")
    graph_dir = os.path.abspath(args.graph_dir)
    method = args.method
    assert method in METHODS

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision(args, device)

    ckpt_dir = os.path.abspath(args.ckpt_dir or os.path.join(graph_dir, "SID-MLP", "ckpt"))
    log_dir = os.path.abspath(args.log_dir or os.path.join(graph_dir, "SID-MLP", "logs"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_prefix = _save_prefix(args)
    log_path = os.path.join(log_dir, f"{save_prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a") as h:
            print(message, file=h)

    teacher_ckpt = torch.load(os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False)
    embedding_weight = teacher_ckpt["t5.shared.weight"]
    del teacher_ckpt

    storage_dtype = torch.float16 if args.storage_fp16 else torch.float32
    _cpu_res = bool(getattr(args, "cpu_resident", False))
    train_ds = prepare_common_data(graph_dir, args.dataset_tag, device, storage_dtype,
                                   gpu_resident=not _cpu_res)
    val_ds = prepare_common_data(graph_dir, args.val_dataset_tag, device, storage_dtype,
                                 gpu_resident=not _cpu_res)

    valid_set = find_valid_set(args.dataset_tag, explicit_path=args.valid_set, graph_dir=graph_dir)
    if valid_set is None:
        raise FileNotFoundError(f"No valid_item_set for {args.dataset_tag}; run build_valid_set.py")
    vt = build_valid_tensors(valid_set, device)
    list_masks = (vt[0].clone(), vt[1].clone(), vt[2].clone(), vt[3].clone())
    keys4 = vt[4].long()
    valid_quads = torch.stack([
        (keys4 // (CODEBOOK_SIZE ** 3)) % CODEBOOK_SIZE,
        (keys4 // (CODEBOOK_SIZE ** 2)) % CODEBOOK_SIZE,
        (keys4 // CODEBOOK_SIZE) % CODEBOOK_SIZE,
        keys4 % CODEBOOK_SIZE,
    ], -1).to(device)

    if method in NEEDS_BEAM_DUMP:
        if not getattr(args, "beam_dump", None):
            raise SystemExit(f"--beam_dump is required for method={method}")
        attach_beam_dump(train_ds, args.beam_dump, device, topm=0)

    teacher_pipe = None
    t5 = None
    # Validation NDCG@10 always requires the frozen teacher history encoder,
    # including short (<5-epoch) runs.
    teacher_pipe = build_tiger_pipeline(args)
    t5 = teacher_pipe.model.t5.to(device).eval()
    for param in t5.parameters():
        param.requires_grad_(False)

    model = build_baseline_student(args, embedding_weight, device)
    arch = str(getattr(args, "arch", "mlp")).lower()
    install_on_policy_context_cache(model)

    if method == "smartgr":
        model.register_parameter("sg_theta_raw", torch.nn.Parameter(torch.zeros(1, device=device)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    make_collate = lambda ds: (lambda indices: collate_fn_gpu(ds, indices))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, collate_fn=make_collate(train_ds))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=make_collate(val_ds))

    split = require_validation_split(teacher_pipe.tokenized_datasets)
    beam_t5 = t5
    beam_loader = DataLoader(teacher_pipe.tokenized_datasets[split],
                             batch_size=args.val_batch_size, shuffle=False,
                             collate_fn=teacher_pipe.tokenizer.collate_fn[split],
                             num_workers=0)
    beam_vt = vt
    best_val_ndcg10 = float("-inf")
    best_val_epoch = None
    best_path = os.path.join(ckpt_dir, f"{save_prefix}_best_val_ndcg10.pth")

    log(f"Training baseline method={method} arch={arch} on {args.dataset}/{args.category}")
    log(f"train={len(train_ds)} val={len(val_ds)} ckpt_dir={ckpt_dir}")

    loss_impl = LOSS_FN[method]
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        comp_sums, comp_counts = {}, {}
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=100)):
            if args.test and batch_idx >= 5:
                break
            enc, msk = batch[0], batch[1]
            d1, d2, d3, _d4 = batch[3], batch[4], batch[5], batch[6]
            tl1, tl2, tl3, tl4 = batch[7], batch[8], batch[9], batch[10]
            gd1, gd2, gd3, gd4 = batch[11], batch[12], batch[13], batch[14]
            extras = batch[15:] if len(batch) > 15 else None

            optimizer.zero_grad(set_to_none=True)
            logits1, logits2, logits3, logits4 = model(enc, msk, d1_ids=d1, d2_ids=d2, d3_ids=d3)
            base_loss = (loss_fn(args, logits1, tl1, gd1) + loss_fn(args, logits2, tl2, gd2)
                         + loss_fn(args, logits3, tl3, gd3) + d4_loss_fn(args, logits4, tl4, gd4))
            kd_loss, comps = loss_impl(args, model, enc, msk, batch, extras, list_masks, t5, valid_quads,
                                       getattr(model, "sg_theta_raw", None))
            kd_w = float(getattr(args, "kd_weight", 1.0))
            loss = base_loss + kd_w * kd_loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
            for k, v in comps.items():
                comp_sums[k] = comp_sums.get(k, 0.0) + v
                comp_counts[k] = comp_counts.get(k, 0) + 1

        scheduler.step()
        digit_metrics = _validate_digits(args, model, val_loader)
        beam_metrics = {}
        if should_beam_validate(epoch, args.epochs, 5):
            beam_metrics = evaluate_loader(
                model, beam_t5, beam_loader, beam_vt, device,
                beam_size=args.beam_size, final_top_n=args.final_top_n,
                max_batches=(5 if args.test else None)
                if not getattr(args, "beam_val_max_batches", None)
                else int(args.beam_val_max_batches))
            best_val_ndcg10, improved = save_if_best_val_ndcg10(
                metrics=beam_metrics, epoch=epoch,
                previous_best=best_val_ndcg10, path=best_path,
                payload={
                    "model_state_dict": model.state_dict(),
                    "args": _checkpoint_args(args),
                    "model_type": f"SID-{arch.upper()}",
                },
                save=torch.save,
            )
            if improved:
                best_val_epoch = epoch
                log(f"BEST_VAL epoch={epoch} ndcg10={best_val_ndcg10:.6f} checkpoint={best_path}")
        comp_str = " ".join(f"{k}={comp_sums[k] / max(comp_counts[k], 1):.4f}" for k in sorted(comp_sums))
        log(f"epoch={epoch} train_loss={np.mean(train_losses):.6f} "
            f"val_loss={digit_metrics['loss']:.6f} "
            f"d1@1={digit_metrics['acc1'][0]:.4f} d2@1={digit_metrics['acc1'][1]:.4f} "
            f"d3@1={digit_metrics['acc1'][2]:.4f} d4@1={digit_metrics['acc1'][3]:.4f} "
            f"ndcg10={beam_metrics.get('ndcg10', 0.0):.6f}"
            + (f" {comp_str}" if comp_str else ""))

    final_path = os.path.join(ckpt_dir, f"{save_prefix}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "args": _checkpoint_args(args),
        "model_type": f"SID-{arch.upper()}",
        "checkpoint_role": "final_not_selected",
    }, final_path)
    log(f"Saved final checkpoint (not used for paper evaluation): {final_path}")
    if best_val_epoch is None:
        raise RuntimeError("No best-validation checkpoint saved; reportable training failed.")
    log(f"Best validation checkpoint: {best_path}, "
        f"epoch={best_val_epoch}, Val NDCG@10={best_val_ndcg10:.6f}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="KD baselines on MLP, GRU or CNN SID students")
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sem_ids", required=True)
    ap.add_argument("--dataset", default="Amazon2014")
    ap.add_argument("--category", required=True)
    ap.add_argument("--dataset_tag", required=True)
    ap.add_argument("--val_dataset_tag", required=True)
    ap.add_argument("--graph_dir", required=True)
    ap.add_argument("--valid_set", default=None)
    ap.add_argument("--beam_dump", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--run_tag", default="")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--val_batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--arch", type=str.lower, default="mlp",
                    choices=["mlp", "gru", "cnn"])
    ap.add_argument("--gru_hidden", type=int, default=384)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--conv_dim", type=int, default=256)
    ap.add_argument("--conv_kernel", type=int, default=3)
    ap.add_argument("--conv_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--attn_dim", type=int, default=384)
    ap.add_argument("--ffn_dim", type=int, default=1024)
    ap.add_argument("--head_hidden", type=int, default=512)
    ap.add_argument("--head_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--loss", default="combined")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--d4_teacher", default="combined")
    ap.add_argument("--beam_size", type=int, default=50)
    ap.add_argument("--final_top_n", type=int, default=50)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--storage_fp16", action="store_true")
    ap.add_argument("--cpu_resident", action="store_true")
    ap.add_argument("--beam_val_max_batches", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kd_weight", type=float, default=1.0)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--test", action="store_true")
    # RD
    ap.add_argument("--rd_topk", type=int, default=50)
    ap.add_argument("--rd_T", type=float, default=2.0)
    ap.add_argument("--rd_mu", type=float, default=0.01)
    ap.add_argument("--rd_dyn", type=int, default=100)
    # RRD
    ap.add_argument("--rrd_K", type=int, default=10)
    ap.add_argument("--rrd_L", type=int, default=40)
    # RCE-KD
    ap.add_argument("--mkd_K", type=int, default=50)
    ap.add_argument("--mkd_L", type=int, default=50)
    ap.add_argument("--mkd_tau", type=float, default=1.0)
    ap.add_argument("--mkd_T", type=float, default=10.0)
    ap.add_argument("--mkd_beta", type=float, default=5.0)
    # LOHRec
    ap.add_argument("--rank_margin", type=float, default=0.001)
    ap.add_argument("--rank_weight", type=float, default=1.0)
    ap.add_argument("--pos_neg_weight", type=float, default=1.0)
    ap.add_argument("--adaptive", action=argparse.BooleanOptionalAction, default=True)
    # GKD
    ap.add_argument("--gkd_lambda", type=float, default=0.5)
    # DistiLLM-2
    ap.add_argument("--d2_alpha", type=float, default=0.9)
    ap.add_argument("--d2_beta", type=float, default=0.1)
    ap.add_argument("--d2_M", type=int, default=8)
    ap.add_argument("--d2_lambda", type=float, default=0.5)
    ap.add_argument("--d2_ctr_temp", type=float, default=1.0)
    ap.add_argument("--sg_beam", type=int, default=16)
    ap.add_argument("--sg_tau", type=float, default=1.0)
    ap.add_argument("--sg_tau_lev", type=float, default=1.0)
    ap.add_argument("--sg_theta_max", type=float, default=5.0)
    ap.add_argument("--sg_lam_sid", type=float, default=0.6)
    ap.add_argument("--sg_lam_beam", type=float, default=0.3)
    args = ap.parse_args()
    train_baseline(args)


if __name__ == "__main__":
    main()
