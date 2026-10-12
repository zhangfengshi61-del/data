"""GPU-resident data packing for the public SID-MLP path."""

import gc
import json
import os
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import DIGIT_OFFSETS, token_to_codebook


class CommonLensDataset(Dataset):
    """GPU-resident dataset used by SID-MLP training."""

    def __init__(self, enc_seq, enc_mask, hk, digits, teacher_logits, gt_digits):
        self.enc_seq = enc_seq
        self.enc_mask = enc_mask
        self.hk = hk
        self.digits = digits
        self.teacher_logits = teacher_logits
        self.gt_digits = gt_digits
        # s24: optional teacher-beam dump (search-distribution distillation targets)
        self.bm_codes = None    # (N, M, 4) uint8 codebook-space, sorted by exact teacher score desc
        self.bm_scores = None   # (N, M) fp16 exact sequence scores (sum digit logprobs + eos)
        self.bm_nvalid = None   # (N,) int16
        self.bm_steplp = None       # s41 CDRS: (N, M, 4) teacher per-digit logp
        self.bm_score_scale = None  # s41 CDRS: (4,) per-depth teacher-gap std
        self.bm_step_scale = None    # s43e DA-CDRS: (4,) per-depth teacher INCREMENT-gap std

    def __len__(self):
        return self.enc_seq.shape[0]

    def __getitem__(self, idx):
        return idx

    @property
    def device(self):
        return getattr(self, "_device", torch.device("cuda"))


def collate_fn_gpu(dataset: CommonLensDataset, indices: List[int]):
    dev = dataset.device
    idx = torch.tensor(indices, dtype=torch.long)

    enc_seq = dataset.enc_seq[idx].float().to(dev, non_blocking=True)
    enc_mask = dataset.enc_mask[idx].to(dev, non_blocking=True)
    hk = None

    digits = dataset.digits[idx].to(dev, non_blocking=True)
    d1, d2, d3, d4 = digits[:, 0], digits[:, 1], digits[:, 2], digits[:, 3]

    tl = dataset.teacher_logits[idx].float().to(dev, non_blocking=True)
    tl1, tl2, tl3, tl4 = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3]

    gt = dataset.gt_digits[idx].to(dev, non_blocking=True)
    gd1, gd2, gd3, gd4 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

    base = (enc_seq, enc_mask, hk, d1, d2, d3, d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4)
    if dataset.bm_codes is None:
        return base
    # s24 extras: candidate codes (long, codebook space), exact scores, valid counts
    bm_cb = dataset.bm_codes[idx].long().to(dev, non_blocking=True)
    bm_sc = dataset.bm_scores[idx].float().to(dev, non_blocking=True)
    bm_nv = dataset.bm_nvalid[idx].long().to(dev, non_blocking=True)
    extras = (bm_cb, bm_sc, bm_nv)
    if getattr(dataset, 'bm_steplp', None) is not None:  # s41 CDRS
        extras = extras + (dataset.bm_steplp[idx].float().to(dev, non_blocking=True),)
    return base + extras


def attach_beam_dump(dataset: CommonLensDataset, dump_path: str, device: torch.device, topm: int = 0):  # s36
    """s24: attach the teacher-beam dump (codes/scores/nvalid) to a packed dataset.
    Alignment is by row index — the dump was produced by iterating the same packed cache."""
    d = torch.load(dump_path, map_location='cpu', weights_only=False)
    codes, scores, nvalid = d['codes'], d['scores'], d['nvalid']
    if codes.ndim != 3 or codes.shape[-1] != 4:
        raise ValueError(f"Expected beam candidate codes (N,W,4): {dump_path}")
    nrows, width = codes.shape[:2]
    if scores.shape != (nrows, width) or nvalid.shape != (nrows,):
        raise ValueError("Beam codes, scores, nvalid shapes disagree")
    if 'step_logp' in d and d['step_logp'].shape != (nrows, width, 4):
        raise ValueError("Beam step_logp is not aligned with (N,W,4) codes")
    if (nvalid < 1).any() or (nvalid > width).any():
        raise ValueError("Beam dump has invalid candidate counts")
    if topm and int(topm) > width:
        raise ValueError(
            f"Requested M={topm} candidate slots but beam dump has width={width}; "
            "a beam-search width of 40 does not by itself prove 64 final candidates"
        )
    if topm and int(topm) > 0:  # teacher-score-sorted top-M truncation
        W0 = d['scores'].shape[1]
        ar = torch.arange(W0).unsqueeze(0)
        sc = d['scores'].float().masked_fill(ar >= d['nvalid'].unsqueeze(1), float('-inf'))
        order = sc.argsort(dim=1, descending=True)
        d = dict(d)
        d['codes'] = d['codes'].gather(1, order.unsqueeze(-1).expand(-1, -1, d['codes'].shape[-1]))[:, :topm].contiguous()
        d['scores'] = d['scores'].gather(1, order)[:, :topm].contiguous()
        # Teacher per-depth log-probabilities MUST follow the identical candidate
        # permutation: DA-CDRS uses these scores for teacher-depth residuals.
        if 'step_logp' in d:
            if d['step_logp'].shape[:2] != order.shape:
                raise RuntimeError("beam dump step_logp and candidate-score shapes do not match")
            d['step_logp'] = d['step_logp'].gather(
                1, order.unsqueeze(-1).expand_as(d['step_logp'])
            )[:, :topm].contiguous()
        d['nvalid'] = d['nvalid'].clamp_max(topm)
        d['W'] = int(topm)
        print(f"s36: beam dump truncated to top-{topm} by teacher score", flush=True)
    n_ds, n_dump = len(dataset), d['codes'].shape[0]
    if n_ds != n_dump:
        raise RuntimeError(f"beam dump rows ({n_dump}) != dataset rows ({n_ds}): {dump_path}")
    if int(d['nvalid'].min()) < 1:
        raise RuntimeError(f"beam dump has empty candidate lists: {dump_path}")
    active = torch.arange(d['scores'].shape[1]).unsqueeze(0) < d['nvalid'].unsqueeze(1)
    if not torch.isfinite(d['scores'][active]).all():
        raise ValueError("Non-finite scores on valid beam candidates")
    # Training data may include a candidate-dependent EOS contribution
    # while step_logp has only four SID increments. Log the discrepancy;
    # do not silently force their equality or rewrite historic scores.
    if 'step_logp' in d:
        missing = d['scores'].float() - d['step_logp'].float().sum(-1)
        spread = (missing.masked_fill(~active, -float('inf')).max(1).values
                  - missing.masked_fill(~active, float('inf')).min(1).values)
        print(f"BEAM_SCORE_AUDIT: max_candidate_dependent_score_remainder="
              f"{spread.max().item():.5f} "
              f"(may include EOS or a distinct score convention)", flush=True)
    dataset.bm_codes = d['codes'].to(device)
    dataset.bm_scores = d['scores'].to(device)
    dataset.bm_nvalid = d['nvalid'].to(device)
    dataset.bm_steplp = d['step_logp'].to(device) if 'step_logp' in d else None  # s41 CDRS
    dataset.bm_score_scale = d['score_scale'].to(device) if 'score_scale' in d else None  # s41 CDRS
    dataset.bm_step_scale = d['step_scale'].to(device) if 'step_scale' in d else None  # s43e DA-CDRS
    print(f"Attached beam dump {dump_path}: N={n_dump} W={d['W']} "
          f"nvalid_mean={d['nvalid'].float().mean():.1f} anchors={d['meta'].get('anchors')}", flush=True)
    return dataset


def _pack_from_raw(graph_dir: str, dataset_tag: str, storage_dtype: torch.dtype) -> dict:
    enc_path = os.path.join(graph_dir, "hidden_states", dataset_tag, "encoder_sequences.pt")
    log_path = os.path.join(graph_dir, "logits", dataset_tag, "logits.pt")
    map_path = os.path.join(graph_dir, "hidden_states", dataset_tag, "item_mapping.json")

    enc_dict = torch.load(enc_path, map_location="cpu", weights_only=False, mmap=True)
    logits_dict = torch.load(log_path, map_location="cpu", weights_only=False, mmap=True)

    with open(map_path) as f:
        mapping = json.load(f)
    item_sem = {int(k): v["semantic_id"] for k, v in mapping.items()}

    total_samples = 0
    max_seq_len = 0
    skipped = 0
    valid_items = []

    for idx in tqdm(list(enc_dict.keys()), desc="Counting samples", ncols=100):
        if idx not in logits_dict or idx not in item_sem:
            skipped += 1
            continue

        gt_d1, gt_d2, gt_d3, gt_d4 = item_sem[idx]
        gd1 = token_to_codebook(gt_d1, 0)
        gd2 = token_to_codebook(gt_d2, 1)
        gd3 = token_to_codebook(gt_d3, 2)
        gd4 = token_to_codebook(gt_d4, 3)

        n_ctx = min(len(enc_dict[idx]), logits_dict[idx].shape[0])
        for c in range(n_ctx):
            enc_c = enc_dict[idx][c] if isinstance(enc_dict[idx], list) else enc_dict[idx][c]
            max_seq_len = max(max_seq_len, enc_c.shape[0])

        total_samples += n_ctx
        valid_items.append((idx, n_ctx, gt_d1, gt_d2, gt_d3, gt_d4, gd1, gd2, gd3, gd4))

    print(f"Found {total_samples} samples, max_seq_len={max_seq_len}, skipped items={skipped}", flush=True)

    enc_seq_all = torch.zeros(total_samples, max_seq_len, 128, dtype=storage_dtype)
    enc_lens = torch.zeros(total_samples, dtype=torch.int32)
    digits_all = torch.zeros(total_samples, 4, dtype=torch.int64)
    tl_all = torch.zeros(total_samples, 4, 256, dtype=storage_dtype)
    gt_all = torch.zeros(total_samples, 4, dtype=torch.int64)

    sample_idx = 0
    for idx, n_ctx, gt_d1, gt_d2, gt_d3, gt_d4, gd1, gd2, gd3, gd4 in tqdm(
        valid_items, desc="Packing tensors", ncols=100
    ):
        for c in range(n_ctx):
            enc_c = enc_dict[idx][c]
            sl = enc_c.shape[0]
            enc_seq_all[sample_idx, :sl, :] = (
                enc_c.to(storage_dtype) if enc_c.dim() == 2 else enc_c.unsqueeze(0).to(storage_dtype)
            )
            enc_lens[sample_idx] = sl
            digits_all[sample_idx] = torch.tensor([gt_d1, gt_d2, gt_d3, gt_d4], dtype=torch.int64)

            full = logits_dict[idx][c]
            for d in range(4):
                start = DIGIT_OFFSETS[d]
                tl_all[sample_idx, d, :] = full[d, start:start + 256].to(storage_dtype)

            gt_all[sample_idx] = torch.tensor([gd1, gd2, gd3, gd4], dtype=torch.int64)
            sample_idx += 1

    del enc_dict, logits_dict
    gc.collect()

    arange = torch.arange(max_seq_len).unsqueeze(0)
    enc_mask_all = (arange >= enc_lens.unsqueeze(1)).to(torch.bool)
    del enc_lens

    return {
        "enc_seq": enc_seq_all,
        "enc_mask": enc_mask_all,
        "digits": digits_all,
        "teacher_logits": tl_all,
        "gt_digits": gt_all,
    }


def _cache_path(graph_dir: str, dataset_tag: str, storage_dtype: torch.dtype) -> str:
    dtype_tag = "fp16" if storage_dtype == torch.float16 else "fp32"
    cache_dir = os.path.join(graph_dir, "hidden_states", dataset_tag, "packed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"k0_{dtype_tag}.pt")


def prepare_common_data(graph_dir: str, dataset_tag: str, device: torch.device,
                        storage_dtype: torch.dtype = torch.float32,
                        gpu_resident: bool = True):
    cache_file = _cache_path(graph_dir, dataset_tag, storage_dtype)

    if os.path.exists(cache_file):
        print(f"Loading packed cache (mmap): {cache_file}", flush=True)
        cached = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)
    else:
        print("No cache found, packing from raw data ...", flush=True)
        cached = _pack_from_raw(graph_dir, dataset_tag, storage_dtype)
        print(f"Saving packed cache: {cache_file}", flush=True)
        torch.save(cached, cache_file)
        del cached
        gc.collect()
        cached = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)

    total_samples = cached["enc_seq"].shape[0]
    if gpu_resident:
        print(f"Moving {total_samples} samples to {device} ...", flush=True)
        enc_seq_gpu = cached["enc_seq"].to(device, non_blocking=True)
        enc_mask_gpu = cached["enc_mask"].to(device, non_blocking=True)
        digits_gpu = cached["digits"].to(device, non_blocking=True)
        tl_gpu = cached["teacher_logits"].to(device, non_blocking=True)
        gt_gpu = cached["gt_digits"].to(device, non_blocking=True)
        del cached
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            allocated_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"GPU memory after data load: {allocated_gb:.2f} GB", flush=True)
        ds = CommonLensDataset(enc_seq_gpu, enc_mask_gpu, None, digits_gpu, tl_gpu, gt_gpu)
    else:
        print(f"CPU-resident mode: {total_samples} samples stay in RAM, streaming to {device}", flush=True)
        ds = CommonLensDataset(cached["enc_seq"], cached["enc_mask"], None,
                               cached["digits"], cached["teacher_logits"], cached["gt_digits"])
        del cached
        gc.collect()
    ds._device = device
    return ds
