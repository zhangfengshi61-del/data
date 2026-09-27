"""Evaluate and benchmark MI-SID under the RQ-SID RecBoard protocol.

Quality uses full-catalog ranking with seen-item masking.
Speed uses the requested TIGER reference configuration: batch=96, fp32,
full Beauty/Sports/Toys test split, and end-to-end elapsed time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
sys.path.insert(0, "/data/fszhang/RecBoard-master/SDQ-403")

import freerec
from hiflow_lib.common import build_dataset, item_count, make_tokenizer_converter
from hiflow_lib.model_misid import MISIDRec
from hiflow_lib.model_sidmlp import SIDMLPRec

ROOT = Path("/data/fszhang/RecBoard-master")
HIF = ROOT / "HiFlow-SID"
PY = "/data/fszhang/anaconda/envs/myenv_t5/bin/python"

CAT = sys.argv[1] if len(sys.argv) > 1 else "Beauty"
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
DEV = f"cuda:{GPU}"
DS = f"Amazon2014{CAT}_550_LOU"
VOCAB = HIF / f"logs/sid_vocab_rqvae_{CAT}.json"
DEFAULT_CKPT = HIF / f"logs/MI-SID/{DS}/misid-rq-{CAT}/best.pt"
CKPT = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CKPT
TIGER_REFERENCE_USERS = 22363
TIGER_REFERENCE_SEC = 40.83
SIDMLP_CKPT = HIF / f"logs/SID-MLP-RecBoard/{DS}/sidmlp-rq-{CAT}/best.pt"
SIDMLP_TEST_GATE = {
    "Beauty": {"NDCG@10": 0.0300, "Recall@10": 0.0568},
    "Sports": {"NDCG@10": 0.0133, "Recall@10": 0.0246},
    "Toys": {"NDCG@10": 0.0282, "Recall@10": 0.0509},
}


ENCODER_TYPE = sys.argv[4] if len(sys.argv) > 4 else "fast"
EXPERT_RANK = int(sys.argv[5]) if len(sys.argv) > 5 else 8

CFG = SimpleNamespace(
    embedding_dim=128,
    num_codewords=256,
    num_slots=4,
    fast_hidden=256,
    fast_layers=2,
    fast_max_seq_len=128,
    fast_num_positions=6,
    encoder_type=ENCODER_TYPE,
    encoder_heads=4,
    expert_rank=EXPERT_RANK,
    tau_align=0.1,
    lam_rank=1.0,
    lam_sid=0.5,
    lam_align=0.2,
    lam_diversity=0.01,
    expert_negatives=2048,
)


def build_model(ds):
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(VOCAB), item_count(ds)
    )
    model = MISIDRec(ds, CFG, tokenizer, converter, code_of_tok, item_codes)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def build_sidmlp(ds):
    from hiflow_lib.common import make_tokenizer_converter
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(VOCAB), item_count(ds)
    )
    cfg = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64,
        intermediate_size=256, num_layers=6, num_heads=4,
        dropout_rate=0.1, num_codewords=256, attn_dim=128,
        ffn_dim=256, head_hidden=512, head_layers=1,
        candidate_chunk=2048, tau_align=0.1, lam_align=0.1,
    )
    model = SIDMLPRec(
        ds, cfg, tokenizer, converter, code_of_tok, item_codes
    )
    state = torch.load(SIDMLP_CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    return model


def make_loader(ds, model, split, batch_size):
    from torchdata.dataloader2 import DataLoader2

    from freerec.data.tags import ID, ITEM, SEQUENCE

    seq = ds.fields[ITEM, ID].fork(SEQUENCE)
    if split == "valid":
        source = ds.valid().ordered_user_ids_source().valid_sampling_("full")
    else:
        source = ds.test().ordered_user_ids_source().test_sampling_("full")
    pipe = (
        source
        .lprune_(20, modified_fields=(seq,))
        .map_(model.format_item_ids, modified_fields=(seq,))
        .map_(model.encode_item_text, modified_fields=(seq,))
        .batch_(batch_size)
        .tensor_()
    )
    return DataLoader2(datapipe=pipe), seq


def metrics(scores, targets):
    from freerec.metrics import hit_rate, mean_reciprocal_rank, normalized_dcg

    result = {}
    for k in (1, 5, 10, 20):
        result[f"Hit@{k}"] = hit_rate(scores, targets, k=k).mean().item()
        result[f"Recall@{k}"] = hit_rate(scores, targets, k=k).mean().item()
        result[f"NDCG@{k}"] = normalized_dcg(scores, targets, k=k).mean().item()
        result[f"MRR@{k}"] = mean_reciprocal_rank(scores, targets, k=k).mean().item()
    return result


@torch.inference_mode()
def evaluate_split(model, ds, split, device):
    loader, seq = make_loader(ds, model, split, 64)
    ISeen = ds.fields[freerec.data.tags.ITEM, freerec.data.tags.ID].fork(
        freerec.data.tags.SEEN
    )
    IUnseen = ds.fields[freerec.data.tags.ITEM, freerec.data.tags.ID].fork(
        freerec.data.tags.UNSEEN
    )
    predictions = []
    truths = []
    for data in loader:
        values = {seq: data[seq]}
        scores = model.recommend_from_full(values)
        scores = scores.to(device)
        seen_mask = torch.zeros_like(scores, dtype=torch.bool)
        targets = torch.zeros_like(scores)
        for row, ids in enumerate(data[ISeen]):
            if len(ids):
                seen_mask[row, torch.tensor(list(ids), device=device)] = True
        for row, ids in enumerate(data[IUnseen]):
            if len(ids):
                targets[row, torch.tensor(list(ids), device=device)] = 1.0
        scores = scores.masked_fill(seen_mask, -1e23)
        predictions.append(scores.cpu())
        truths.append(targets.cpu())
    return metrics(torch.cat(predictions), torch.cat(truths))


@torch.inference_mode()
def benchmark(model, ds, device):
    loader, seq = make_loader(ds, model, "test", 96)
    for index, data in enumerate(loader):
        model.recommend_from_full({seq: data[seq]})
        if index >= 1:
            break
    torch.cuda.synchronize(device)
    loader, seq = make_loader(ds, model, "test", 96)
    count = 0
    start = time.perf_counter()
    for data in loader:
        model.recommend_from_full({seq: data[seq]})
        count += len(data[seq])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "users": count,
        "batch_size": 96,
        "dtype": "fp32",
        "warmup_batches": 2,
        "elapsed_sec": elapsed,
        "throughput_users_s": count / max(elapsed, 1e-9),
        "tiger_reference_users": TIGER_REFERENCE_USERS,
        "tiger_reference_sec_beauty": TIGER_REFERENCE_SEC,
        "tiger_reference_sec_scaled": TIGER_REFERENCE_SEC * count / TIGER_REFERENCE_USERS,
        "speedup_vs_tiger_40_83s": TIGER_REFERENCE_SEC / elapsed,
        "speedup_vs_tiger_scaled": (
            TIGER_REFERENCE_SEC * count / TIGER_REFERENCE_USERS
        ) / elapsed,
    }


def main():
    if not CKPT.exists():
        raise FileNotFoundError(f"missing checkpoint: {CKPT}")
    ds = build_dataset(CAT)
    model = build_model(ds).to(DEV).float().eval()
    print(f"[MI-SID] category={CAT} checkpoint={CKPT}", flush=True)
    valid = evaluate_split(model, ds, "valid", DEV)
    print(f"[MI-SID] valid={valid}", flush=True)
    test = evaluate_split(model, ds, "test", DEV)
    print(f"[MI-SID] test={test}", flush=True)
    speed = benchmark(model, ds, DEV)
    print(f"[MI-SID] speed={speed}", flush=True)
    sidmlp_speed = None
    if SIDMLP_CKPT.exists():
        sidmlp = build_sidmlp(ds).to(DEV).float().eval()
        sidmlp_speed = benchmark(sidmlp, ds, DEV)
        sidmlp_speed["method"] = "SID-MLP-RQ"
        print(f"[SID-MLP] speed={sidmlp_speed}", flush=True)
        del sidmlp
        torch.cuda.empty_cache()
    gate_target = SIDMLP_TEST_GATE[CAT]
    quality_gate = (
        test["NDCG@10"] >= gate_target["NDCG@10"]
        and test["Recall@10"] >= gate_target["Recall@10"]
    )
    speed_gate = (
        speed["elapsed_sec"] <= speed["tiger_reference_sec_scaled"] / 10.0
        and sidmlp_speed is not None
        and speed["elapsed_sec"] < sidmlp_speed["elapsed_sec"]
    )
    gates = {
        "quality_vs_sidmlp": quality_gate,
        "speed_10x_vs_tiger": speed["elapsed_sec"] <= speed["tiger_reference_sec_scaled"] / 10.0,
        "faster_than_sidmlp": (
            sidmlp_speed is not None
            and speed["elapsed_sec"] < sidmlp_speed["elapsed_sec"]
        ),
        "accepted": quality_gate and speed_gate,
        "sidmlp_quality_gate": gate_target,
    }
    print(f"[MI-SID] gates={gates}", flush=True)
    output = {
        "method": "MI-SID",
        "dataset": CAT,
        "sid_vocab": str(VOCAB),
        "checkpoint": str(CKPT),
        "encoder_type": ENCODER_TYPE,
        "expert_rank": EXPERT_RANK,
        "protocol": {
            "max_history": 20,
            "ranking": "full",
            "seen_mask": True,
            "seed": 2025,
        },
        "valid": valid,
        "test": test,
        "speed": speed,
        "sidmlp_speed": sidmlp_speed,
        "gates": gates,
    }
    suffix = "" if ENCODER_TYPE == "fast" else "_" + ENCODER_TYPE
    dst = HIF / f"results/misid_rq_protocol_{CAT}{suffix}.json"
    dst.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[MI-SID] saved {dst}", flush=True)


if __name__ == "__main__":
    main()
