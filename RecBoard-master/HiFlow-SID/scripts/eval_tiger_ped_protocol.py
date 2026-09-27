"""Evaluate and benchmark TIGER-PED under the RQ-SID RecBoard protocol.

Quality uses full-catalog ranking with seen-item masking.
Speed uses the requested TIGER reference configuration: batch=96, fp32,
full Beauty/Sports/Toys test split, and end-to-end elapsed time.
"""

from __future__ import annotations

import json
import os
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
from hiflow_lib.model_hiflow import HiFlowRec
from hiflow_lib.model_sidmlp import SIDMLPRec

ROOT = Path("/data/fszhang/RecBoard-master")
HIF = ROOT / "HiFlow-SID"
PY = "/data/fszhang/anaconda/envs/myenv_t5/bin/python"

CAT = sys.argv[1] if len(sys.argv) > 1 else "Beauty"
GPU = sys.argv[2] if len(sys.argv) > 2 else "0"
DEV = f"cuda:{GPU}"
DS = f"Amazon2014{CAT}_550_LOU"
VOCAB = HIF / f"logs/sid_vocab_rqvae_{CAT}.json"
DEFAULT_CKPT = HIF / f"logs/TIGER-PED/{DS}/ped-{CAT}/best.pt"
CKPT = Path(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_CKPT
TIGER_REFERENCE_USERS = 22363
TIGER_REFERENCE_SEC = 40.83
SIDMLP_CKPT = HIF / f"logs/SID-MLP-RecBoard/{DS}/sidmlp-rq-{CAT}/best.pt"
SIDMLP_TEST_GATE = {
    "Beauty": {"NDCG@10": 0.0300, "Recall@10": 0.0568},
    "Sports": {"NDCG@10": 0.0133, "Recall@10": 0.0246},
    "Toys": {"NDCG@10": 0.0282, "Recall@10": 0.0509},
}


CFG = SimpleNamespace(
    embedding_dim=128, attention_size=64, intermediate_size=256,
    num_layers=int(os.environ.get("PED_ENCODER_LAYERS","6")), num_heads=4, dropout_rate=0.1, num_codewords=256,
    num_slots=4, fast_hidden=256, fast_layers=2, fast_max_seq_len=128,
    fast_num_positions=6, expert_score=True, expert_score_t5=True,
    fast_encoder=False, distill_encoder=False, expert_rank=int(os.environ.get("PED_EXPERT_RANK","8")),
    expert_multiplicative=bool(int(os.environ.get("PED_EXPERT_MULT","0"))),
    history_attention_pool=bool(int(os.environ.get("PED_POOL","0"))),
    expert_score_scale=1.0, expert_negatives=2048, tau_align=0.1,
    lam_align=0.1, lam_prefix=0.5, lam_parallel=1.0,
    lam_expert_rank=1.0, tau_joint=0.1, use_joint=True,
    lam_joint_score=0.1, maxlen=20, flow_steps=1, top_b=-1,
    use_prior=True, single_expert=False, prefix_mode="soft",
)


def build_model(ds):
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(VOCAB), item_count(ds)
    )
    model = HiFlowRec(ds, CFG, tokenizer, converter, code_of_tok, item_codes)
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    is_ped = any(key.startswith("backbone.") for key in state)
    if is_ped:
        current = model.state_dict()
        compatible = {
            key: value for key, value in state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"[TIGER-PED] loaded PED checkpoint keys={len(compatible)} missing={len(missing)}", flush=True)
    else:
        enc_state = {}
        for key, value in state.items():
            if key.startswith("t5."):
                key = key[3:]
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            if key == "shared.weight" or key.startswith("encoder."):
                enc_state[key] = value
        missing, unexpected = model.backbone.load_state_dict(enc_state, strict=False)
        with torch.no_grad():
            shared = model.backbone.shared.weight
            for level in range(model.num_sid_tokens):
                for code in range(model.num_codewords):
                    token = f"<sid_{level}_{code}>"
                    tid = tokenizer.convert_tokens_to_ids(token)
                    if tid is not None and int(tid) < shared.size(0):
                        model.code_embs[level].weight[code].copy_(shared[int(tid)])
        print(f"[TIGER-PED] loaded TIGER encoder keys={len(enc_state)} missing={len(missing)}", flush=True)
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
    print(f"[TIGER-PED] category={CAT} checkpoint={CKPT}", flush=True)
    valid = evaluate_split(model, ds, "valid", DEV)
    print(f"[TIGER-PED] valid={valid}", flush=True)
    test = evaluate_split(model, ds, "test", DEV)
    print(f"[TIGER-PED] test={test}", flush=True)
    speed = benchmark(model, ds, DEV)
    print(f"[TIGER-PED] speed={speed}", flush=True)
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
    print(f"[TIGER-PED] gates={gates}", flush=True)
    output = {
        "method": "TIGER-PED",
        "dataset": CAT,
        "sid_vocab": str(VOCAB),
        "checkpoint": str(CKPT),
        "encoder": "original_TIGER_T5_encoder",
        "decoder": "parallel_expert_no_beam",
        "expert_rank": CFG.expert_rank,
        "expert_score_scale": CFG.expert_score_scale,
        "joint_energy": True,
        "protocol": {
            "max_history": 20,
            "ranking": "full",
            "seen_mask": True,
            "seed": 2025,
            "device": DEV,
            "dtype": "fp32",
            "eval_batch_size": 96,
        },
        "valid": valid,
        "test": test,
        "speed": speed,
        "sidmlp_speed": sidmlp_speed,
        "gates": gates,
        "fairness": {
            "teacher_or_distillation": False,
            "tiger_beam_candidates": False,
            "external_candidates": False,
            "second_stage_rerank": False,
            "full_catalog_single_stage": True,
            "test_evaluated_once_after_selection": True,
        },
    }
    suffix = ""
    dst = HIF / f"results/tiger_ped_protocol_{CAT}{suffix}.json"
    dst.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"[TIGER-PED] saved {dst}", flush=True)


if __name__ == "__main__":
    main()
