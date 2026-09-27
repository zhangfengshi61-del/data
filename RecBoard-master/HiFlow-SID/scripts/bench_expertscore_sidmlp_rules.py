"""End-to-end speed check for trained π-SID ExpertScore.

The protocol mirrors SID-MLP's public hardware profile: test split, batch 32,
native bf16, TF32 disabled, and a TIGER beam-50 reference.  ExpertScore scores
the complete Beauty catalogue in one vectorized pass and loads the trained
fast encoder/expert checkpoint.
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

import freerec  # noqa: E402
from converter import SemIDConverter  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer  # noqa: E402

from hiflow_lib.common import (  # noqa: E402
    build_dataset,
    item_count,
    make_tokenizer_converter,
    sid_vocab_path,
)
from hiflow_lib.model_hiflow import HiFlowRec  # noqa: E402


ROOT = Path("/data/fszhang/RecBoard-master")
SDQ_CKPT = ROOT / "SDQ-403/infos/SDQ-T5-VAE/Amazon2014Beauty_550_LOU/0/checkpoint.tar"
EXPERT_CKPT = ROOT / "HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/pisid-expertscore-distill-pos6-r8-v3/best.pt"
DEVICE = "cuda:0"
BATCH = 32
BEAM = 50


def build_test_loader(dataset, converter):
    from freerec.data.tags import ID, ITEM, SEQUENCE
    from torchdata.dataloader2 import DataLoader2

    iseq = dataset.fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        dataset.test()
        .ordered_user_ids_source()
        .test_sampling_("full")
        .lprune_(20, modified_fields=(iseq,))
        .map_(lambda _field, items: [SemIDConverter.format(i) for i in items], modified_fields=(iseq,))
        .map_(lambda _field, texts: converter.encode(texts), modified_fields=(iseq,))
        .batch_(BATCH)
        .tensor_()
    )
    return DataLoader2(datapipe=pipe), iseq


def time_testset(name, loader, iseq, fn):
    # Two warmup batches are excluded from the reported full-test wall time.
    warmup = []
    for index, data in enumerate(loader):
        fn(data, iseq)
        if index >= 1:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    users = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for data in loader:
            fn(data, iseq)
            users += len(data[iseq])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    row = {
        "method": name,
        "users": users,
        "elapsed_sec": elapsed,
        "throughput_users_s": users / max(elapsed, 1e-9),
    }
    print(f"{name}: {users} users, {elapsed:.3f}s, {row['throughput_users_s']:.1f} users/s", flush=True)
    return row


def build_tiger(tokenizer):
    cfg = T5Config(
        vocab_size=len(tokenizer), d_model=128, d_kv=64, d_ff=256,
        num_layers=6, num_decoder_layers=6, num_heads=4, dropout_rate=0.1,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )
    model = T5ForConditionalGeneration(cfg)
    checkpoint = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
    state = {key[3:]: value for key, value in checkpoint["model"].items() if key.startswith("t5.")}
    model.load_state_dict(state)
    return model.to(DEVICE).bfloat16().eval()


def build_expert(dataset, tokenizer, converter, code_of_tok, item_codes):
    cfg = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        expert_layers=2, expert_type="mlp", expert_hidden=256,
        expert_score=True, expert_rank=8, expert_negatives=0,
        expert_score_scale=1.0, fast_encoder=True, fast_hidden=256,
        fast_layers=2, fast_max_seq_len=128, distill_encoder=False,
        tau_align=0.1, lam_align=0.2, tau_joint=0.1, lam_joint_score=0.1,
        use_joint=True, prefix_mode="soft", prefix_temperature=1.0,
        flow_steps=1, top_b=-1, use_prior=True, single_expert=False,
        prior_noise=0.05, tau_code=0.1, lam_fm=1.0, lam_code=0.5,
        lam_joint=0.1, lam_prefix=1.0, lam_parallel=1.0,
    )
    model = HiFlowRec(dataset, cfg, tokenizer, converter, code_of_tok, item_codes)
    state = torch.load(EXPERT_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).bfloat16().eval()


def main():
    dataset = build_dataset("Beauty")
    with open(sid_vocab_path("Beauty"), encoding="utf-8") as handle:
        raw_vocab = json.load(handle)
    tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
    converter = SemIDConverter(raw_vocab, tokenizer)
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path("Beauty")), item_count(dataset)
    )

    results = []
    tiger = build_tiger(tokenizer)
    loader, iseq = build_test_loader(dataset, converter)

    def tiger_fn(data, field):
        texts = data[field]
        encoded = tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        ids = encoded["input_ids"].to(DEVICE)
        mask = encoded["attention_mask"].to(DEVICE)
        prefix = tokenizer(
            [SemIDConverter.SID_START_TOKEN] * len(texts),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEVICE)
        tiger.generate(
            input_ids=ids, attention_mask=mask, decoder_input_ids=prefix,
            max_new_tokens=4, num_beams=BEAM, num_return_sequences=BEAM,
            use_cache=True,
        )

    results.append(time_testset("TIGER beam50 bf16", loader, iseq, tiger_fn))
    del tiger
    torch.cuda.empty_cache()

    expert = build_expert(dataset, tokenizer, converter, code_of_tok, item_codes)
    loader, iseq = build_test_loader(dataset, converter)
    results.append(time_testset(
        "π-SID ExpertScore rank8 full-catalog bf16",
        loader,
        iseq,
        lambda data, field: expert.recommend_from_full({expert.ISeq: data[field]}),
    ))

    baseline = results[0]["elapsed_sec"]
    for row in results[1:]:
        row["speedup_vs_tiger"] = baseline / row["elapsed_sec"]
    output = ROOT / "HiFlow-SID/results/speed_expertscore_sidmlp_rules.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
