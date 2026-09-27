"""Speed benchmark under SID-MLP's rules, on our own data (Beauty 2014).

SID-MLP hardware profile (README + configs/infer.yaml + genrec TIGER model):
  - batch_size = 32
  - TIGER: HF generate, KV cache, num_beams=50, num_return_sequences=50,
    max_new_tokens = n_digit + 1 (3-digit SID -> 4), plain beam (no trie)
  - native bf16, TF32 disabled
  - split = test
  - metric: elapsed_sec over the full test set + throughput = users / elapsed

Our models (MTP / HiFlow-1 / pi-SID) run full-catalog (12,101-item) scoring
under the same rules (batch 32, bf16, TF32 off, test split).
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
from hiflow_lib.model_mtp import MTPRec  # noqa: E402

ROOT = Path("/data/fszhang/RecBoard-master")
SDQ_CKPT = ROOT / "SDQ-403/infos/SDQ-T5-VAE/Amazon2014Beauty_550_LOU/0/checkpoint.tar"
MTP_BEST = ROOT / "HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/mtp/best.pt"
DEV = "cuda:0"
BATCH = 32
BEAM = 50

with open(sid_vocab_path("Beauty")) as _f:
    _RAW_SID = json.load(_f)
CONV = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


def build_test_loader(ds):
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = ds.fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        ds.test()
        .ordered_user_ids_source()
        .test_sampling_("full")
        .lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONV.encode(texts), modified_fields=(ISeq,))
        .batch_(BATCH)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2

    return DataLoader2(datapipe=pipe), ISeq


def time_testset(name, loader, ISeq, fn):
    n_users = 0
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            fn(data, ISeq)
            n_users += len(data[ISeq])
    torch.cuda.synchronize()
    wall = time.time() - t0
    row = {
        "method": name,
        "users": n_users,
        "elapsed_sec": wall,
        "throughput_users_s": n_users / max(wall, 1e-6),
    }
    print(f"{name}: {n_users} users, {wall:.2f}s, {n_users / wall:.1f} users/s", flush=True)
    return row


def main():
    ds = build_dataset("Beauty")
    results = []
    tok = T5Tokenizer(vocab=None, extra_ids=0)
    conv = SemIDConverter(_RAW_SID, tok)

    # ---------------- TIGER under SID-MLP rules
    cfg = T5Config(
        vocab_size=len(tok), d_model=128, d_kv=64, d_ff=256,
        num_layers=6, num_decoder_layers=6, num_heads=4, dropout_rate=0.1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )
    t5 = T5ForConditionalGeneration(cfg)
    ck = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
    sd = {k[3:]: v for k, v in ck["model"].items() if k.startswith("t5.")}
    t5.load_state_dict(sd)
    t5 = t5.to(DEV).bfloat16().eval()

    loader, ISeq = build_test_loader(ds)

    def tiger_fn(data, iseq):
        texts = data[iseq]
        enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
        prefix = tok(
            [SemIDConverter.SID_START_TOKEN] * len(texts),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEV)
        t5.generate(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=prefix,
            max_new_tokens=4,
            num_beams=BEAM,
            num_return_sequences=BEAM,
            use_cache=True,
        )

    results.append(time_testset(f"TIGER (SID-MLP rules, beam{BEAM}, batch{BATCH}, bf16)", loader, ISeq, tiger_fn))
    del t5
    torch.cuda.empty_cache()

    # ---------------- shared model components
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path("Beauty")), item_count(ds)
    )

    cfg_mtp = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
    )
    mtp = MTPRec(ds, cfg_mtp, tokenizer, converter, code_of_tok, item_codes)
    mtp.load_state_dict(torch.load(MTP_BEST, map_location="cpu"))
    mtp = mtp.to(DEV).bfloat16().eval()

    loader, ISeq = build_test_loader(ds)
    results.append(time_testset(
        f"MTP (SID-MLP rules, batch{BATCH}, bf16, full-catalog)",
        loader, ISeq, lambda d, s: mtp.recommend_from_full({s: d[s]}),
    ))

    cfg_hif = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
        expert_layers=2, prior_noise=0.05, tau_code=0.1, tau_joint=0.1,
        lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix_2=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=True, lam_joint_score=0.1,
    )
    hif = HiFlowRec(ds, cfg_hif, tokenizer, converter, code_of_tok, item_codes)
    hif.load_state_dict(torch.load(MTP_BEST, map_location="cpu"), strict=False)
    hif = hif.to(DEV).bfloat16().eval()

    loader, ISeq = build_test_loader(ds)
    results.append(time_testset(
        f"HiFlow-1 (SID-MLP rules, batch{BATCH}, bf16, full-catalog)",
        loader, ISeq, lambda d, s: hif.recommend_from_full({s: d[s]}),
    ))

    cfg_pi = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        expert_layers=1, expert_type="mlp", expert_hidden=64,
        fast_encoder=True, fast_hidden=64, fast_layers=1, fast_max_seq_len=128,
        prior_noise=0.05, tau_code=0.1, tau_joint=0.1,
        lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=False,
        lam_joint_score=0.1, single_expert=True, prefix_mode="argmax",
        prefix_temperature=0.7, train_soft_prefix=True,
    )
    pi = HiFlowRec(ds, cfg_pi, tokenizer, converter, code_of_tok, item_codes)
    state = torch.load(MTP_BEST, map_location="cpu")
    pi.load_state_dict(state, strict=False)
    if "backbone.shared.weight" in state:
        with torch.no_grad():
            pi.backbone.embedding.weight.copy_(state["backbone.shared.weight"])
    pi = pi.to(DEV).bfloat16().eval()

    loader, ISeq = build_test_loader(ds)
    results.append(time_testset(
        f"pi-SID (SID-MLP rules, batch{BATCH}, bf16, full-catalog)",
        loader, ISeq, lambda d, s: pi.recommend_from_full({s: d[s]}),
    ))

    tiger_t = results[0]["elapsed_sec"]
    for row in results[1:]:
        row["speedup_vs_tiger"] = tiger_t / row["elapsed_sec"]
    out = ROOT / "HiFlow-SID/results/speed_sidmlp_rules.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
