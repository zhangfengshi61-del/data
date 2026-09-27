"""Full TEST-set inference wall time (RPG-style: one full epoch on the test set).

TIGER (user's own setup: batch96, beam30, no trie, fp32, candidate decode)
MTP / HiFlow-1 (1 backbone + full-catalog 12,101-item scoring)

All three run over the exact same 22,363 test users on the same GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
sys.path.insert(0, "/data/fszhang/RecBoard-master/SDQ-403")

import freerec  # noqa: E402
from converter import SemIDConverter  # noqa: E402
from transformers import (  # noqa: E402
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

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
HIFLOW_BEST = ROOT / "HiFlow-SID/logs/HiFlow-SID/Amazon2014Beauty_550_LOU/hiflow-s1/best.pt"
DEV = "cuda:0"
BATCH = 96

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


def main():
    ds = build_dataset("Beauty")
    loader, ISeq = build_test_loader(ds)
    results = []

    # ---------------- TIGER (user's own setup)
    tok = T5Tokenizer(vocab=None, extra_ids=0)
    conv = SemIDConverter(_RAW_SID, tok)
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
    t5 = t5.to(DEV).eval()
    kw = {
        "num_beams": 30,
        "num_return_sequences": 30,
        "return_dict_in_generate": True,
        "max_new_tokens": 4,
    }
    n_users = 0
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            texts = data[ISeq]
            enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
            ids, mask = enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)
            prefix = tok(
                [SemIDConverter.SID_START_TOKEN] * len(texts),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(DEV)
            t5.generate(input_ids=ids, attention_mask=mask, decoder_input_ids=prefix, **kw)
            n_users += len(texts)
    torch.cuda.synchronize()
    tiger_t = time.time() - t0
    results.append({"method": "TIGER (their setup)", "users": n_users, "wall_s": tiger_t})
    print(f"TIGER test-set: {n_users} users, {tiger_t:.2f}s", flush=True)
    del t5
    torch.cuda.empty_cache()

    # ---------------- MTP
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path("Beauty")), item_count(ds)
    )
    cfg_mtp = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
    )
    mtp = MTPRec(ds, cfg_mtp, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    mtp.load_state_dict(torch.load(MTP_BEST, map_location="cpu"))
    loader, ISeq = build_test_loader(ds)
    n_users = 0
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            mtp.recommend_from_full({ISeq: data[ISeq]})
            n_users += len(data[ISeq])
    torch.cuda.synchronize()
    mtp_t = time.time() - t0
    results.append({"method": "MTP (full-catalog)", "users": n_users, "wall_s": mtp_t})
    print(f"MTP test-set: {n_users} users, {mtp_t:.2f}s", flush=True)

    # ---------------- HiFlow-1 (full 256 branches)
    cfg_hif = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
        expert_layers=2, prior_noise=0.05, tau_code=0.1, tau_joint=0.1,
        lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix_2=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=True, lam_joint_score=0.1,
    )
    hif = HiFlowRec(ds, cfg_hif, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    ckpt = HIFLOW_BEST if HIFLOW_BEST.exists() else MTP_BEST
    hif.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    print(f"[note] HiFlow weights: {ckpt.name}", flush=True)
    loader, ISeq = build_test_loader(ds)
    n_users = 0
    t0 = time.time()
    with torch.no_grad():
        for data in loader:
            h = hif.encode_history({ISeq: data[ISeq]})
            pl = hif.prefix_head(h)
            hif.score_batch(h, pl, flow_steps=1, top_b=-1)
            n_users += len(data[ISeq])
    torch.cuda.synchronize()
    hif_t = time.time() - t0
    results.append({"method": "HiFlow-1 full-catalog", "users": n_users, "wall_s": hif_t})
    print(f"HiFlow-1 test-set: {n_users} users, {hif_t:.2f}s", flush=True)

    results.append({
        "speedup_vs_tiger": tiger_t / hif_t,
        "speedup_mtp_vs_tiger": tiger_t / mtp_t,
    })
    out = ROOT / "HiFlow-SID/results/speed_testset.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
