"""Latency benchmark: TIGER-beam-30 vs MTP vs HiFlow-SID-OneStep.

Loads the trained SDQ-T5-VAE checkpoint (TIGER-beam baseline), the MTP
checkpoint and a MTP-initialized HiFlow model, then measures end-to-end
next-item recommendation latency on the same test users.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
sys.path.insert(0, "/data/fszhang/RecBoard-master/SDQ-403")

import freerec  # noqa: E402
from converter import SemIDConverter, prefix_allowed_tokens_fn  # noqa: E402
from transformers import (  # noqa: E402
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from transformers.generation.stopping_criteria import StoppingCriteriaList  # noqa: E402

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

NUM_USERS = 400
BATCH = 16
DEV = "cuda:0"

import json as _json

with open(sid_vocab_path("Beauty")) as _f:
    _RAW_SID = _json.load(_f)
CONVERTER = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


class TigerT5(torch.nn.Module):
    """Minimal TIGER-style T5 wrapper for latency measurement."""

    def __init__(self, tokenizer, converter, d=128, d_kv=64, d_ff=256, layers=6, heads=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.converter = converter
        cfg = T5Config(
            vocab_size=len(tokenizer),
            d_model=d,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=layers,
            num_decoder_layers=layers,
            num_heads=heads,
            dropout_rate=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
        )
        self.t5 = T5ForConditionalGeneration(cfg)
        self.generate_kwargs = {
            "num_beams": 30,
            "num_return_sequences": 30,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn(self.converter),
            "stopping_criteria": StoppingCriteriaList(
                [self.converter.stopping_criteria(num_items=1)]
            ),
            "max_new_tokens": self.converter.max_num_sid_tokens + 1,
            "return_dict_in_generate": True,
            "output_scores": True,
        }

    def tokenize(self, texts):
        enc = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)

    @torch.no_grad()
    def recommend(self, texts):
        ids, mask = self.tokenize(texts)
        prefix = self.tokenizer(
            [SemIDConverter.SID_START_TOKEN] * len(texts),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEV)
        self.t5.generate(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=prefix,
            **self.generate_kwargs,
        )


def load_tiger():
    ck = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    with open(sid_vocab_path("Beauty")) as f:
        import json

        raw = json.load(f)
    tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
    converter = SemIDConverter(raw, tokenizer)
    model = TigerT5(tokenizer, converter)
    model.load_state_dict(sd)
    return model.to(DEV).eval()


def sample_test_users(dataset, num_users, batch):
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        dataset.test()
        .ordered_user_ids_source()
        .test_sampling_("full")
        .lprune_(50, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONVERTER.encode(texts), modified_fields=(ISeq,))
        .batch_(batch)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService

    loader = DataLoader2(datapipe=pipe, reading_service=MultiProcessingReadingService(2))
    batches = []
    for data in loader:
        batches.append(data)
        if (len(batches)) * batch >= num_users:
            break
    return batches


def fmt(secs):
    return ", ".join(f"{s:.2f}s" for s in secs)


def main():
    dataset = build_dataset("Beauty")
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path("Beauty")), item_count(dataset)
    )

    print("[bench] collecting test batches ...")
    batches = sample_test_users(dataset, NUM_USERS, BATCH)
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)
    print(f"[bench] {len(batches)} batches x {BATCH}")

    # ---------------- TIGER-beam-30
    print("[bench] TIGER-beam-30 ...")
    tiger = load_tiger()
    tiger_times = []
    with torch.no_grad():
        for data in batches:
            torch.cuda.synchronize()
            t0 = time.time()
            tiger.recommend([s for s in data[ISeq]])
            torch.cuda.synchronize()
            tiger_times.append(time.time() - t0)
    print(f"[bench] TIGER-beam-30 per-user: batch_time {fmt(tiger_times)}")

    # ---------------- MTP
    print("[bench] MTP ...")
    cfg_mtp = SimpleNamespace(
        maxlen=50, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
    )
    mtp = MTPRec(dataset, cfg_mtp, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    mtp.load_state_dict(torch.load(MTP_BEST, map_location="cpu"))
    ISeq = mtp.ISeq
    mtp_times = []
    with torch.no_grad():
        for data in batches:
            texts = data[ISeq]
            torch.cuda.synchronize()
            t0 = time.time()
            mtp.recommend_from_full({ISeq: texts})
            torch.cuda.synchronize()
            mtp_times.append(time.time() - t0)
    print(f"[bench] MTP per-user: batch_time {fmt(mtp_times)}")

    # ---------------- HiFlow variants
    print("[bench] HiFlow ...")
    cfg_hif = SimpleNamespace(
        maxlen=50, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
        expert_layers=2, prior_noise=0.05, tau_code=0.1, tau_joint=0.1,
        lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix_2=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=True, lam_joint_score=0.1,
    )
    hif = HiFlowRec(dataset, cfg_hif, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    hif.load_state_dict(torch.load(MTP_BEST, map_location="cpu"), strict=False)

    for name, steps, top_b in [
        ("HiFlow-1 full(256)", 1, -1),
        ("HiFlow-1 top16", 1, 16),
        ("HiFlow-4 full(256)", 4, -1),
    ]:
        times = []
        with torch.no_grad():
            for data in batches:
                texts = data[ISeq]
                h = hif.encode_history({ISeq: texts})
                pl = hif.prefix_head(h)
                torch.cuda.synchronize()
                t0 = time.time()
                hif.score_batch(h, pl, flow_steps=steps, top_b=top_b)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
        print(f"[bench] {name} per-user: batch_time {fmt(times)}")

    print("[bench] done")


if __name__ == "__main__":
    main()
