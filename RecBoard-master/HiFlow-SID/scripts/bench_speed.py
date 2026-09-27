"""Full speed benchmark: TIGER (user's own setup) vs MTP vs HiFlow-SID.

TIGER (user's own protocol, tiger_reference/model/main.py):
  - T5ForConditionalGeneration, beam=30, num_return_sequences=30
  - plain beam search (no trie constraints), ~4 new tokens
  - eval batch `infer_size=96`, fp32, candidate-only decode
  - history: <=20 items, flattened to token ids

Compared variants (end-to-end next-item recommendation on the same users):
  - MTP: 1 backbone forward + full-catalog (12,101-item) vectorized scoring
  - HiFlow-1 full: 256 prefix branches, 1 Euler step, full-catalog scoring
  - HiFlow-1 top16: Top-16 prefix pruning, 1 step
  - HiFlow-2 / HiFlow-4 full: 2 / 4 Euler steps

Outputs: per-user P50/P95 latency, throughput, extrapolated full-valid-set
eval time (22,363 users), and a real full-pass timing for MTP/HiFlow.
"""

from __future__ import annotations

import json
import statistics
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

DATASET = "Beauty"
NUM_USERS = 480
BATCH = 96  # user's own infer_size
DEV = "cuda:0"
WARMUP = 2
ROUNDS = 3
NUM_VALID_USERS = 22363

with open(sid_vocab_path(DATASET)) as _f:
    _RAW_SID = json.load(_f)
CONVERTER = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


class TigerT5(torch.nn.Module):
    """TIGER-style T5 (same architecture as tiger_reference/model/main.py)."""

    def __init__(self, tokenizer, converter, constrained=False, d=128, d_kv=64, d_ff=256, layers=6, heads=4):
        super().__init__()
        self.tokenizer = tokenizer
        self.converter = converter
        self.constrained = constrained
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
        self.kwargs = {
            "num_beams": 30,
            "num_return_sequences": 30,
            "return_dict_in_generate": True,
        }
        if constrained:
            self.kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn(self.converter)
            self.kwargs["stopping_criteria"] = StoppingCriteriaList(
                [self.converter.stopping_criteria(num_items=1)]
            )
        else:
            self.kwargs["max_new_tokens"] = 4

    def tokenize(self, texts):
        enc = self.tokenizer(texts, add_special_tokens=False, padding=True, return_tensors="pt")
        return enc["input_ids"].to(DEV), enc["attention_mask"].to(DEV)

    @torch.no_grad()
    def recommend(self, texts, autocast=False):
        ids, mask = self.tokenize(texts)
        prefix = self.tokenizer(
            [SemIDConverter.SID_START_TOKEN] * len(texts),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(DEV)
        if self.constrained:
            self.kwargs["max_new_tokens"] = self.converter.max_num_sid_tokens + 1
        if autocast:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                self.t5.generate(
                    input_ids=ids, attention_mask=mask,
                    decoder_input_ids=prefix, **self.kwargs,
                )
        else:
            self.t5.generate(
                input_ids=ids, attention_mask=mask,
                decoder_input_ids=prefix, **self.kwargs,
            )


def load_tiger(constrained=False):
    ck = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
    sd = ck["model"]
    tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
    converter = SemIDConverter(_RAW_SID, tokenizer)
    model = TigerT5(tokenizer, converter, constrained=constrained)
    model.load_state_dict(sd)
    return model.to(DEV).eval()


def sample_users(dataset, num_users, batch):
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = dataset.fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        dataset.test()
        .ordered_user_ids_source()
        .test_sampling_("full")
        .lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONVERTER.encode(texts), modified_fields=(ISeq,))
        .batch_(batch)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2

    loader = DataLoader2(datapipe=pipe)
    batches = []
    for data in loader:
        batches.append(data)
        if len(batches) * batch >= num_users:
            break
    return batches, ISeq


def bench(name, fn, batches, rounds=ROUNDS):
    times = []
    for _ in range(WARMUP):
        fn(batches[0])
    for _ in range(rounds):
        for data in batches:
            torch.cuda.synchronize()
            t0 = time.time()
            fn(data)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    per_user = [t / BATCH for t in times]
    p50 = statistics.median(per_user)
    p95 = sorted(per_user)[int(len(per_user) * 0.95) - 1]
    tp = BATCH / statistics.median(times)
    full_valid_est = p50 * NUM_VALID_USERS
    return {
        "name": name,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "throughput_users_s": tp,
        "full_valid_est_s": full_valid_est,
    }


def main():
    dataset = build_dataset(DATASET)
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path(DATASET)), item_count(dataset)
    )
    print(f"[bench] {NUM_USERS} users, batch {BATCH}, rounds {ROUNDS}", flush=True)
    batches, ISeq = sample_users(dataset, NUM_USERS, BATCH)
    print(f"[bench] {len(batches)} batches collected", flush=True)

    results = []

    # ---- TIGER: user's own setup (plain beam, batch 96, fp32)
    tiger = load_tiger(constrained=False)
    fn = lambda data: tiger.recommend([s for s in data[ISeq]], autocast=False)
    results.append(bench("TIGER-beam30 (their setup, fp32, batch96)", fn, batches))

    # ---- TIGER: bf16
    fn = lambda data: tiger.recommend([s for s in data[ISeq]], autocast=True)
    results.append(bench("TIGER-beam30 (their setup, bf16, batch96)", fn, batches))

    # ---- TIGER: protocol full-ranking setup (trie + batch 16)
    batches16, _ = sample_users(dataset, NUM_USERS, 16)
    tiger_c = load_tiger(constrained=True)
    fn = lambda data: tiger_c.recommend([s for s in data[ISeq]], autocast=True)
    results.append(bench("TIGER-beam30 (protocol, trie, bf16, batch16)", fn, batches16))
    del tiger, tiger_c
    torch.cuda.empty_cache()

    # ---- MTP
    cfg_mtp = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
    )
    mtp = MTPRec(dataset, cfg_mtp, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    mtp.load_state_dict(torch.load(MTP_BEST, map_location="cpu"))
    fn = lambda data: mtp.recommend_from_full({ISeq: data[ISeq]})
    results.append(bench("MTP (1 forward + full-catalog scoring)", fn, batches))

    # ---- HiFlow
    cfg_hif = SimpleNamespace(
        maxlen=20, embedding_dim=128, attention_size=64, intermediate_size=256,
        num_layers=6, num_heads=4, dropout_rate=0.1, num_codewords=256,
        tau_align=0.1, lam_prefix=1.0, lam_suffix=1.0, lam_align=0.1,
        expert_layers=2, prior_noise=0.05, tau_code=0.1, tau_joint=0.1,
        lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix_2=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=True, lam_joint_score=0.1,
    )
    hif = HiFlowRec(dataset, cfg_hif, tokenizer, converter, code_of_tok, item_codes).to(DEV).eval()
    hif.load_state_dict(torch.load(MTP_BEST, map_location="cpu"), strict=False)

    def hif_fn(data, steps, top_b):
        h = hif.encode_history({ISeq: data[ISeq]})
        pl = hif.prefix_head(h)
        return hif.score_batch(h, pl, flow_steps=steps, top_b=top_b)

    results.append(bench("HiFlow-1 full (256 branches, 1 step)", lambda d: hif_fn(d, 1, -1), batches))
    results.append(bench("HiFlow-1 top16 (prefix pruning)", lambda d: hif_fn(d, 1, 16), batches))
    results.append(bench("HiFlow-2 full (256 branches, 2 steps)", lambda d: hif_fn(d, 2, -1), batches))
    results.append(bench("HiFlow-4 full (256 branches, 4 steps)", lambda d: hif_fn(d, 4, -1), batches))

    # save partial results before the slow full-pass section
    out = ROOT / "HiFlow-SID/results/speed_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)

    # ---- real full-valid-pass timing for MTP / HiFlow (22,363 users)
    from torchdata.dataloader2 import DataLoader2

    vpipe = (
        dataset.valid()
        .ordered_user_ids_source()
        .valid_sampling_("full")
        .lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONVERTER.encode(texts), modified_fields=(ISeq,))
        .batch_(BATCH)
        .tensor_()
    )
    vloader = DataLoader2(datapipe=vpipe)
    t0 = time.time()
    with torch.no_grad():
        for data in vloader:
            mtp.recommend_from_full({ISeq: data[ISeq]})
    torch.cuda.synchronize()
    results.append({"name": "MTP: real full valid-set eval (22,363 users)", "wall_s": time.time() - t0})

    t0 = time.time()
    with torch.no_grad():
        for data in vloader:
            h = hif.encode_history({ISeq: data[ISeq]})
            pl = hif.prefix_head(h)
            hif.score_batch(h, pl, flow_steps=1, top_b=-1)
    torch.cuda.synchronize()
    results.append({"name": "HiFlow-1 full: real full valid-set eval (22,363 users)", "wall_s": time.time() - t0})

    print(json.dumps(results, indent=2, ensure_ascii=False))
    out = ROOT / "HiFlow-SID/results/speed_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"[bench] saved -> {out}")


if __name__ == "__main__":
    main()
