"""Decompose TIGER latency: batch size x trie constraint.

Same 480 test users, same beam-30 model. Four configs:
  - batch 16 / no trie
  - batch 96 / no trie
  - batch 16 / trie
  - batch 96 / trie
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

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

ROOT = Path("/data/fszhang/RecBoard-master")
SDQ_CKPT = ROOT / "SDQ-403/infos/SDQ-T5-VAE/Amazon2014Beauty_550_LOU/0/checkpoint.tar"
NUM_USERS = 480
DEV = "cuda:0"
WARMUP = 2
ROUNDS = 3

with open(sid_vocab_path("Beauty")) as _f:
    _RAW_SID = json.load(_f)
CONV = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


class TigerT5(torch.nn.Module):
    def __init__(self, tokenizer, converter, constrained=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.converter = converter
        self.constrained = constrained
        cfg = T5Config(
            vocab_size=len(tokenizer), d_model=128, d_kv=64, d_ff=256,
            num_layers=6, num_decoder_layers=6, num_heads=4, dropout_rate=0.1,
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
            self.kwargs["max_new_tokens"] = self.converter.max_num_sid_tokens + 1
        else:
            self.kwargs["max_new_tokens"] = 4

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
            input_ids=ids, attention_mask=mask,
            decoder_input_ids=prefix, **self.kwargs,
        )


def collect(batch):
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = build_dataset("Beauty").fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        build_dataset("Beauty").test()
        .ordered_user_ids_source()
        .test_sampling_("full")
        .lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONV.encode(texts), modified_fields=(ISeq,))
        .batch_(batch)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2

    loader = DataLoader2(datapipe=pipe)
    batches = []
    for data in loader:
        batches.append(data)
        if len(batches) * batch >= NUM_USERS:
            break
    return batches, ISeq


def main():
    results = []
    for batch in (16, 96):
        batches, ISeq = collect(batch)
        for constrained in (False, True):
            tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
            converter = SemIDConverter(_RAW_SID, tokenizer)
            model = TigerT5(tokenizer, converter, constrained=constrained)
            ck = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model"])
            model = model.to(DEV).eval()

            times = []
            for _ in range(WARMUP):
                model.recommend([s for s in batches[0][ISeq]])
            for _ in range(ROUNDS):
                for data in batches:
                    torch.cuda.synchronize()
                    t0 = time.time()
                    model.recommend([s for s in data[ISeq]])
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
            per_user = [t / batch for t in times]
            results.append({
                "config": f"batch{batch}" + ("+trie" if constrained else ""),
                "p50_ms": statistics.median(per_user) * 1000,
                "batch_ms_median": statistics.median(times) * 1000,
            })
            print(results[-1], flush=True)
            del model
            torch.cuda.empty_cache()

    out = ROOT / "HiFlow-SID/results/tiger_decompose.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
