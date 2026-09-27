"""Real full valid-set wall time for TIGER (user's own setup).

batch=96, plain beam-30 (no trie), fp32, max 4 new tokens.
Iterates all 22,363 valid users exactly once, times generation end-to-end.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

from hiflow_lib.common import build_dataset, sid_vocab_path  # noqa: E402

ROOT = Path("/data/fszhang/RecBoard-master")
SDQ_CKPT = ROOT / "SDQ-403/infos/SDQ-T5-VAE/Amazon2014Beauty_550_LOU/0/checkpoint.tar"
DEV = "cuda:0"
BATCH = 96

with open(sid_vocab_path("Beauty")) as _f:
    _RAW_SID = json.load(_f)
CONV = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


def main():
    tok = T5Tokenizer(vocab=None, extra_ids=0)
    conv = SemIDConverter(_RAW_SID, tok)
    cfg = T5Config(
        vocab_size=len(tok), d_model=128, d_kv=64, d_ff=256,
        num_layers=6, num_decoder_layers=6, num_heads=4, dropout_rate=0.1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )
    m = T5ForConditionalGeneration(cfg)
    ck = torch.load(SDQ_CKPT, map_location="cpu", weights_only=False)
    sd = {k[3:]: v for k, v in ck["model"].items() if k.startswith("t5.")}
    m.load_state_dict(sd)
    m = m.to(DEV).eval()

    from freerec.data.tags import ID, ITEM, SEQUENCE

    ds = build_dataset("Beauty")
    ISeq = ds.fields[ITEM, ID].fork(SEQUENCE)
    pipe = (
        ds.valid()
        .ordered_user_ids_source()
        .valid_sampling_("full")
        .lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONV.encode(texts), modified_fields=(ISeq,))
        .batch_(BATCH)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2

    loader = DataLoader2(datapipe=pipe)

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
            m.generate(input_ids=ids, attention_mask=mask, decoder_input_ids=prefix, **kw)
            n_users += len(texts)
    torch.cuda.synchronize()
    wall = time.time() - t0
    print(f"[full-valid] users={n_users} wall={wall:.2f}s per_user={wall / n_users * 1000:.3f}ms", flush=True)
    out = ROOT / "HiFlow-SID/results/tiger_fullvalid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"users": n_users, "wall_s": wall}) + "\n")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
