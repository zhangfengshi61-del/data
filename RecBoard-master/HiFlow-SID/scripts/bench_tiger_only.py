"""TIGER-only speed reference, repeated runs to check stability.

SID-MLP rules: batch32 / beam50 / native bf16 / TF32 off / test set /
elapsed + throughput. Same checkpoint used by every method comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
sys.path.insert(0, "/data/fszhang/RecBoard-master/SDQ-403")

import freerec  # noqa: E402
from converter import SemIDConverter  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer  # noqa: E402

from hiflow_lib.common import build_dataset, sid_vocab_path  # noqa: E402

ROOT = Path("/data/fszhang/RecBoard-master")
DEV = "cuda:0"
BATCH = 32
BEAM = 50
REPEATS = 2

CAT = sys.argv[1] if len(sys.argv) > 1 else "Beauty"
DS = f"Amazon2014{CAT}_550_LOU"
SDQ_CKPT = ROOT / f"SDQ-403/infos/SDQ-T5-VAE/{DS}/0/checkpoint.tar"

with open(sid_vocab_path(CAT)) as _f:
    _RAW_SID = json.load(_f)
CONV = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


def main():
    ds = build_dataset(CAT)
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
    t5 = t5.to(DEV).bfloat16().eval()

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

    times = []
    for rep in range(REPEATS):
        loader = DataLoader2(datapipe=pipe)
        n = 0
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
                t5.generate(
                    input_ids=ids, attention_mask=mask, decoder_input_ids=prefix,
                    max_new_tokens=4, num_beams=BEAM, num_return_sequences=BEAM,
                    use_cache=True,
                )
                n += len(texts)
        torch.cuda.synchronize()
        wall = time.time() - t0
        times.append(wall)
        print(f"run{rep + 1}: {n} users, {wall:.2f}s, {n / wall:.1f} users/s", flush=True)
    print(f"spread: {max(times) - min(times):.2f}s ({100 * (max(times) - min(times)) / min(times):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
