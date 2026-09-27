"""Protocol-constrained (trie) full-ranking evaluation of the RQ-TIGER T5.

Evaluates best.pt AND model.pt on valid with constrained beam-30; picks the
checkpoint with better constrained valid NDCG@10, reports its test metrics.
This mirrors the SDQ baseline's best-by-constrained-val procedure.

Usage: python eval_t5_protocol.py <Beauty|Sports|Toys> [gpu]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
sys.path.insert(0, "/data/fszhang/RecBoard-master/SDQ-403")

import freerec  # noqa: E402
from converter import SemIDConverter, prefix_allowed_tokens_fn  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer  # noqa: E402
from transformers.generation.stopping_criteria import StoppingCriteriaList  # noqa: E402

from hiflow_lib.common import build_dataset  # noqa: E402

ROOT = Path("/data/fszhang/RecBoard-master")
CAT = sys.argv[1] if len(sys.argv) > 1 else "Beauty"
DEV = f"cuda:{sys.argv[2]}" if len(sys.argv) > 2 else "cuda:0"
DS = f"Amazon2014{CAT}_550_LOU"
LOG = ROOT / f"SDQ-403/logs/TIGER-T5/{DS}/t5-rq-fast"
RQVOCAB = ROOT / f"HiFlow-SID/logs/sid_vocab_rqvae_{CAT}.json"
BATCH = 64
BEAM = 30

with open(RQVOCAB) as _f:
    _RAW_SID = json.load(_f)
CONV = SemIDConverter(_RAW_SID, T5Tokenizer(vocab=None, extra_ids=0))


def build_model():
    tok = T5Tokenizer(vocab=None, extra_ids=0)
    conv = SemIDConverter(_RAW_SID, tok)
    cfg = T5Config(
        vocab_size=len(tok), d_model=128, d_kv=64, d_ff=256,
        num_layers=6, num_decoder_layers=6, num_heads=4, dropout_rate=0.1,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        decoder_start_token_id=tok.pad_token_id,
    )
    m = T5ForConditionalGeneration(cfg)
    return m, tok, conv


def eval_split(model, tok, conv, ds, split, seen_map):
    from freerec.data.tags import ID, ITEM, SEQUENCE

    ISeq = ds.fields[ITEM, ID].fork(SEQUENCE)
    IUnseen = ds.fields[ITEM, ID].fork(freerec.data.tags.UNSEEN)
    ISeen = ds.fields[ITEM, ID].fork(freerec.data.tags.SEEN)
    pipe = (
        (ds.valid() if split == "valid" else ds.test())
        .ordered_user_ids_source()
        .valid_sampling_("full") if split == "valid" else
        (ds.test().ordered_user_ids_source().test_sampling_("full"))
    )
    pipe = (
        pipe.lprune_(20, modified_fields=(ISeq,))
        .map_(lambda f, items: [SemIDConverter.format(i) for i in items], modified_fields=(ISeq,))
        .map_(lambda f, texts: CONV.encode(texts), modified_fields=(ISeq,))
        .batch_(BATCH)
        .tensor_()
    )
    from torchdata.dataloader2 import DataLoader2

    loader = DataLoader2(datapipe=pipe)
    kwargs = {
        "num_beams": BEAM,
        "num_return_sequences": BEAM,
        "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn(conv),
        "stopping_criteria": StoppingCriteriaList([conv.stopping_criteria(num_items=1)]),
        "max_new_tokens": conv.max_num_sid_tokens + 1,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    n_items = ds.fields[ITEM, ID].count
    y_pred, y_true = [], []
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
            out = model.generate(
                input_ids=ids, attention_mask=mask, decoder_input_ids=prefix, **kwargs
            )
            decoded = conv.batch_decode(tok.batch_decode(out.sequences))
            cand = torch.tensor(
                [items[0] if len(items) == 1 else n_items for items in decoded],
                dtype=torch.long, device=DEV,
            ).view(len(texts), BEAM)
            scores = torch.rand((len(texts), n_items + 1), device=DEV).mul_(1e-3)
            raised = out.sequences_scores.view(len(texts), BEAM)
            raised = raised - raised.min(dim=1, keepdim=True).values + 1.0
            scores.scatter_(1, cand, raised)
            scores = scores[:, :n_items]
            seen = torch.zeros((len(texts), n_items), device=DEV)
            for r, items in enumerate(data[ISeen]):
                if len(items):
                    seen[r, torch.tensor(list(items), device=DEV)] = 1.0
            targets = torch.zeros((len(texts), n_items), device=DEV)
            for r, items in enumerate(data[IUnseen]):
                if len(items):
                    targets[r, torch.tensor(list(items), device=DEV)] = 1.0
            scores[seen.bool()] = -1e23
            y_pred.append(scores)
            y_true.append(targets)
    preds = torch.cat(y_pred)
    truths = torch.cat(y_true)
    from freerec.metrics import hit_rate, normalized_dcg, mean_reciprocal_rank

    res = {}
    for k in (5, 10, 20):
        res[f"Hit@{k}"] = hit_rate(preds, truths, k=k).mean().item()
        res[f"NDCG@{k}"] = normalized_dcg(preds, truths, k=k).mean().item()
        res[f"MRR@{k}"] = mean_reciprocal_rank(preds, truths, k=k).mean().item()
    return res


def main():
    ds = build_dataset(CAT)
    model, tok, conv = build_model()
    model = model.to(DEV).bfloat16().eval()
    results = {}
    for name in ("best", "model"):
        ck = LOG / f"{name}.pt"
        sd = torch.load(ck, map_location="cpu")
        if "t5." in next(iter(sd.keys())):
            sd = {k[3:]: v for k, v in sd.items() if k.startswith("t5.")}
        model.load_state_dict(sd)
        results[name] = eval_split(model, tok, conv, ds, "valid", None)
        print(f"[{name}] valid: {results[name]}", flush=True)
    pick = max(results, key=lambda n: results[n]["NDCG@10"])
    sd = torch.load(LOG / f"{pick}.pt", map_location="cpu")
    if "t5." in next(iter(sd.keys())):
        sd = {k[3:]: v for k, v in sd.items() if k.startswith("t5.")}
    model.load_state_dict(sd)
    test = eval_split(model, tok, conv, ds, "test", None)
    out = {
        "dataset": CAT,
        "pick": pick,
        "valid": {k: results[k] for k in results},
        "test": test,
    }
    dst = ROOT / f"HiFlow-SID/results/t5_rq_protocol_{CAT}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"saved -> {dst}")


if __name__ == "__main__":
    main()
