"""Break down π-SID-fast latency into tokenization, encoder, and catalogue score."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")
import bench_speed as base  # noqa: E402
from hiflow_lib.common import build_dataset, item_count, make_tokenizer_converter, sid_vocab_path, tokenize_batch  # noqa: E402
from hiflow_lib.model_hiflow import HiFlowRec  # noqa: E402


def main():
    dataset = build_dataset(base.DATASET)
    tok, conv, cot, item_codes = make_tokenizer_converter(str(sid_vocab_path(base.DATASET)), item_count(dataset))
    batches, ISeq = base.sample_users(dataset, base.NUM_USERS, base.BATCH)
    cfg = SimpleNamespace(
        embedding_dim=128, attention_size=64, intermediate_size=256, num_layers=6,
        num_heads=4, dropout_rate=0.1, num_codewords=256, expert_layers=1,
        expert_type="mlp", expert_hidden=64, fast_encoder=True, fast_hidden=64,
        fast_layers=1, fast_max_seq_len=128, prior_noise=0.05, tau_code=0.1,
        tau_joint=0.1, lam_fm=1.0, lam_code=0.5, lam_joint=0.1, lam_prefix=0.5,
        flow_steps=1, top_b=-1, use_prior=True, use_joint=False,
        lam_joint_score=0.1, single_expert=True, prefix_mode="argmax",
        prefix_temperature=0.7, train_soft_prefix=True,
    )
    model = HiFlowRec(dataset, cfg, tok, conv, cot, item_codes).to(base.DEV).eval()
    state = torch.load(base.MTP_BEST, map_location="cpu")
    model.load_state_dict(state, strict=False)
    if "backbone.shared.weight" in state:
        with torch.no_grad(): model.backbone.embedding.weight.copy_(state["backbone.shared.weight"])
    texts = list(batches[0][ISeq])
    # Warmup
    for _ in range(5): model.recommend_from_full({ISeq: texts})
    torch.cuda.synchronize(); rounds = 20
    def timed(fn):
        vals=[]
        for _ in range(rounds):
            torch.cuda.synchronize(); t=time.time(); fn(); torch.cuda.synchronize(); vals.append((time.time()-t)*1000/base.BATCH)
        vals.sort(); return vals[len(vals)//2]
    total = timed(lambda: model.recommend_from_full({ISeq: texts}))
    ctx = timed(lambda: tokenize_batch(model.tokenizer, next(model.parameters()).device, texts))
    fast_ctx = timed(lambda: model._tokenize_fast(texts, next(model.parameters()).device))
    encoded = tokenize_batch(model.tokenizer, next(model.parameters()).device, texts)
    enc = timed(lambda: model.backbone(encoded["input_ids"], encoded["attention_mask"]))
    h = model.backbone(encoded["input_ids"], encoded["attention_mask"])
    h = (h * encoded["attention_mask"].unsqueeze(-1)).sum(1) / encoded["attention_mask"].sum(1,keepdim=True).clamp_min(1)
    pl = model.prefix_head(h)
    score = timed(lambda: model.score_batch_single_expert(h, pl))
    print({"total_ms_user": total, "t5_tokenize_ms_user": ctx, "fast_tokenize_ms_user": fast_ctx, "encoder_ms_user": enc, "score_ms_user": score})


if __name__ == "__main__": main()
