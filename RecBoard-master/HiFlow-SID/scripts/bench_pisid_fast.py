"""Strict end-to-end speed check for the π-SID fast path.

The comparison uses the same TIGER setup as ``bench_speed.py`` (batch 96,
plain beam-30, fp32) and includes tokenization, user encoding and full
catalogue scoring.  It deliberately does not use the slower batch-16 Trie
variant, so an inflated speedup cannot hide an underperforming implementation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/data/fszhang/RecBoard-master/HiFlow-SID")

import bench_speed as base  # noqa: E402
import freerec  # noqa: E402
from hiflow_lib.common import (  # noqa: E402
    build_dataset,
    item_count,
    make_tokenizer_converter,
    sid_vocab_path,
)
from hiflow_lib.model_hiflow import HiFlowRec  # noqa: E402

PISID_BEST = Path(
    "/data/fszhang/RecBoard-master/HiFlow-SID/logs/HiFlow-SID/"
    "Amazon2014Beauty_550_LOU/pisid-distill-v1/best.pt"
)


def main() -> None:
    dataset = build_dataset(base.DATASET)
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        str(sid_vocab_path(base.DATASET)), item_count(dataset)
    )
    batches, ISeq = base.sample_users(dataset, base.NUM_USERS, base.BATCH)

    results = []
    tiger = base.load_tiger(constrained=False)
    results.append(
        base.bench(
            "TIGER-beam30 (strict reference, fp32, batch96)",
            lambda d: tiger.recommend([s for s in d[ISeq]], autocast=False),
            batches,
        )
    )

    cfg = SimpleNamespace(
        maxlen=20,
        embedding_dim=128,
        attention_size=64,
        intermediate_size=256,
        num_layers=6,
        num_heads=4,
        dropout_rate=0.1,
        num_codewords=256,
        expert_layers=1,
        expert_type="mlp",
        expert_hidden=128,
        fast_encoder=True,
        fast_hidden=128,
        fast_layers=2,
        fast_max_seq_len=128,
        prior_noise=0.05,
        tau_code=0.1,
        tau_joint=0.1,
        lam_fm=1.0,
        lam_code=0.5,
        lam_joint=0.1,
        lam_prefix=0.5,
        flow_steps=1,
        top_b=-1,
        use_prior=True,
        use_joint=True,
        lam_joint_score=0.1,
        single_expert=True,
        prefix_mode="soft",
        prefix_temperature=0.7,
        train_soft_prefix=True,
    )
    model = HiFlowRec(dataset, cfg, tokenizer, converter, code_of_tok, item_codes)
    checkpoint = PISID_BEST if PISID_BEST.exists() else base.MTP_BEST
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    if (
        "backbone.shared.weight" in state
        and state["backbone.shared.weight"].shape == model.backbone.embedding.weight.shape
    ):
        with torch.no_grad():
            model.backbone.embedding.weight.copy_(state["backbone.shared.weight"])
    model = model.to(base.DEV).eval()

    def run_fast(data):
        return model.recommend_from_full({ISeq: data[ISeq]})

    results.append(base.bench("π-SID-fast (one chunk expert)", run_fast, batches))

    for row in results:
        row["speedup_vs_tiger"] = results[0]["p50_ms"] / row["p50_ms"]
    out = Path("/data/fszhang/RecBoard-master/HiFlow-SID/results/pisid_fast_speed.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
