"""Phase 1: discrete pre-training / MTP baseline.

One backbone forward predicts ``c1`` (coarse prefix) and ``c2/c3`` (fine
suffix) with independent heads, plus an item-alignment InfoNCE. The trained
checkpoint is reused as the discrete teacher for the phase-2 flow expert and
evaluated on its own as the MTP/RPG-style baseline.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import freerec
import torch

from hiflow_lib.common import (
    build_dataset,
    item_count,
    make_tokenizer_converter,
    sid_vocab_path,
)
from hiflow_lib.model_mtp import MTPRec

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=20, help="maximum item sequence length")
cfg.add_argument("--embedding-dim", type=int, default=128, help="T5 d_model")
cfg.add_argument("--attention-size", type=int, default=64, help="T5 d_kv")
cfg.add_argument("--intermediate-size", type=int, default=64 * 4, help="T5 d_ff")
cfg.add_argument("--num-heads", type=int, default=4, help="number of attention heads")
cfg.add_argument("--num-layers", type=int, default=6, help="number of encoder layers")
cfg.add_argument("--dropout-rate", type=float, default=0.1, help="dropout rate")
cfg.add_argument("--num-codewords", type=int, default=256, help="SID codebook size")
cfg.add_argument("--sid-vocab-file", type=str, default=None)
cfg.add_argument(
    "--category",
    type=str,
    default="Beauty",
    choices=("Beauty", "Sports", "Toys"),
)
cfg.add_argument("--tau-align", type=float, default=0.1)
cfg.add_argument("--lam-prefix", type=float, default=1.0)
cfg.add_argument("--lam-suffix", type=float, default=1.0)
cfg.add_argument("--lam-align", type=float, default=0.1)

cfg.set_defaults(
    description="HiFlow-SID",
    root="data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=100,
    batch_size=512,
    optimizer="AdamW",
    lr=5e-4,
    weight_decay=1e-3,
    seed=2025,
    eval_freq=5,
    ranking="full",
    which4best="NDCG@10",
    monitors=[
        "LOSS", "HitRate@1", "HitRate@5", "HitRate@10", "HitRate@20",
        "NDCG@5", "NDCG@10", "NDCG@20", "MRR@5", "MRR@10", "MRR@20",
    ],
)
cfg.compile()


class CoachForMTP(freerec.launcher.Coach):
    def train_per_epoch(self, epoch: int):
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["rec_loss"]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.monitor(
                loss.item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )


def main():
    dataset = build_dataset(cfg.category)
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        cfg.sid_vocab_file or str(sid_vocab_path(cfg.category)), item_count(dataset)
    )

    model = MTPRec(dataset, cfg, tokenizer, converter, code_of_tok, item_codes)

    trainpipe = (
        dataset.train()
        .shuffled_roll_seqs_source(
            minlen=2, maxlen=cfg.maxlen, keep_at_least_itself=True
        )
        .seq_train_yielding_pos_(start_idx_for_target=-1, end_idx_for_input=-1)
        .map_(model.format_item_ids, modified_fields=(model.ISeq, model.IPos))
        .map_(model.encode_item_text, modified_fields=(model.ISeq, model.IPos))
        .batch_(cfg.batch_size)
        .tensor_()
    )
    validpipe = (
        dataset.valid()
        .ordered_user_ids_source()
        .valid_sampling_(cfg.ranking)
        .lprune_(cfg.maxlen, modified_fields=(model.ISeq,))
        .map_(model.format_item_ids, modified_fields=(model.ISeq,))
        .map_(model.encode_item_text, modified_fields=(model.ISeq,))
        .batch_(64)
        .tensor_()
    )
    testpipe = (
        dataset.test()
        .ordered_user_ids_source()
        .test_sampling_(cfg.ranking)
        .lprune_(cfg.maxlen, modified_fields=(model.ISeq,))
        .map_(model.format_item_ids, modified_fields=(model.ISeq,))
        .map_(model.encode_item_text, modified_fields=(model.ISeq,))
        .batch_(64)
        .tensor_()
    )

    coach = CoachForMTP(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()


if __name__ == "__main__":
    main()
