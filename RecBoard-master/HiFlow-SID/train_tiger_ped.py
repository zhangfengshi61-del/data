"""TIGER-PED: TIGER encoder + parallel expert decoder training.

Reuses a standard TIGER T5 encoder checkpoint and trains a small,
fully-supervised parallel SID path expert. The T5 decoder and beam search
are never used by this model; inference scores the complete catalog in one
pass. No teacher, distillation, candidate retrieval or reranking is used.
"""

from __future__ import annotations

import os
from typing import Dict

import freerec
import torch

from hiflow_lib.common import (
    build_dataset,
    item_count,
    make_tokenizer_converter,
    sid_vocab_path,
)
from hiflow_lib.model_hiflow import HiFlowRec

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
cfg.add_argument("--mtp-checkpoint", type=str, default=None, help="optional compatible model checkpoint")
cfg.add_argument("--tiger-checkpoint", type=str, default=None, help="standard TIGER best.pt; only t5.encoder/shared are loaded")
cfg.add_argument("--expert-score-t5", type=eval, default=True, help="use original TIGER T5 encoder for ExpertScore")
cfg.add_argument("--freeze-backbone", action="store_true", default=False)

cfg.add_argument("--expert-layers", type=int, default=2)
cfg.add_argument(
    "--expert-type",
    type=str,
    default="transformer",
    choices=("transformer", "mlp"),
    help="low-level SID chunk expert; mlp is the speed-oriented π-SID path",
)
cfg.add_argument("--expert-hidden", type=int, default=256)
cfg.add_argument(
    "--expert-score",
    type=eval,
    default=False,
    help="train π-SID ExpertScore directly from interactions without a teacher",
)
cfg.add_argument("--expert-rank", type=int, default=8)
cfg.add_argument("--expert-multiplicative", type=eval, default=False,
                  help="add a vectorized user x prefix interaction inside the SID expert")
cfg.add_argument("--history-attention-pool", type=eval, default=False,
                  help="learn a masked attention pooling residual over T5 encoder states")
cfg.add_argument("--lam-expert-rank", type=float, default=1.0)
cfg.add_argument("--expert-score-scale", type=float, default=1.0)
cfg.add_argument(
    "--expert-negatives", type=int, default=2048,
    help="catalog SIDs sampled per batch for ExpertScore ranking",
)
cfg.add_argument("--hard-prefix-negatives", type=int, default=1024, help="same-prefix catalog negatives per batch for ExpertScore ranking")
cfg.add_argument("--tau-align", type=float, default=0.1)
cfg.add_argument("--lam-align", type=float, default=0.1)
cfg.add_argument("--prior-noise", type=float, default=0.05)
cfg.add_argument("--tau-code", type=float, default=0.1)
cfg.add_argument("--tau-joint", type=float, default=0.1)
cfg.add_argument("--lam-fm", type=float, default=1.0)
cfg.add_argument("--lam-code", type=float, default=0.5)
cfg.add_argument("--lam-joint", type=float, default=0.1)
cfg.add_argument("--lam-prefix", type=float, default=0.5)
cfg.add_argument("--lam-parallel", type=float, default=1.0)
cfg.add_argument(
    "--train-soft-prefix",
    type=eval,
    default=False,
    help="train the low-level expert from the predicted semantic prefix distribution",
)
cfg.add_argument("--prefix-temperature", type=float, default=1.0)
cfg.add_argument(
    "--fast-encoder",
    type=eval,
    default=False,
    help="replace T5 with a global-context position-wise MLP encoder",
)
cfg.add_argument("--fast-hidden", type=int, default=256)
cfg.add_argument("--fast-layers", type=int, default=2)
cfg.add_argument("--fast-max-seq-len", type=int, default=128)
cfg.add_argument(
    "--distill-encoder",
    type=eval,
    default=False,
    help="use the MTP/T5 encoder as a frozen hidden-state teacher during training",
)
cfg.add_argument("--lam-distill", type=float, default=1.0)
cfg.add_argument("--lam-logit-distill", type=float, default=0.0)
cfg.add_argument("--kd-temperature", type=float, default=2.0)

# inference defaults (also overridable for test-time variants)
cfg.add_argument("--flow-steps", type=int, default=1)
cfg.add_argument("--top-b", type=int, default=-1, help="-1: all 256 branches")
cfg.add_argument("--use-prior", type=eval, default=True)
cfg.add_argument("--use-joint", type=eval, default=True)
cfg.add_argument("--lam-joint-score", type=float, default=0.1)
cfg.add_argument(
    "--single-expert",
    type=eval,
    default=False,
    help="π-SID path: one semantic-conditioned suffix expert per user",
)
cfg.add_argument(
    "--prefix-mode",
    type=str,
    default="soft",
    choices=("soft", "argmax"),
    help="prefix representation for the single-expert path",
)

cfg.set_defaults(
    description="HiFlow-SID",
    root="data",
    dataset="Amazon2014Beauty_550_LOU",
    epochs=40,
    batch_size=256,
    optimizer="AdamW",
    lr=3e-4,
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


class CoachForHiFlow(freerec.launcher.Coach):
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
    cfg.expert_score = True
    cfg.fast_encoder = False
    cfg.expert_score_t5 = True

    dataset = build_dataset(cfg.category)
    default_rq_vocab = (
        __import__("pathlib").Path(__file__).resolve().parent
        / f"logs/sid_vocab_rqvae_{cfg.category}.json"
    )
    vocab_file = cfg.sid_vocab_file or (
        str(default_rq_vocab) if default_rq_vocab.exists()
        else str(sid_vocab_path(cfg.category))
    )
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        vocab_file, item_count(dataset)
    )

    model = HiFlowRec(dataset, cfg, tokenizer, converter, code_of_tok, item_codes)

    if cfg.tiger_checkpoint:
        raw = torch.load(cfg.tiger_checkpoint, map_location="cpu", weights_only=False)
        state = raw.get("model", raw) if isinstance(raw, dict) else raw
        enc_state = {}
        for key, value in state.items():
            if key.startswith("t5."):
                key = key[3:]
            if key == "shared.weight" or key.startswith("encoder."):
                enc_state[key] = value
        missing, unexpected = model.backbone.load_state_dict(enc_state, strict=False)
        print(
            f"[TIGER-PED] loaded TIGER encoder checkpoint={cfg.tiger_checkpoint} "
            f"keys={len(enc_state)} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    elif cfg.mtp_checkpoint:
        state = torch.load(cfg.mtp_checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[TIGER-PED] loaded compatible model checkpoint={cfg.mtp_checkpoint} "
            f"missing={len(missing)} unexpected={len(unexpected)}", flush=True
        )
    # Initialize SID code factors from the shared TIGER token embedding.
    # This is a parameter initialization, not decoder-logit distillation.
    if cfg.tiger_checkpoint:
        with torch.no_grad():
            shared = model.backbone.shared.weight
            for level in range(model.num_sid_tokens):
                for code in range(model.num_codewords):
                    token = f"<sid_{level}_{code}>"
                    tid = tokenizer.convert_tokens_to_ids(token)
                    if tid is not None and int(tid) < shared.size(0):
                        model.code_embs[level].weight[code].copy_(shared[int(tid)])
        print("[TIGER-PED] initialized catalog SID embeddings from TIGER shared table", flush=True)
    if cfg.tiger_checkpoint or cfg.mtp_checkpoint:
        if cfg.distill_encoder and model.teacher_backbone is not None:
            teacher_state = {
                key[len("backbone."):]: value
                for key, value in state.items()
                if key.startswith("backbone.")
            }
            missing_teacher, unexpected_teacher = model.teacher_backbone.load_state_dict(
                teacher_state, strict=False
            )
            if missing_teacher or unexpected_teacher:
                raise RuntimeError(
                    "Could not initialize frozen T5 teacher cleanly: "
                    f"missing={missing_teacher}, unexpected={unexpected_teacher}"
                )
            model.teacher_backbone.eval()
            if model.teacher_prefix_head is not None:
                model.teacher_prefix_head.load_state_dict(
                    {
                        "weight": state["prefix_head.weight"],
                        "bias": state["prefix_head.bias"],
                    }
                )
                for index, head in enumerate(model.teacher_suffix_heads):
                    head.load_state_dict(
                        {
                            "weight": state[f"suffix_heads.{index}.weight"],
                            "bias": state[f"suffix_heads.{index}.bias"],
                        }
                    )
                model.teacher_prefix_head.eval()
                model.teacher_suffix_heads.eval()
            print("[HiFlow] loaded frozen MTP/T5 teacher for encoder feature alignment")
        if cfg.freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("[HiFlow] backbone frozen")

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

    coach = CoachForHiFlow(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    print(
        "[TIGER-PED] no-beam parallel expert training: T5 encoder only; "
        "no decoder/teacher/distillation/candidate rerank", flush=True
    )
    coach.fit()


if __name__ == "__main__":
    main()
