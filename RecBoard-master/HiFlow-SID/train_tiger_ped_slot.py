"""Train MI-SID on the RecBoard full-ranking protocol.

This script is intentionally independent of train_hiflow.py:
it never accepts or loads an MTP/TIGER checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import freerec
import torch

from hiflow_lib.common import (
    build_dataset,
    item_count,
    make_tokenizer_converter,
    sid_vocab_path,
)
from hiflow_lib.model_pate_misid import MISIDRec

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=20)
cfg.add_argument("--embedding-dim", type=int, default=128)
cfg.add_argument("--attention-size", type=int, default=64)
cfg.add_argument("--intermediate-size", type=int, default=256)
cfg.add_argument("--dropout-rate", type=float, default=0.1)
cfg.add_argument("--t5-layers", type=int, default=6)
cfg.add_argument("--t5-pooled", action="store_true", default=False)
cfg.add_argument("--t5-prefix", action="store_true", default=False)
cfg.add_argument("--t5-prefix-width", type=int, default=1)
cfg.add_argument("--sid-residual", action="store_true", default=False)
cfg.add_argument("--sid-residual-weight", type=float, default=1.0)
cfg.add_argument("--tiger-checkpoint", type=str, default=None)
cfg.add_argument("--freeze-backbone", action="store_true", default=False)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--num-slots", type=int, default=4)
cfg.add_argument("--fast-hidden", type=int, default=256)
cfg.add_argument("--fast-layers", type=int, default=2)
cfg.add_argument("--fast-max-seq-len", type=int, default=128)
cfg.add_argument("--fast-num-positions", type=int, default=6)
cfg.add_argument("--encoder-type", type=str, default="fast", choices=("fast", "transformer", "t5"))
cfg.add_argument("--encoder-heads", type=int, default=4)
cfg.add_argument("--expert-rank", type=int, default=16)
cfg.add_argument("--expert-mixtures", type=int, default=2)
cfg.add_argument("--residual-rank", type=int, default=0)
cfg.add_argument("--hierarchical-rank", type=int, default=0)
cfg.add_argument("--freeze-base", type=int, default=0)
cfg.add_argument("--freeze-recent-transition", type=int, default=0)
cfg.add_argument("--hierarchical-scale-init", type=float, default=-5.0)
cfg.add_argument("--semantic-score-weight", type=float, default=0.0)
cfg.add_argument("--recent-score-weight", type=float, default=0.0)
cfg.add_argument("--recent-window", type=int, default=1)
cfg.add_argument("--recent-decay", type=float, default=0.5)
cfg.add_argument("--transition-support-weight", type=float, default=0.0)
cfg.add_argument("--transition-pair-weight", type=float, default=0.0)
cfg.add_argument("--init-checkpoint", type=str, default=None)
cfg.add_argument("--full-catalog-train", type=int, default=0)
cfg.add_argument("--use-temporal", type=int, default=1)
cfg.add_argument("--expert-negatives", type=int, default=1024)
cfg.add_argument("--hard-prefix-negatives", type=int, default=1024)
cfg.add_argument("--tau-align", type=float, default=0.1)
cfg.add_argument("--lam-rank", type=float, default=1.0)
cfg.add_argument("--lam-sid", type=float, default=0.5)
cfg.add_argument("--lam-align", type=float, default=0.35)
cfg.add_argument("--lam-diversity", type=float, default=0.01)
cfg.add_argument("--slot-assign-temperature", type=float, default=0.5)
cfg.add_argument(
    "--category", type=str, default="Beauty",
    choices=("Beauty", "Sports", "Toys"),
)
cfg.add_argument("--sid-vocab-file", type=str, default=None)

cfg.set_defaults(
    description="PATE-MI-SID",
    root="data",
    dataset="Amazon2014Beauty_550_LOU",
    encoder_type="t5",
    epochs=20,
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
        "Recall@1", "Recall@5", "Recall@10", "Recall@20",
        "NDCG@5", "NDCG@10", "NDCG@20",
        "MRR@5", "MRR@10", "MRR@20",
    ],
)
cfg.compile()


class CoachForMISID(freerec.launcher.Coach):
    def train_per_epoch(self, epoch: int):
        self.model.train()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["rec_loss"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            self.monitor(
                loss.item(),
                n=len(data[self.User]),
                reduction="mean",
                mode="train",
                pool=["LOSS"],
            )


def build_pipe(dataset, model, split, batch_size):
    if split == "train":
        return (
            dataset.train()
            .shuffled_roll_seqs_source(
                minlen=2, maxlen=cfg.maxlen, keep_at_least_itself=True
            )
            .seq_train_yielding_pos_(
                start_idx_for_target=-1, end_idx_for_input=-1
            )
            .map_(model.format_item_ids, modified_fields=(model.ISeq, model.IPos))
            .map_(model.encode_item_text, modified_fields=(model.ISeq, model.IPos))
            .batch_(batch_size)
            .tensor_()
        )
    if split == "valid":
        return (
            dataset.valid()
            .ordered_user_ids_source()
            .valid_sampling_(cfg.ranking)
            .lprune_(cfg.maxlen, modified_fields=(model.ISeq,))
            .map_(model.format_item_ids, modified_fields=(model.ISeq,))
            .map_(model.encode_item_text, modified_fields=(model.ISeq,))
            .batch_(batch_size)
            .tensor_()
        )
    return (
        dataset.test()
        .ordered_user_ids_source()
        .test_sampling_(cfg.ranking)
        .lprune_(cfg.maxlen, modified_fields=(model.ISeq,))
        .map_(model.format_item_ids, modified_fields=(model.ISeq,))
        .map_(model.encode_item_text, modified_fields=(model.ISeq,))
        .batch_(batch_size)
        .tensor_()
    )


def main():
    dataset = build_dataset(cfg.category)
    default_rq_vocab = (
        Path(__file__).resolve().parent
        / f"logs/sid_vocab_rqvae_{cfg.category}.json"
    )
    vocab_file = cfg.sid_vocab_file or (
        str(default_rq_vocab)
        if default_rq_vocab.exists()
        else str(sid_vocab_path(cfg.category))
    )
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        vocab_file, item_count(dataset)
    )
    cfg.encoder_type = "t5_prefix" if cfg.t5_prefix else ("t5_pooled" if cfg.t5_pooled else "t5")
    model = MISIDRec(
        dataset, cfg, tokenizer, converter, code_of_tok, item_codes
    )
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
        with torch.no_grad():
            shared = model.backbone.shared.weight
            for level in range(model.num_sid_tokens):
                for code in range(model.num_codewords):
                    token = f"<sid_{level}_{code}>"
                    tid = tokenizer.convert_tokens_to_ids(token)
                    if tid is not None and int(tid) < shared.size(0):
                        model.code_embs[level].weight[code].copy_(shared[int(tid)])
        if cfg.freeze_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        print(f"[TIGER-PED-Slot] loaded original TIGER encoder keys={len(enc_state)} missing={len(missing)} unexpected={len(unexpected)}; backbone_frozen={cfg.freeze_backbone}", flush=True)
    if cfg.init_checkpoint:
        state = torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if state and next(iter(state)).startswith("module."):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        # Allow capacity sweeps (e.g. larger expert rank/mixtures) to reuse
        # compatible backbone and heads while leaving only changed expert
        # tensors freshly initialized. This is still self-supervised training
        # from the same model family, never teacher/distillation supervision.
        current = model.state_dict()
        compatible = {
            key: value for key, value in state.items()
            if key in current and tuple(value.shape) == tuple(current[key].shape)
        }
        skipped = sorted(set(state) - set(compatible))
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(
            f"[PATE-MI-SID] initialized from own checkpoint={cfg.init_checkpoint} "
            f"loaded={len(compatible)} missing={len(missing)} unexpected={len(unexpected)} "
            f"shape_skipped={len(skipped)}",
            flush=True,
        )
    if bool(getattr(cfg, "freeze_base", 0)) and model.hierarchical_expert is not None:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("hierarchical_expert.")
        model.hierarchical_expert.scale_raw.data.fill_(
            float(getattr(cfg, "hierarchical_scale_init", -5.0))
        )
        print(
            "[PATE-MI-SID] frozen-base residual training: only "
            "hierarchical_expert parameters are trainable",
            flush=True,
        )
    if bool(getattr(cfg, "freeze_recent_transition", 0)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("recent_query_proj.") or name.startswith("recent_key_proj.")
        print(
            "[PATE-MI-SID] frozen-base transition training: only recent "
            "query/key projections are trainable",
            flush=True,
        )
    print(
        "[TIGER-PED-Slot] no-beam full-catalog run: original T5 encoder, parallel "
        "multi-interest experts, no decoder/teacher/distillation/candidate rerank",
        flush=True,
    )
    trainpipe = build_pipe(dataset, model, "train", cfg.batch_size)
    validpipe = build_pipe(dataset, model, "valid", 64)
    testpipe = build_pipe(dataset, model, "test", 64)
    coach = CoachForMISID(
        dataset=dataset,
        trainpipe=trainpipe,
        validpipe=validpipe,
        testpipe=testpipe,
        model=model,
        cfg=cfg,
    )
    coach.fit()
    coach.shutdown()


if __name__ == "__main__":
    main()
