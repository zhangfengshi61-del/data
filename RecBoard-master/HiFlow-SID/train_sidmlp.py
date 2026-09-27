"""Train SID-MLP on the exact RecBoard Beauty protocol."""

from __future__ import annotations

import torch
import freerec

from hiflow_lib.common import build_dataset, item_count, make_tokenizer_converter, sid_vocab_path
from hiflow_lib.model_sidmlp import SIDMLPRec

freerec.declare(version="1.0.1")

cfg = freerec.parser.Parser()
cfg.add_argument("--maxlen", type=int, default=20)
cfg.add_argument("--embedding-dim", type=int, default=128)
cfg.add_argument("--attention-size", type=int, default=64)
cfg.add_argument("--intermediate-size", type=int, default=256)
cfg.add_argument("--num-heads", type=int, default=4)
cfg.add_argument("--num-layers", type=int, default=6)
cfg.add_argument("--num-codewords", type=int, default=256)
cfg.add_argument("--attn-dim", type=int, default=128)
cfg.add_argument("--ffn-dim", type=int, default=256)
cfg.add_argument("--head-hidden", type=int, default=512)
cfg.add_argument("--head-layers", type=int, default=1)
cfg.add_argument("--candidate-chunk", type=int, default=12101)
cfg.add_argument("--tau-align", type=float, default=0.1)
cfg.add_argument("--lam-align", type=float, default=0.1)
cfg.add_argument("--category", type=str, default="Beauty", choices=("Beauty", "Sports", "Toys"))
cfg.add_argument("--sid-vocab-file", type=str, default=None)
cfg.add_argument("--teacher-checkpoint", type=str, default=None)
cfg.set_defaults(
    description="SID-MLP-RecBoard",
    root="data", dataset="Amazon2014Beauty_550_LOU", ranking="full",
    epochs=200, batch_size=512, optimizer="AdamW", lr=5e-4, weight_decay=1e-4,
    seed=2025, eval_freq=5, which4best="NDCG@10",
    monitors=["LOSS", "HitRate@1", "HitRate@5", "HitRate@10", "HitRate@20",
              "Recall@1", "Recall@5", "Recall@10", "Recall@20",
              "NDCG@5", "NDCG@10", "NDCG@20"],
)
cfg.compile()


class CoachForSIDMLP(freerec.launcher.Coach):
    def train_per_epoch(self, epoch: int):
        self.model.train()
        for data in self.dataloader:
            data = self.dict_to_device(data)
            losses = self.model(data)
            loss = losses["rec_loss"]
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.monitor(loss.item(), n=len(data[self.User]), reduction="mean",
                         mode="train", pool=["LOSS"])


def main():
    dataset = build_dataset(cfg.category)
    tokenizer, converter, code_of_tok, item_codes = make_tokenizer_converter(
        cfg.sid_vocab_file or str(sid_vocab_path(cfg.category)), item_count(dataset)
    )
    model = SIDMLPRec(dataset, cfg, tokenizer, converter, code_of_tok, item_codes)
    ckpt = cfg.teacher_checkpoint
    if ckpt:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        # The SDQ checkpoint is a full T5 (encoder + decoder).  RecBoard
        # SID-MLP only needs the frozen encoder and shared embedding.
        state = {
            k[3:]: v for k, v in state.items()
            if k.startswith("t5.shared.") or k.startswith("t5.encoder.")
        }
        missing, unexpected = model.backbone.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"teacher load mismatch missing={missing} unexpected={unexpected}")
        print(f"[SID-MLP] loaded frozen teacher {ckpt}", flush=True)

    trainpipe = (dataset.train().shuffled_roll_seqs_source(minlen=2, maxlen=cfg.maxlen,
        keep_at_least_itself=True).seq_train_yielding_pos_(start_idx_for_target=-1,
        end_idx_for_input=-1).map_(model.format_item_ids, modified_fields=(model.ISeq, model.IPos))
        .map_(model.encode_item_text, modified_fields=(model.ISeq, model.IPos)).batch_(cfg.batch_size).tensor_())
    validpipe = (dataset.valid().ordered_user_ids_source().valid_sampling_(cfg.ranking)
        .lprune_(cfg.maxlen, modified_fields=(model.ISeq,)).map_(model.format_item_ids,
        modified_fields=(model.ISeq,)).map_(model.encode_item_text, modified_fields=(model.ISeq,))
        .batch_(64).tensor_())
    testpipe = (dataset.test().ordered_user_ids_source().test_sampling_(cfg.ranking)
        .lprune_(cfg.maxlen, modified_fields=(model.ISeq,)).map_(model.format_item_ids,
        modified_fields=(model.ISeq,)).map_(model.encode_item_text, modified_fields=(model.ISeq,))
        .batch_(64).tensor_())
    coach = CoachForSIDMLP(dataset=dataset, trainpipe=trainpipe, validpipe=validpipe,
                           testpipe=testpipe, model=model, cfg=cfg)
    # Protocol-style run: periodic validation every eval_freq epochs, best
    # checkpoint selected by val NDCG@10, early stopping on patience, and a
    # final valid + test evaluation at the best epoch (eval_at_best).
    coach.fit()
    coach.shutdown()


if __name__ == "__main__":
    main()
