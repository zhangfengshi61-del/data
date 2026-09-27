"""MTP baseline (RPG-style one-shot parallel SID prediction).

One backbone forward predicts the coarse prefix ``c1`` and the fine suffix
``c2/c3`` with independent linear heads. This is phase 1 of the two-stage
HiFlow-SID recipe and also serves as the MTP/RPG-style baseline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from hiflow_lib.common import mean_pool, tokenize_batch


class MTPRec(freerec.models.SeqRecArch):
    def __init__(
        self,
        dataset: freerec.data.datasets.RecDataSet,
        cfg,
        tokenizer: T5Tokenizer,
        converter,
        code_of_tok: List[torch.Tensor],
        item_codes: torch.Tensor,
    ) -> None:
        super().__init__(dataset)

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.converter = converter
        self.code_of_tok = code_of_tok
        self.num_sid_tokens = len(code_of_tok)
        self.num_codewords = cfg.num_codewords
        self.dim = cfg.embedding_dim

        model_config = T5Config(
            vocab_size=len(self.tokenizer),
            d_model=cfg.embedding_dim,
            d_kv=cfg.attention_size,
            d_ff=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
        )
        self.backbone = T5EncoderModel(model_config)

        self.prefix_head = nn.Linear(self.dim, self.num_codewords)
        self.suffix_heads = nn.ModuleList(
            [nn.Linear(self.dim, self.num_codewords) for _ in range(self.num_sid_tokens - 1)]
        )

        self.code_embs = nn.ModuleList(
            [nn.Embedding(self.num_codewords, self.dim) for _ in range(self.num_sid_tokens)]
        )
        self.item_proj = nn.Linear(self.num_sid_tokens * self.dim, self.dim)
        self.h_proj = nn.Linear(self.dim, self.dim)

        self.register_buffer("item_codes", item_codes, persistent=False)
        for level, table in enumerate(self.code_of_tok):
            self.register_buffer(f"code_of_tok_{level}", table, persistent=False)

    def _code_table(self, level: int) -> torch.Tensor:
        return getattr(self, f"code_of_tok_{level}")

    def format_item_ids(self, field, item_ids: Iterable[int]) -> List[str]:
        return [self.converter.format(item) for item in item_ids]

    def encode_item_text(self, field, items: Iterable[str]) -> str:
        return self.converter.encode(items)

    def encode_history(self, data: Dict) -> torch.Tensor:
        context = tokenize_batch(self.tokenizer, self.backbone.device, data[self.ISeq])
        outs = self.backbone(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
            return_dict=True,
        )
        return mean_pool(outs.last_hidden_state, context["attention_mask"])

    def parse_target_codes(self, data: Dict) -> List[torch.Tensor]:
        targets = tokenize_batch(self.tokenizer, self.backbone.device, data[self.IPos])
        ids = targets["input_ids"]
        return [
            self._code_table(level)[ids[:, level + 1]]
            for level in range(self.num_sid_tokens)
        ]

    def item_repr(self, codes: List[torch.Tensor]) -> torch.Tensor:
        embs = [emb(code) for emb, code in zip(self.code_embs, codes)]
        return self.item_proj(torch.cat(embs, dim=-1))

    def forward_heads(self, h: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        prefix_logits = self.prefix_head(h)
        suffix_logits = [head(h) for head in self.suffix_heads]
        return prefix_logits, suffix_logits

    def fit(self, data: Dict) -> Dict[str, torch.Tensor]:
        h = self.encode_history(data)
        codes = self.parse_target_codes(data)

        prefix_logits, suffix_logits = self.forward_heads(h)
        prefix_loss = F.cross_entropy(prefix_logits, codes[0])
        suffix_loss = sum(
            F.cross_entropy(suffix_logits[m], codes[m + 1])
            for m in range(self.num_sid_tokens - 1)
        )

        h_proj = F.normalize(self.h_proj(h), dim=-1)
        item_rep = F.normalize(self.item_repr(codes), dim=-1)
        logits = h_proj @ item_rep.t() / self.cfg.tau_align
        align_loss = F.cross_entropy(
            logits, torch.arange(h.size(0), device=h.device)
        )

        rec_loss = (
            self.cfg.lam_prefix * prefix_loss
            + self.cfg.lam_suffix * suffix_loss
            + self.cfg.lam_align * align_loss
        )
        return {
            "rec_loss": rec_loss,
            "prefix_loss": prefix_loss,
            "suffix_loss": suffix_loss,
            "align_loss": align_loss,
        }

    @torch.no_grad()
    def recommend_from_full(self, data: Dict) -> torch.Tensor:
        h = self.encode_history(data)
        prefix_logits, suffix_logits = self.forward_heads(h)

        codes = self.item_codes  # (N, 3)
        scores = F.log_softmax(prefix_logits, dim=-1)[:, codes[:, 0]]
        for m in range(self.num_sid_tokens - 1):
            scores = scores + F.log_softmax(suffix_logits[m], dim=-1)[:, codes[:, m + 1]]
        return scores
