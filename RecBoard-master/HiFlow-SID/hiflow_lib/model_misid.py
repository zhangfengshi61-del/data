"""MI-SID: distillation-free multi-intent one-pass semantic-ID recommender.

This module implements the method described in
/data/fszhang/tiger并行加速优化/MI-SID-无蒸馏一步式多兴趣专家路由方案.md.

The model uses:
  * a small fast history encoder;
  * K soft multi-interest slots;
  * parallel SID position heads;
  * a low-rank prefix-conditioned suffix score;
  * direct interaction supervision only.

No MTP/TIGER teacher is constructed or loaded.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer

from hiflow_lib.common import mean_pool, tokenize_batch
from hiflow_lib.model_hiflow import FastHistoryEncoder


class LightHistoryEncoder(nn.Module):
    """Small train-from-scratch self-attention history encoder."""

    def __init__(self, vocab_size: int, dim: int, hidden: int, layers: int,
                 max_seq_len: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("embedding dimension must be divisible by attention heads")
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=hidden,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        length = input_ids.size(1)
        if length > self.pos_emb.num_embeddings:
            raise ValueError("history token length exceeds encoder limit")
        positions = torch.arange(length, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_emb(positions).unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        return self.norm(x)


class MultiIntentSuffixExpert(nn.Module):
    """Low-rank joint SID score for K user-interest slots."""

    def __init__(
        self,
        dim: int,
        num_codewords: int,
        num_sid_tokens: int,
        rank: int,
        num_slots: int,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.num_slots = num_slots
        self.user_proj = nn.Linear(dim, rank, bias=False)
        self.prefix_proj = nn.Linear(dim, rank, bias=False)
        self.suffix_factors = nn.ModuleList(
            [nn.Embedding(num_codewords, rank) for _ in range(num_sid_tokens - 1)]
        )
        self.gate = nn.Parameter(torch.tensor(0.05))
        for factor in self.suffix_factors:
            nn.init.normal_(factor.weight, mean=0.0, std=0.02)

    def forward(
        self,
        slot_states: torch.Tensor,
        candidate_codes: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return [batch, slots, candidates] joint scores."""
        # slot_states: [B,K,D], candidate_codes: [M,L]
        user = self.user_proj(slot_states)
        prefix = self.prefix_proj(prefix_embedding[candidate_codes[:, 0]])
        conditioned = torch.tanh(
            user.unsqueeze(2) + prefix.unsqueeze(0).unsqueeze(0)
        )
        suffix = self.suffix_factors[0](candidate_codes[:, 1])
        for offset, factor in enumerate(self.suffix_factors[1:], start=2):
            suffix = suffix * factor(candidate_codes[:, offset])
        score = (conditioned * suffix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        return self.gate * score / (self.rank ** 0.5)


class MISIDRec(freerec.models.SeqRecArch):
    """RecBoard-compatible MI-SID model with one-pass full-catalog scoring."""

    def __init__(
        self,
        dataset,
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
        self._token_to_id = tokenizer.get_vocab()
        self._unk_id = int(tokenizer.unk_token_id or 0)
        self._pad_id = int(tokenizer.pad_token_id or 0)
        self.code_of_tok = code_of_tok
        self.num_sid_tokens = len(code_of_tok)
        self.num_codewords = int(cfg.num_codewords)
        self.dim = int(cfg.embedding_dim)
        self.num_slots = int(cfg.num_slots)

        # The existing fast encoder is a lightweight global-context encoder.
        # It receives protocol token IDs and has no T5/teacher dependency.
        encoder_type = str(getattr(cfg, "encoder_type", "fast"))
        if encoder_type == "transformer":
            self.backbone = LightHistoryEncoder(
                vocab_size=len(tokenizer), dim=self.dim,
                hidden=int(cfg.fast_hidden), layers=int(cfg.fast_layers),
                max_seq_len=int(cfg.fast_max_seq_len),
                heads=int(getattr(cfg, "encoder_heads", 4)),
            )
        else:
            self.backbone = FastHistoryEncoder(
                vocab_size=len(tokenizer),
                dim=self.dim,
                hidden=int(cfg.fast_hidden),
                layers=int(cfg.fast_layers),
                max_seq_len=int(cfg.fast_max_seq_len),
                num_positions=int(cfg.fast_num_positions),
            )
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, self.dim) * 0.02)
        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.SiLU(),
            nn.Linear(self.dim // 2, 1),
        )

        # Shared code projections plus a small slot-specific adapter.
        self.prefix_heads = nn.ModuleList(
            [nn.Linear(self.dim, self.num_codewords) for _ in range(self.num_slots)]
        )
        self.suffix_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(self.dim, self.num_codewords)
                     for _ in range(self.num_sid_tokens - 1)]
                )
                for _ in range(self.num_slots)
            ]
        )
        self.slot_adapters = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.dim, self.dim), nn.Tanh())
             for _ in range(self.num_slots)]
        )

        self.code_embs = nn.ModuleList(
            [nn.Embedding(self.num_codewords, self.dim)
             for _ in range(self.num_sid_tokens)]
        )
        self.item_proj = nn.Linear(self.num_sid_tokens * self.dim, self.dim)
        self.user_proj = nn.Linear(self.dim, self.dim)
        self.joint_expert = MultiIntentSuffixExpert(
            dim=self.dim,
            num_codewords=self.num_codewords,
            num_sid_tokens=self.num_sid_tokens,
            rank=int(cfg.expert_rank),
            num_slots=self.num_slots,
        )

        self.register_buffer("item_codes", item_codes, persistent=False)
        for level, table in enumerate(self.code_of_tok):
            self.register_buffer(f"code_of_tok_{level}", table, persistent=False)

    def _code_table(self, level: int) -> torch.Tensor:
        return getattr(self, f"code_of_tok_{level}")

    def format_item_ids(self, field, item_ids: Iterable[int]) -> List[str]:
        return [self.converter.format(item) for item in item_ids]

    def encode_item_text(self, field, items: Iterable[str]) -> str:
        return self.converter.encode(items)

    def _tokenize_fast(self, texts, device: torch.device) -> Dict[str, torch.Tensor]:
        rows: List[List[int]] = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            row = [
                self._token_to_id.get(token, self._unk_id)
                for token in text.split()
            ]
            rows.append(row or [self._pad_id])
        max_len = max(len(row) for row in rows)
        ids = [row + [self._pad_id] * (max_len - len(row)) for row in rows]
        masks = [
            [1] * len(row) + [0] * (max_len - len(row))
            for row in rows
        ]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        }

    def encode_history_sequence(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        context = self._tokenize_fast(data[self.ISeq], device)
        hidden = self.backbone(
            input_ids=context["input_ids"],
            attention_mask=context["attention_mask"],
        )
        return hidden, context["attention_mask"]

    def encode_history(self, data: Dict) -> torch.Tensor:
        hidden, mask = self.encode_history_sequence(data)
        return mean_pool(hidden, mask)

    def _slot_states(
        self, hidden: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden [B,T,D], mask [B,T], output slots [B,K,D].
        queries = self.slot_queries.to(hidden.dtype)
        logits = torch.einsum("btd,kd->bkt", hidden, queries)
        logits = logits / (self.dim ** 0.5)
        logits = logits.masked_fill(~mask.bool().unsqueeze(1), -1e4)
        weights = torch.softmax(logits, dim=-1)
        slots = torch.einsum("bkt,btd->bkd", weights, hidden)
        slots = torch.stack(
            [
                slots[:, k] + self.slot_adapters[k](slots[:, k])
                for k in range(self.num_slots)
            ],
            dim=1,
        )
        route_logits = self.route(slots).squeeze(-1)
        return slots, route_logits

    def encode_slots(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, mask = self.encode_history_sequence(data)
        return self._slot_states(hidden, mask)

    def parse_target_codes(self, data: Dict) -> torch.Tensor:
        device = next(self.parameters()).device
        targets = tokenize_batch(self.tokenizer, device, data[self.IPos])
        ids = targets["input_ids"]
        codes = [
            self._code_table(level)[ids[:, level + 1]]
            for level in range(self.num_sid_tokens)
        ]
        return torch.stack(codes, dim=1)

    def item_repr(self, codes: torch.Tensor) -> torch.Tensor:
        embs = [
            self.code_embs[level](codes[:, level])
            for level in range(self.num_sid_tokens)
        ]
        return self.item_proj(torch.cat(embs, dim=-1))

    def position_logits(self, slots: torch.Tensor) -> List[List[torch.Tensor]]:
        result: List[List[torch.Tensor]] = []
        for k in range(self.num_slots):
            h = slots[:, k]
            result.append(
                [self.prefix_heads[k](h)]
                + [head(h) for head in self.suffix_heads[k]]
            )
        return result

    def _slot_candidate_scores(
        self,
        slots: torch.Tensor,
        position_logits: List[List[torch.Tensor]],
        candidate_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Return [B,K,M] scores for candidate SIDs."""
        b = slots.size(0)
        m = candidate_codes.size(0)
        base = slots.new_zeros((b, self.num_slots, m))
        for k in range(self.num_slots):
            current = slots.new_zeros((b, m))
            logits_k = position_logits[k]
            for level, logits in enumerate(logits_k):
                log_probs = F.log_softmax(logits, dim=-1)
                current = current + log_probs[:, candidate_codes[:, level]]
            base[:, k] = current
        base = base + self.joint_expert(
            slots, candidate_codes, self.code_embs[0].weight
        )
        return base

    def _aggregate_candidate_scores(
        self,
        slots: torch.Tensor,
        route_logits: torch.Tensor,
        position_logits: List[List[torch.Tensor]],
        candidate_codes: torch.Tensor,
    ) -> torch.Tensor:
        slot_scores = self._slot_candidate_scores(
            slots, position_logits, candidate_codes
        )
        route_logp = F.log_softmax(route_logits, dim=-1).unsqueeze(-1)
        return torch.logsumexp(route_logp + slot_scores, dim=1)

    def _diversity_loss(self, slots: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(slots, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        eye = torch.eye(self.num_slots, device=slots.device).unsqueeze(0)
        return ((sim * (1.0 - eye)) ** 2).sum() / max(
            slots.size(0) * self.num_slots * (self.num_slots - 1), 1
        )

    def fit(self, data: Dict) -> Dict[str, torch.Tensor]:
        slots, route_logits = self.encode_slots(data)
        codes = self.parse_target_codes(data)
        position_logits = self.position_logits(slots)

        # Softly assign each observed target to the slot that explains it best.
        # This avoids forcing every slot to predict every target equally.
        positive_slot_scores = self._slot_candidate_scores(
            slots, position_logits, codes
        )
        positive_slot_scores = positive_slot_scores.diagonal(
            dim1=0, dim2=2
        ).transpose(0, 1)
        assign_temperature = float(
            getattr(self.cfg, "slot_assign_temperature", 0.5)
        )
        assignment = torch.softmax(
            positive_slot_scores / max(assign_temperature, 1e-4), dim=-1
        ).detach()
        slot_sid_loss = slots.new_zeros(())
        for k in range(self.num_slots):
            for level, logits in enumerate(position_logits[k]):
                slot_sid_loss = slot_sid_loss + (
                    assignment[:, k] * F.cross_entropy(
                        logits, codes[:, level], reduction="none"
                    )
                ).mean()
        slot_sid_loss = slot_sid_loss / self.num_sid_tokens

        negative_count = int(getattr(self.cfg, "expert_negatives", 2048))
        if negative_count > 0:
            sample_ids = torch.randint(
                self.item_codes.size(0),
                (negative_count,),
                device=slots.device,
            )
            candidate_codes = torch.cat([codes, self.item_codes[sample_ids]], dim=0)
        else:
            candidate_codes = codes

        candidate_scores = self._aggregate_candidate_scores(
            slots, route_logits, position_logits, candidate_codes
        )
        rank_loss = F.cross_entropy(
            candidate_scores, torch.arange(slots.size(0), device=slots.device)
        )

        route_prob = F.softmax(route_logits, dim=-1)
        user = F.normalize(
            self.user_proj((route_prob.unsqueeze(-1) * slots).sum(dim=1)),
            dim=-1,
        )
        item = F.normalize(self.item_repr(codes), dim=-1)
        align_loss = F.cross_entropy(
            user @ item.t() / float(self.cfg.tau_align),
            torch.arange(slots.size(0), device=slots.device),
        )

        diversity_loss = self._diversity_loss(slots)
        loss = (
            float(self.cfg.lam_rank) * rank_loss
            + float(self.cfg.lam_sid) * slot_sid_loss
            + float(self.cfg.lam_align) * align_loss
            + float(self.cfg.lam_diversity) * diversity_loss
        )
        return {
            "rec_loss": loss,
            "rank_loss": rank_loss,
            "sid_loss": slot_sid_loss,
            "align_loss": align_loss,
            "diversity_loss": diversity_loss,
        }

    @torch.no_grad()
    def recommend_from_full(self, data: Dict) -> torch.Tensor:
        slots, route_logits = self.encode_slots(data)
        position_logits = self.position_logits(slots)
        return self._aggregate_candidate_scores(
            slots, route_logits, position_logits, self.item_codes
        )
