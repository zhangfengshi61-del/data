"""PATE-MI-SID: stronger expert, distillation-free full-catalog recommender.

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
from transformers import T5Config, T5EncoderModel, T5Tokenizer

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



class PathTensorExpert(nn.Module):
    """Low-rank user x prefix x suffix tensor expert for full-catalog scoring."""

    def __init__(
        self,
        dim: int,
        num_codewords: int,
        num_sid_tokens: int,
        rank: int,
        num_slots: int,
        mixtures: int = 2,
    ) -> None:
        super().__init__()
        self.mixtures = int(mixtures)
        self.rank = max(4, int(rank) // self.mixtures)
        self.num_slots = num_slots
        width = self.mixtures * self.rank
        self.user_proj = nn.Linear(dim, width, bias=False)
        self.pair_user_proj = nn.Linear(dim, width, bias=False)
        self.mix_gate = nn.Linear(dim, self.mixtures)
        self.code_factors = nn.ModuleList(
            [nn.Embedding(num_codewords, width) for _ in range(num_sid_tokens)]
        )
        self.pair01_proj = nn.Linear(width, width, bias=False)
        self.pair012_proj = nn.Linear(width, width, bias=False)
        self.gate_raw = nn.Parameter(torch.tensor(-2.2))
        self.pair_scale_raw = nn.Parameter(torch.tensor(-0.7))
        for factor in self.code_factors:
            nn.init.normal_(factor.weight, mean=0.0, std=0.035)
        # Static candidate factors are cached only in eval mode.
        self._catalog_factor_cache = None

    def forward(
        self,
        slot_states: torch.Tensor,
        candidate_codes: torch.Tensor,
        prefix_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # slot_states [B,K,D], candidate_codes [M,L].
        b, k, _ = slot_states.shape
        e, r = self.mixtures, self.rank
        u = self.user_proj(slot_states).view(b, k, e, r)
        up = self.pair_user_proj(slot_states).view(b, k, e, r)
        use_cache = not self.training
        version = tuple(f.weight._version for f in self.code_factors)
        version += (self.pair01_proj.weight._version, self.pair012_proj.weight._version)
        cache_key = (int(candidate_codes.data_ptr()), str(candidate_codes.device), version)
        cached = self._catalog_factor_cache
        if use_cache and cached is not None and cached[0] == cache_key:
            path, pair01, pair012 = cached[1:]
        else:
            f0 = self.code_factors[0](candidate_codes[:, 0]).view(-1, e, r)
            f1 = self.code_factors[1](candidate_codes[:, 1]).view(-1, e, r)
            if len(self.code_factors) > 2:
                f2 = self.code_factors[2](candidate_codes[:, 2]).view(-1, e, r)
            else:
                f2 = torch.ones_like(f1)
            path = f0 * f1 * f2
            pair01 = self.pair01_proj((f0 * f1).reshape(-1, e * r)).view(-1, e, r)
            pair012 = self.pair012_proj(path.reshape(-1, e * r)).view(-1, e, r)
            if use_cache:
                self._catalog_factor_cache = (cache_key, path, pair01, pair012)

        tensor_score = torch.einsum("bker,mer->bkem", u, path)
        pair_score = torch.einsum("bker,mer->bkem", up, pair01 + 0.5 * pair012)
        mix = torch.softmax(self.mix_gate(slot_states), dim=-1).unsqueeze(-1)
        raw = ((tensor_score + torch.sigmoid(self.pair_scale_raw) * pair_score) * mix).sum(dim=2)
        gate = 0.05 + 0.25 * torch.sigmoid(self.gate_raw)
        return gate * raw / (r ** 0.5)



class HierarchicalPathExpert(nn.Module):
    """Parallel prefix-conditioned SID path expert.

    It scores every catalog item in one matrix pass while exposing the
    conditional structure that an autoregressive SID decoder uses:
    level-0, prefix-conditioned level-1, and prefix-pair-conditioned
    subsequent levels. No code is generated sequentially and no candidate
    subset is formed.
    """

    def __init__(self, dim: int, num_codewords: int, num_sid_tokens: int,
                 rank: int) -> None:
        super().__init__()
        self.rank = max(4, int(rank))
        self.num_sid_tokens = int(num_sid_tokens)
        self.user_projs = nn.ModuleList(
            [nn.Linear(dim, self.rank, bias=False)
             for _ in range(self.num_sid_tokens)]
        )
        self.code_factors = nn.ModuleList(
            [nn.Embedding(num_codewords, self.rank)
             for _ in range(self.num_sid_tokens)]
        )
        self.prefix_proj = nn.Linear(2 * self.rank, self.rank, bias=False)
        self.path_proj = nn.Linear(2 * self.rank, self.rank, bias=False)
        # Start close to the existing score so a new branch cannot destroy a
        # good checkpoint at epoch zero; its scale is learned by ranking loss.
        self.scale_raw = nn.Parameter(torch.tensor(-5.0))
        for factor in self.code_factors:
            nn.init.normal_(factor.weight, mean=0.0, std=0.05)

    def forward(self, slot_states: torch.Tensor,
                candidate_codes: torch.Tensor) -> torch.Tensor:
        # slot_states [B,K,D], candidate_codes [M,L].
        users = torch.stack(
            [proj(slot_states) for proj in self.user_projs], dim=2
        )  # [B,K,L,R]
        path = self.code_factors[0](candidate_codes[:, 0])
        score = torch.einsum("bkr,mr->bkm", users[:, :, 0], path)
        if self.num_sid_tokens > 1:
            path = torch.tanh(self.prefix_proj(torch.cat(
                [path, self.code_factors[1](candidate_codes[:, 1])], dim=-1
            )))
            score = score + torch.einsum(
                "bkr,mr->bkm", users[:, :, 1], path
            )
        for level in range(2, self.num_sid_tokens):
            path = torch.tanh(self.path_proj(torch.cat(
                [path, self.code_factors[level](candidate_codes[:, level])],
                dim=-1
            )))
            score = score + torch.einsum(
                "bkr,mr->bkm", users[:, :, level], path
            )
        return (0.20 * torch.sigmoid(self.scale_raw)
                * score / (self.rank ** 0.5))

class LegacySuffixExpert(nn.Module):
    """Small suffix-path residual expert kept fully inside the one-pass score.

    This is the factorized interaction used by MI-SID, exposed as an optional
    parallel residual branch. It never narrows the catalog and does not consume
    a teacher or a retrieved candidate list.
    """

    def __init__(
        self,
        dim: int,
        num_codewords: int,
        num_sid_tokens: int,
        rank: int,
        num_slots: int,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.num_slots = num_slots
        self.user_proj = nn.Linear(dim, self.rank, bias=False)
        self.prefix_proj = nn.Linear(dim, self.rank, bias=False)
        self.suffix_factors = nn.ModuleList(
            [nn.Embedding(num_codewords, self.rank)
             for _ in range(num_sid_tokens - 1)]
        )
        self.gate_raw = nn.Parameter(torch.tensor(-1.5))
        for factor in self.suffix_factors:
            nn.init.normal_(factor.weight, mean=0.0, std=0.02)

    def forward(
        self,
        slot_states: torch.Tensor,
        candidate_codes: torch.Tensor,
        prefix_embedding: torch.Tensor,
    ) -> torch.Tensor:
        user = self.user_proj(slot_states)
        prefix = self.prefix_proj(prefix_embedding[candidate_codes[:, 0]])
        conditioned = torch.tanh(
            user.unsqueeze(2) + prefix.unsqueeze(0).unsqueeze(0)
        )
        suffix = self.suffix_factors[0](candidate_codes[:, 1])
        for offset, factor in enumerate(self.suffix_factors[1:], start=2):
            suffix = suffix * factor(candidate_codes[:, offset])
        score = (conditioned * suffix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        gate = 0.05 + 0.20 * torch.sigmoid(self.gate_raw)
        return gate * score / (self.rank ** 0.5)


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
        # CPU lookup avoids GPU .item() synchronization while parsing history.
        self._token_code_cpu = torch.stack(
            [table.detach().long().cpu() for table in code_of_tok], dim=1
        ).tolist()
        self.num_codewords = int(cfg.num_codewords)
        self.dim = int(cfg.embedding_dim)
        self.num_slots = int(cfg.num_slots)

        # The fast encoders remain available for legacy MI-SID. TIGER-PED-Slot
        # can instead reuse the original TIGER T5 encoder directly; no decoder
        # or teacher path is constructed.
        encoder_type = str(getattr(cfg, "encoder_type", "fast"))
        self.encoder_type = encoder_type
        self.t5_prefix_width = max(1, int(getattr(cfg, "t5_prefix_width", 1)))
        self.use_sid_residual = bool(getattr(cfg, "use_sid_residual", False))
        self.sid_residual_weight = float(getattr(cfg, "sid_residual_weight", 1.0))
        self.sid_residual_proj = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.Tanh(),
        )
        self.sid_residual_gate = nn.Parameter(torch.tensor(-5.0))
        if encoder_type in ("t5", "t5_pooled", "t5_prefix"):
            t5_config = T5Config(
                vocab_size=len(tokenizer), d_model=self.dim,
                d_kv=int(getattr(cfg, "attention_size", 64)),
                d_ff=int(getattr(cfg, "intermediate_size", 256)),
                num_layers=int(getattr(cfg, "t5_layers", 6)),
                num_heads=int(getattr(cfg, "encoder_heads", 4)),
                dropout_rate=float(getattr(cfg, "dropout_rate", 0.1)),
            )
            self.backbone = T5EncoderModel(t5_config)
        elif encoder_type == "transformer":
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
        self.use_temporal = bool(getattr(cfg, "use_temporal", True))
        self.temporal_conv = nn.Conv1d(
            self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim
        )
        self.temporal_gate = nn.Linear(self.dim, self.dim)
        self.temporal_proj = nn.Linear(3 * self.dim, self.dim)

        self.code_embs = nn.ModuleList(
            [nn.Embedding(self.num_codewords, self.dim)
             for _ in range(self.num_sid_tokens)]
        )
        self.item_proj = nn.Linear(self.num_sid_tokens * self.dim, self.dim)
        self.user_proj = nn.Linear(self.dim, self.dim)
        # A trainable one-step transition expert.  Identity initialization
        # preserves the direct recent-item score for old checkpoints; tuning
        # this branch learns last-item -> next-item structure from interactions.
        self.recent_query_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.recent_key_proj = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.eye_(self.recent_query_proj.weight)
        nn.init.eye_(self.recent_key_proj.weight)
        # Direct semantic user-item score.  It is an additive term in the
        # same full-catalog score, never a retrieval or reranking stage.
        self.semantic_score_weight = float(
            getattr(cfg, "semantic_score_weight", 0.0)
        )
        self.path_score_weight = float(
            getattr(cfg, "path_score_weight", 1.0)
        )
        self.route_temperature = max(
            1e-3, float(getattr(cfg, "route_temperature", 1.0))
        )
        self.recent_score_weight = float(
            getattr(cfg, "recent_score_weight", 0.0)
        )
        self.recent_window = max(
            1, int(getattr(cfg, "recent_window", 1))
        )
        self.recent_decay = float(
            getattr(cfg, "recent_decay", 0.5)
        )
        self.transition_support_weight = float(
            getattr(cfg, "transition_support_weight", 0.0)
        )
        self.transition_pair_weight = float(
            getattr(cfg, "transition_pair_weight", 0.0)
        )
        raw_level_weights = getattr(cfg, "transition_level_weights", None)
        if raw_level_weights is None:
            self.transition_level_weights = [1.0] * self.num_sid_tokens
        else:
            self.transition_level_weights = [
                float(raw_level_weights[level]) if level < len(raw_level_weights) else 1.0
                for level in range(self.num_sid_tokens)
            ]
        if self.transition_support_weight != 0.0:
            alpha = max(float(getattr(cfg, "transition_support_alpha", 1.0)), 1e-6)
            counts = torch.full(
                (self.num_sid_tokens, self.num_codewords, self.num_codewords),
                alpha, dtype=torch.float32
            )
            pair_counts = torch.full(
                (self.num_codewords * self.num_codewords, self.num_codewords),
                alpha, dtype=torch.float32
            )
            item_codes_cpu = item_codes.detach().long().cpu()
            try:
                for row in dataset.train().to_seqs(maxlen=int(getattr(cfg, "maxlen", 20))):
                    seq = [int(item) for item in row[self.ISeq]]
                    for source, target in zip(seq[:-1], seq[1:]):
                        if source < 0 or target < 0 or source >= item_codes_cpu.size(0) or target >= item_codes_cpu.size(0):
                            continue
                        src = item_codes_cpu[source]
                        dst = item_codes_cpu[target]
                        for level in range(self.num_sid_tokens):
                            counts[level, src[level], dst[level]] += 1.0
                        pair_key = int(src[0]) * self.num_codewords + int(src[1])
                        pair_counts[pair_key, dst[2]] += 1.0
            except Exception as exc:
                print(f"[PATE-MI-SID] transition support unavailable: {exc}", flush=True)
            row_sum = counts.sum(dim=-1, keepdim=True)
            self.register_buffer(
                "transition_logprob", torch.log(counts / row_sum.clamp_min(1e-6)),
                persistent=False,
            )
            pair_row_sum = pair_counts.sum(dim=-1, keepdim=True)
            self.register_buffer(
                "transition_pair_logprob",
                torch.log(pair_counts / pair_row_sum.clamp_min(1e-6)),
                persistent=False,
            )
        else:
            self.transition_logprob = None
            self.transition_pair_logprob = None
        self.joint_expert = PathTensorExpert(
            dim=self.dim,
            num_codewords=self.num_codewords,
            num_sid_tokens=self.num_sid_tokens,
            rank=int(cfg.expert_rank),
            num_slots=self.num_slots,
            mixtures=int(getattr(cfg, "expert_mixtures", 2)),
        )
        hierarchical_rank = int(getattr(cfg, "hierarchical_rank", 0))
        self.hierarchical_expert = (
            HierarchicalPathExpert(
                dim=self.dim,
                num_codewords=self.num_codewords,
                num_sid_tokens=self.num_sid_tokens,
                rank=hierarchical_rank,
            )
            if hierarchical_rank > 0 else None
        )
        residual_rank = int(getattr(cfg, "residual_rank", 0))
        self.residual_expert = (
            LegacySuffixExpert(
                dim=self.dim,
                num_codewords=self.num_codewords,
                num_sid_tokens=self.num_sid_tokens,
                rank=residual_rank,
                num_slots=self.num_slots,
            )
            if residual_rank > 0 else None
        )

        self.register_buffer("item_codes", item_codes, persistent=False)
        self._eval_catalog_item_cache = None
        item_codes_cpu = item_codes.detach().long().cpu()
        prefix_counts = torch.zeros(self.num_codewords, dtype=torch.long)
        max_bucket = 1
        for prefix in range(self.num_codewords):
            count = int((item_codes_cpu[:, 0] == prefix).sum().item())
            prefix_counts[prefix] = count
            max_bucket = max(max_bucket, count)
        prefix_ids = torch.zeros(
            self.num_codewords, max_bucket, dtype=torch.long
        )
        for prefix in range(self.num_codewords):
            ids = torch.nonzero(item_codes_cpu[:, 0] == prefix).flatten()
            if ids.numel():
                prefix_ids[prefix, :ids.numel()] = ids
        self.register_buffer("prefix_bucket_ids", prefix_ids, persistent=False)
        self.register_buffer("prefix_bucket_counts", prefix_counts, persistent=False)
        pair_keys = item_codes_cpu[:, 0] * self.num_codewords + item_codes_cpu[:, 1]
        pair_counts = torch.zeros(self.num_codewords * self.num_codewords, dtype=torch.long)
        max_pair_bucket = 1
        for key in torch.unique(pair_keys).tolist():
            count = int((pair_keys == key).sum().item())
            pair_counts[key] = count
            max_pair_bucket = max(max_pair_bucket, count)
        pair_ids = torch.zeros(
            self.num_codewords * self.num_codewords, max_pair_bucket, dtype=torch.long
        )
        for key in torch.unique(pair_keys).tolist():
            ids = torch.nonzero(pair_keys == key).flatten()
            pair_ids[key, :ids.numel()] = ids
        self.register_buffer("pair_bucket_ids", pair_ids, persistent=False)
        self.register_buffer("pair_bucket_counts", pair_counts, persistent=False)
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

    def _recent_item_codes(self, texts, device: torch.device) -> torch.Tensor:
        """Parse the most recent SID blocks without any GPU synchronization.

        The returned tensor is [B, W, L] for a window W>1 and [B, L] for
        W=1. It is only used inside the same full-catalog score matrix.
        """
        window = self.recent_window
        rows = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            paths = []
            current = None
            for token in reversed(text.split()):
                if token == "</SID>":
                    current = [-1] * self.num_sid_tokens
                    continue
                if token == "<SID>":
                    if current is not None and all(code >= 0 for code in current):
                        paths.append(current)
                    current = None
                    if len(paths) >= window:
                        break
                    continue
                if current is None:
                    continue
                token_id = self._token_to_id.get(token, self._unk_id)
                if 0 <= token_id < len(self._token_code_cpu):
                    for level, code in enumerate(self._token_code_cpu[token_id]):
                        if code >= 0 and current[level] < 0:
                            current[level] = int(code)
            if not paths:
                paths = [[0] * self.num_sid_tokens]
            while len(paths) < window:
                paths.append(paths[-1])
            rows.append(paths[:window])
        result = torch.tensor(rows, dtype=torch.long, device=device)
        return result[:, 0] if window == 1 else result

    def _last_item_codes(self, texts, device: torch.device) -> torch.Tensor:
        return self._recent_item_codes(texts, device)
 
    def _tokenize_t5_prefix(self, texts, device: torch.device):
        """Use one original TIGER SID token per history item."""
        rows, code_rows = [], []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            ids, current, inside, blocks = [], [], False, []
            for token in text.split():
                if token == "<SID>":
                    current, inside = [], True
                    continue
                if token == "</SID>":
                    if current:
                        ids.extend(current[:self.t5_prefix_width])
                        block = current[:self.num_sid_tokens]
                        block = block + [self._pad_id] * (self.num_sid_tokens - len(block))
                        blocks.append(block)
                    current, inside = [], False
                    continue
                if inside:
                    current.append(self._token_to_id.get(token, self._unk_id))
            rows.append(ids or [self._pad_id])
            code_rows.append(blocks or [[self._pad_id] * self.num_sid_tokens])
        length = max(len(row) for row in rows)
        padded = [row + [self._pad_id] * (length - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (length - len(row)) for row in rows]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        mask = torch.tensor(masks, dtype=torch.long, device=device)
        block_len = max(len(row) for row in code_rows)
        code_padded = [
            row + [[self._pad_id] * self.num_sid_tokens] * (block_len - len(row))
            for row in code_rows
        ]
        code_ids = torch.tensor(code_padded, dtype=torch.long, device=device)
        code_mask = torch.tensor(
            [[1] * len(row) + [0] * (block_len - len(row)) for row in code_rows],
            dtype=torch.float32, device=device,
        )
        shared = self.backbone.shared.weight
        code_emb = shared[code_ids]
        residual = (code_emb.mean(dim=2) * code_mask.unsqueeze(-1)).sum(dim=1)
        residual = residual / code_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return ids, mask, residual

    def _tokenize_t5_pooled(self, texts, device: torch.device):
        """Represent each complete SID block by one TIGER shared-embedding token."""
        shared = self.backbone.shared.weight
        rows = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            blocks, current, inside = [], [], False
            for token in text.split():
                if token == "<SID>":
                    current, inside = [], True
                    continue
                if token == "</SID>":
                    if current:
                        ids = torch.tensor(current, dtype=torch.long, device=device)
                        blocks.append(shared[ids].mean(dim=0))
                    current, inside = [], False
                    continue
                if inside:
                    current.append(self._token_to_id.get(token, self._unk_id))
            if not blocks:
                blocks = [shared.new_zeros((self.dim,))]
            rows.append(torch.stack(blocks, dim=0))
        length = max(row.size(0) for row in rows)
        embeds = shared.new_zeros((len(rows), length, self.dim))
        mask = torch.zeros((len(rows), length), dtype=torch.long, device=device)
        for row, value in enumerate(rows):
            embeds[row, :value.size(0)] = value
            mask[row, :value.size(0)] = 1
        return embeds, mask

    def encode_history_sequence(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.encoder_type == "t5_prefix":
            ids, mask, residual = self._tokenize_t5_prefix(data[self.ISeq], device)
            outputs = self.backbone(
                input_ids=ids,
                attention_mask=mask,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            if self.use_sid_residual:
                correction = self.sid_residual_proj(residual)
                correction = self.sid_residual_weight * torch.sigmoid(self.sid_residual_gate) * correction
                hidden = hidden + correction.unsqueeze(1) * mask.unsqueeze(-1).to(hidden.dtype)
            return hidden, mask
        if self.encoder_type == "t5_pooled":
            embeds, mask = self._tokenize_t5_pooled(data[self.ISeq], device)
            outputs = self.backbone(
                inputs_embeds=embeds,
                attention_mask=mask,
                return_dict=True,
            )
            return outputs.last_hidden_state, mask
        if self.encoder_type == "t5":
            context = tokenize_batch(self.tokenizer, device, data[self.ISeq])
            outputs = self.backbone(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                return_dict=True,
            )
            return outputs.last_hidden_state, context["attention_mask"]
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
        if self.use_temporal:
            x = hidden.transpose(1, 2)
            temporal = self.temporal_conv(x).transpose(1, 2)
            temporal = temporal * torch.sigmoid(self.temporal_gate(temporal))
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            global_h = (hidden * mask_f).sum(dim=1) / denom.squeeze(1)
            recent_mask = mask_f.clone()
            t = hidden.size(1)
            recent_mask[:, :max(0, t - 4)] = 0
            recent_denom = recent_mask.sum(dim=1).clamp_min(1.0)
            recent_h = (temporal * recent_mask).sum(dim=1) / recent_denom
            last_idx = mask.long().sum(dim=1).clamp_min(1) - 1
            last_h = temporal[torch.arange(hidden.size(0), device=hidden.device), last_idx]
            temporal_context = self.temporal_proj(torch.cat([global_h, recent_h, last_h], dim=-1))
            hidden = hidden + temporal_context.unsqueeze(1) * mask_f
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

    def _catalog_item_repr(self) -> torch.Tensor:
        if self.training:
            return self.item_repr(self.item_codes)
        versions = tuple(emb.weight._version for emb in self.code_embs)
        versions += (self.item_proj.weight._version, self.item_proj.bias._version)
        key = (int(self.item_codes.data_ptr()), str(self.item_codes.device), versions)
        cached = self._eval_catalog_item_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        table = self.item_repr(self.item_codes)
        self._eval_catalog_item_cache = (key, table)
        return table

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
        if self.residual_expert is not None:
            base = base + self.residual_expert(
                slots, candidate_codes, self.code_embs[0].weight
            )
        if self.hierarchical_expert is not None:
            base = base + self.hierarchical_expert(slots, candidate_codes)
        return base

    def _aggregate_candidate_scores(
        self,
        slots: torch.Tensor,
        route_logits: torch.Tensor,
        position_logits: List[List[torch.Tensor]],
        candidate_codes: torch.Tensor,
        recent_codes: torch.Tensor | None = None,
        catalog_item_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slot_scores = self._slot_candidate_scores(
            slots, position_logits, candidate_codes
        )
        route_logp = F.log_softmax(
            route_logits / self.route_temperature, dim=-1
        ).unsqueeze(-1)
        score = self.path_score_weight * torch.logsumexp(
            route_logp + slot_scores, dim=1
        )
        raw_item = None
        item = None
        if self.semantic_score_weight > 0.0:
            route_prob = F.softmax(
                route_logits / self.route_temperature, dim=-1
            )
            user = F.normalize(
                self.user_proj((route_prob.unsqueeze(-1) * slots).sum(dim=1)),
                dim=-1,
            )
            if (catalog_item_repr is not None and
                    candidate_codes.data_ptr() == self.item_codes.data_ptr()):
                raw_item = catalog_item_repr
            else:
                raw_item = self.item_repr(candidate_codes)
            item = F.normalize(raw_item, dim=-1)
            score = score + self.semantic_score_weight * (user @ item.t())
        if self.recent_score_weight != 0.0 and recent_codes is not None:
            if raw_item is None:
                if (catalog_item_repr is not None and
                        candidate_codes.data_ptr() == self.item_codes.data_ptr()):
                    raw_item = catalog_item_repr
                else:
                    raw_item = self.item_repr(candidate_codes)
            recent_item = F.normalize(self.recent_key_proj(raw_item), dim=-1)
            if recent_codes.dim() == 3:
                b, w, l = recent_codes.shape
                recent_raw = self.item_repr(recent_codes.reshape(b * w, l)).reshape(b, w, -1)
                recent = F.normalize(self.recent_query_proj(recent_raw), dim=-1)
                decay = max(float(self.recent_decay), 0.0)
                weights = decay ** torch.arange(w, device=score.device, dtype=score.dtype)
                weights = weights / weights.sum().clamp_min(1e-6)
                recent_score = torch.einsum("bwd,md,w->bm", recent, recent_item, weights)
            else:
                recent_raw = self.item_repr(recent_codes)
                recent = F.normalize(self.recent_query_proj(recent_raw), dim=-1)
                recent_score = recent @ recent_item.t()
            score = score + self.recent_score_weight * recent_score
        if self.transition_support_weight != 0.0 and recent_codes is not None:
            support_codes = recent_codes[:, 0] if recent_codes.dim() == 3 else recent_codes
            support = score.new_zeros(score.shape)
            for level in range(self.num_sid_tokens):
                table = self.transition_logprob[level]
                support = support + self.transition_level_weights[level] * table[
                    support_codes[:, level].long().unsqueeze(1),
                    candidate_codes[:, level].long().unsqueeze(0),
                ]
            support = support - support.mean(dim=1, keepdim=True)
            support = support / support.std(dim=1, keepdim=True).clamp_min(1e-6)
            score = score + self.transition_support_weight * support
        if self.transition_pair_weight != 0.0 and recent_codes is not None:
            pair_codes = recent_codes[:, 0] if recent_codes.dim() == 3 else recent_codes
            pair_key = pair_codes[:, 0].long() * self.num_codewords + pair_codes[:, 1].long()
            pair_support = self.transition_pair_logprob[
                pair_key.unsqueeze(1),
                candidate_codes[:, 2].long().unsqueeze(0),
            ]
            pair_support = pair_support - pair_support.mean(dim=1, keepdim=True)
            pair_support = pair_support / pair_support.std(dim=1, keepdim=True).clamp_min(1e-6)
            score = score + self.transition_pair_weight * pair_support
        return score

    def _diversity_loss(self, slots: torch.Tensor) -> torch.Tensor:
        norm = F.normalize(slots, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        eye = torch.eye(self.num_slots, device=slots.device).unsqueeze(0)
        return ((sim * (1.0 - eye)) ** 2).sum() / max(
            slots.size(0) * self.num_slots * (self.num_slots - 1), 1
        )

    def _sample_same_prefix(self, codes: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return codes.new_empty((0, codes.size(1)))
        b = codes.size(0)
        rows = torch.randint(b, (count,), device=codes.device)
        prefixes = codes[rows, 0]
        counts = self.prefix_bucket_counts[prefixes]
        valid = counts > 1
        out = torch.empty((count, codes.size(1)), dtype=codes.dtype, device=codes.device)
        if valid.any():
            vcounts = counts[valid]
            offsets = (torch.rand(int(valid.sum()), device=codes.device) * vcounts).long()
            item_ids = self.prefix_bucket_ids[prefixes[valid], offsets]
            out[valid] = self.item_codes[item_ids]
        if (~valid).any():
            uniform_ids = torch.randint(
                self.item_codes.size(0), (int((~valid).sum()),), device=codes.device
            )
            out[~valid] = self.item_codes[uniform_ids]
        same = (out == codes[rows]).all(dim=1)
        if same.any():
            uniform_ids = torch.randint(
                self.item_codes.size(0), (int(same.sum()),), device=codes.device
            )
            out[same] = self.item_codes[uniform_ids]
        return out

    def _sample_same_prefix2(self, codes: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return codes.new_empty((0, codes.size(1)))
        b = codes.size(0)
        rows = torch.randint(b, (count,), device=codes.device)
        keys = codes[rows, 0] * self.num_codewords + codes[rows, 1]
        counts = self.pair_bucket_counts[keys]
        valid = counts > 1
        out = torch.empty((count, codes.size(1)), dtype=codes.dtype, device=codes.device)
        if valid.any():
            vcounts = counts[valid]
            offsets = (torch.rand(int(valid.sum()), device=codes.device) * vcounts).long()
            item_ids = self.pair_bucket_ids[keys[valid], offsets]
            out[valid] = self.item_codes[item_ids]
        if (~valid).any():
            # Singleton c0,c1 paths cannot provide a valid same-prefix2 negative.
            # Fall back to a safe same-prefix or uniform sample.
            out[~valid] = self._sample_same_prefix(
                codes[rows[~valid]], int((~valid).sum())
            )
        same = (out == codes[rows]).all(dim=1)
        if same.any():
            uniform_ids = torch.randint(
                self.item_codes.size(0), (int(same.sum()),), device=codes.device
            )
            out[same] = self.item_codes[uniform_ids]
        return out

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

        full_catalog_train = bool(
            getattr(self.cfg, "full_catalog_train", False)
        )
        if full_catalog_train:
            # Train the same B x M full-catalog score used at inference.
            # Targets are located by exact SID-code matching; no candidate
            # retrieval or reranking is introduced.
            candidate_codes = self.item_codes
            target_mask = (
                candidate_codes.unsqueeze(0) == codes.unsqueeze(1)
            ).all(dim=-1)
            target = target_mask.to(torch.float32).argmax(dim=1)
        else:
            negative_count = int(getattr(self.cfg, "expert_negatives", 1024))
            hard_count = int(
                getattr(self.cfg, "hard_prefix_negatives", negative_count)
            )
            candidate_parts = [codes]
            if negative_count > 0:
                sample_ids = torch.randint(
                    self.item_codes.size(0),
                    (negative_count,),
                    device=slots.device,
                )
                candidate_parts.append(self.item_codes[sample_ids])
            if hard_count > 0:
                candidate_parts.append(self._sample_same_prefix(codes, hard_count))
                candidate_parts.append(self._sample_same_prefix2(codes, hard_count))
            candidate_codes = torch.cat(candidate_parts, dim=0)
            target = torch.arange(slots.size(0), device=slots.device)

        recent_codes = None
        if self.recent_score_weight != 0.0:
            recent_codes = self._recent_item_codes(data[self.ISeq], slots.device)
        candidate_scores = self._aggregate_candidate_scores(
            slots, route_logits, position_logits, candidate_codes, recent_codes
        )
        rank_loss = F.cross_entropy(candidate_scores, target)

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
        recent_codes = None
        if self.recent_score_weight != 0.0:
            recent_codes = self._recent_item_codes(data[self.ISeq], slots.device)
        catalog_item_repr = None
        if self.semantic_score_weight > 0.0 or self.recent_score_weight != 0.0:
            catalog_item_repr = self._catalog_item_repr()
        return self._aggregate_candidate_scores(
            slots, route_logits, position_logits, self.item_codes, recent_codes,
            catalog_item_repr=catalog_item_repr,
        )
