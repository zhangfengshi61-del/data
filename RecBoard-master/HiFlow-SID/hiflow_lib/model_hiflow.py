"""HiFlow-SID-OneStep: intent-conditioned one-step flow decoding for SID rec.

Pipeline
--------
1. shared backbone encodes the user history once -> ``h``
2. prefix head predicts the coarse semantic prefix ``g = c1`` (Top-B)
3. behavior-conditioned prior maps ``(h, g)`` to a suffix latent ``z0``
4. a tiny flow expert runs one (or few) Euler step(s) to ``z1_hat``
5. ``z1_hat`` is projected onto the ``c2`` / ``c3`` codebooks
6. a joint item energy restores cross-position / item-level preferences

Scoring
-------
    score(i) = log p(g_i | h)
             + q2(c2(i) | z1_hat, g_i)
             + q3(c3(i) | z1_hat, g_i)
             + lambda_joint * E(h, SID(i))
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from hiflow_lib.common import mean_pool, tokenize_batch


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.to(self.mlp[0].weight.dtype).unsqueeze(-1))


class FlowExpert(nn.Module):
    """Small bidirectional transformer producing the suffix velocity field."""

    def __init__(
        self,
        dim: int,
        num_positions: int,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.pos_emb = nn.Embedding(num_positions, dim)
        self.time_emb = TimeEmbedding(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        g_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_pos, _ = z.shape
        pos = self.pos_emb(
            torch.arange(num_pos, device=z.device)
        )
        x = (
            z
            + pos.unsqueeze(0)
            + self.time_emb(t).unsqueeze(1)
            + h.unsqueeze(1)
            + g_emb.unsqueeze(1)
        )
        return self.layers(x)


class ChunkMLPExpert(nn.Module):
    """Vectorized low-level SID chunk expert.

    π0.5 predicts a complete low-level action chunk after the high-level
    subtask is known.  For SID decoding the chunk is the two suffix vectors;
    flattening those two positions lets a small MLP produce the whole chunk
    in one call and avoids a Transformer layer at inference time.
    """

    def __init__(self, dim: int, num_positions: int, hidden: int = 256) -> None:
        super().__init__()
        self.num_positions = num_positions
        self.dim = dim
        self.time_emb = TimeEmbedding(dim)
        self.net = nn.Sequential(
            nn.Linear(num_positions * dim + 3 * dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_positions * dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        g_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [z.reshape(z.size(0), -1), h, g_emb, self.time_emb(t)], dim=-1
        )
        return self.net(x).view(z.size(0), self.num_positions, self.dim)


class PrefixSuffixScoreExpert(nn.Module):
    """Low-rank joint score conditioned on user state and SID prefix.

    The expert scores a complete candidate SID in one vectorized operation.
    It does not generate tokens or run an iterative flow solver.
    """

    def __init__(self, dim: int, num_codewords: int, num_positions: int, rank: int, multiplicative: bool = False):
        super().__init__()
        self.rank = rank
        self.num_positions = num_positions
        self.multiplicative = bool(multiplicative)
        self.user_proj = nn.Linear(dim, rank, bias=False)
        self.prefix_proj = nn.Linear(dim, rank, bias=False)
        # Optional multiplicative interaction gives the expert a direct
        # user x semantic-prefix path. It remains a single vectorized
        # full-catalog score and adds no candidate-generation stage.
        if self.multiplicative:
            self.user_mul = nn.Linear(dim, rank, bias=False)
            self.prefix_mul = nn.Linear(dim, rank, bias=False)
            self.mul_gate = nn.Parameter(torch.tensor(0.05))
        # Start as a small residual.  A zero gate blocks gradients to the
        # expert at initialization (the score is multiplied by the gate),
        # which makes the low-rank branch learn only through a weak scalar
        # update.  A small non-zero gate keeps the parallel MTP prior stable
        # while allowing the expert factors to receive signal immediately.
        self.joint_gate = nn.Parameter(torch.tensor(0.05))
        self.suffix_factors = nn.ModuleList(
            [nn.Embedding(num_codewords, rank) for _ in range(num_positions - 1)]
        )
        for factor in self.suffix_factors:
            nn.init.normal_(factor.weight, mean=0.0, std=0.02)

    def score_candidates(
        self,
        h: torch.Tensor,
        candidate_codes: torch.Tensor,
        prefix_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # h: (B,D), candidate_codes: (M,L), output: (B,M)
        prefix_states = torch.tanh(
            self.user_proj(h).unsqueeze(1)
            + self.prefix_proj(prefix_embeddings).unsqueeze(0)
        )
        if self.multiplicative:
            user_factor = self.user_mul(h).unsqueeze(1)
            prefix_factor = self.prefix_mul(prefix_embeddings).unsqueeze(0)
            prefix_states = prefix_states + self.mul_gate * torch.tanh(
                user_factor * prefix_factor
            )
        candidate_prefix = candidate_codes[:, 0]
        conditioned = prefix_states[:, candidate_prefix, :]

        suffix_feature = self.suffix_factors[0](candidate_codes[:, 1])
        for offset, factor in enumerate(self.suffix_factors[1:], start=2):
            suffix_feature = suffix_feature * factor(candidate_codes[:, offset])
        return (conditioned * suffix_feature.unsqueeze(0)).sum(dim=-1) / (self.rank ** 0.5)


class FastHistoryEncoder(nn.Module):
    """SID-MLP++-style lightweight user encoder trained end to end.

    Each layer shares a masked global context across positions and applies a
    position-wise MLP.  It keeps the sequence-level context that a T5 encoder
    provides, but removes self-attention and its quadratic/token-wise overhead.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        hidden: int = 256,
        layers: int = 2,
        max_seq_len: int = 128,
        num_positions: int = 6,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.num_positions = num_positions
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(2 * dim, hidden),
                            nn.LayerNorm(hidden),
                            nn.SiLU(),
                            nn.Linear(hidden, dim),
                        )
                        for _ in range(num_positions)
                    ]
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.pos_emb.num_embeddings:
            raise ValueError(
                f"history token length {seq_len} exceeds fast encoder limit "
                f"{self.pos_emb.num_embeddings}"
            )
        pos = self.pos_emb(torch.arange(seq_len, device=input_ids.device))
        x = self.embedding(input_ids) + pos.unsqueeze(0)
        position_ids = torch.arange(seq_len, device=input_ids.device) % self.num_positions
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            context = (x * mask).sum(dim=1, keepdim=True) / denom
            block_input = torch.cat([x, context.expand_as(x)], dim=-1)
            update = torch.zeros_like(x)
            for position, expert in enumerate(block):
                update = update + expert(block_input) * (
                    position_ids == position
                ).view(1, -1, 1).to(x.dtype)
            x = x + update
        return self.norm(x)


class BehaviorPrior(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(dim, dim)
        self.logvar = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, g_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = torch.cat([h, g_emb], dim=-1)
        feat = self.net(feat)
        mu = self.mu(feat)
        logvar = self.logvar(feat).clamp(min=-3.0, max=3.0)
        return mu, logvar


class HiFlowRec(freerec.models.SeqRecArch):
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
        # The fast path receives already formatted SID protocol strings.  A
        # small vocabulary lookup avoids invoking the general T5 tokenizer on
        # every request; this preprocessing cost was larger than the MLP
        # encoder itself in the first strict benchmark.
        self._fast_token_to_id = tokenizer.get_vocab()
        self._fast_unk_id = int(tokenizer.unk_token_id or 0)
        self.code_of_tok = code_of_tok
        self.num_sid_tokens = len(code_of_tok)
        self.num_suffix_positions = self.num_sid_tokens - 1
        self.num_codewords = cfg.num_codewords
        self.dim = cfg.embedding_dim
        self.history_attention_pool = bool(getattr(cfg, "history_attention_pool", False))
        if self.history_attention_pool:
            self.history_pool_score = nn.Linear(self.dim, 1)
            self.history_pool_gate = nn.Parameter(torch.tensor(0.1))

        self.expert_score_mode = bool(getattr(cfg, "expert_score", False))
        # ExpertScore can run on the original TIGER T5 encoder. The fast
        # encoder remains the default for legacy runs, while expert_score_t5
        # explicitly selects the reusable TIGER encoder for the new decoder.
        self.expert_score_t5 = bool(getattr(cfg, "expert_score_t5", False))
        self.fast_encoder = bool(getattr(cfg, "fast_encoder", False)) or (
            self.expert_score_mode and not self.expert_score_t5
        )
        self.distill_encoder = bool(getattr(cfg, "distill_encoder", False))
        if self.fast_encoder:
            self.backbone = FastHistoryEncoder(
                vocab_size=len(self.tokenizer),
                dim=cfg.embedding_dim,
                hidden=int(getattr(cfg, "fast_hidden", cfg.intermediate_size)),
                layers=int(getattr(cfg, "fast_layers", 2)),
                max_seq_len=int(getattr(cfg, "fast_max_seq_len", 128)),
            )
        else:
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
        if self.fast_encoder and self.distill_encoder:
            teacher_config = T5Config(
                vocab_size=len(self.tokenizer),
                d_model=cfg.embedding_dim,
                d_kv=cfg.attention_size,
                d_ff=cfg.intermediate_size,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout_rate=cfg.dropout_rate,
            )
            self.teacher_backbone = T5EncoderModel(teacher_config)
            self.teacher_backbone.requires_grad_(False)
            self.teacher_backbone.eval()
            self.teacher_prefix_head = nn.Linear(self.dim, self.num_codewords)
            self.teacher_suffix_heads = nn.ModuleList(
                [
                    nn.Linear(self.dim, self.num_codewords)
                    for _ in range(self.num_suffix_positions)
                ]
            )
            self.teacher_prefix_head.requires_grad_(False)
            self.teacher_suffix_heads.requires_grad_(False)
        else:
            self.teacher_backbone = None
            self.teacher_prefix_head = None
            self.teacher_suffix_heads = None

        self.prefix_head = nn.Linear(self.dim, self.num_codewords)
        # The parallel SID heads provide the pretrained high-level semantic
        # prior; the flow expert models the conditional correction between
        # the two suffix positions.
        self.suffix_heads = nn.ModuleList(
            [
                nn.Linear(self.dim, self.num_codewords)
                for _ in range(self.num_suffix_positions)
            ]
        )

        self.code_embs = nn.ModuleList(
            [nn.Embedding(self.num_codewords, self.dim) for _ in range(self.num_sid_tokens)]
        )
        self.item_proj = nn.Linear(self.num_sid_tokens * self.dim, self.dim)
        self.h_proj = nn.Linear(self.dim, self.dim)

        if self.expert_score_mode:
            self.joint_energy_gate = nn.Parameter(torch.zeros(()))
            self.prior = None
            self.flow_expert = None
            self.score_expert = PrefixSuffixScoreExpert(
                dim=self.dim,
                num_codewords=self.num_codewords,
                num_positions=self.num_sid_tokens,
                rank=int(getattr(cfg, "expert_rank", 8)),
                multiplicative=bool(getattr(cfg, "expert_multiplicative", False)),
            )
        else:
            self.prior = BehaviorPrior(self.dim)
            self.score_expert = None
        if self.expert_score_mode:
            pass
        elif getattr(cfg, "expert_type", "transformer") == "mlp":
            self.flow_expert = ChunkMLPExpert(
                dim=self.dim,
                num_positions=self.num_suffix_positions,
                hidden=int(getattr(cfg, "expert_hidden", cfg.intermediate_size)),
            )
        else:
            self.flow_expert = FlowExpert(
                dim=self.dim,
                num_positions=self.num_suffix_positions,
                num_layers=cfg.expert_layers,
                num_heads=cfg.num_heads,
                d_ff=cfg.intermediate_size,
                dropout=cfg.dropout_rate,
            )

        self.register_buffer("item_codes", item_codes, persistent=False)
        # Prefix buckets support supervised hard negatives that share the
        # first SID code with the positive. They are only used during expert
        # training; inference still scores the complete catalog.
        item_codes_cpu = item_codes.detach().long().cpu()
        prefix_counts = torch.zeros(self.num_codewords, dtype=torch.long)
        max_prefix_bucket = 1
        for prefix in range(self.num_codewords):
            count = int((item_codes_cpu[:, 0] == prefix).sum().item())
            prefix_counts[prefix] = count
            max_prefix_bucket = max(max_prefix_bucket, count)
        prefix_bucket_ids = torch.zeros(
            self.num_codewords, max_prefix_bucket, dtype=torch.long
        )
        for prefix in range(self.num_codewords):
            ids = torch.nonzero(item_codes_cpu[:, 0] == prefix).flatten()
            if ids.numel():
                prefix_bucket_ids[prefix, :ids.numel()] = ids
        self.register_buffer("prefix_bucket_ids", prefix_bucket_ids, persistent=False)
        self.register_buffer("prefix_bucket_counts", prefix_counts, persistent=False)
        for level, table in enumerate(self.code_of_tok):
            self.register_buffer(f"code_of_tok_{level}", table, persistent=False)

    def _code_table(self, level: int) -> torch.Tensor:
        return getattr(self, f"code_of_tok_{level}")

    def format_item_ids(self, field, item_ids) -> List[str]:
        return [self.converter.format(item) for item in item_ids]

    def encode_item_text(self, field, items) -> str:
        return self.converter.encode(items)

    # ------------------------------------------------------------------ utils

    def _tokenize_fast(self, texts, device: torch.device) -> Dict[str, torch.Tensor]:
        rows = []
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            rows.append(
                [self._fast_token_to_id.get(tok, self._fast_unk_id) for tok in text.split()]
            )
        max_len = max((len(row) for row in rows), default=1)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        padded = [row + [pad_id] * (max_len - len(row)) for row in rows]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (max_len - len(row)) for row in rows],
            dtype=torch.long,
            device=device,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_history_sequence(self, data: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.fast_encoder:
            context = self._tokenize_fast(data[self.ISeq], device)
        else:
            context = tokenize_batch(self.tokenizer, device, data[self.ISeq])
        if self.fast_encoder:
            hidden = self.backbone(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
            )
        else:
            outs = self.backbone(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                return_dict=True,
            )
            hidden = outs.last_hidden_state
        return hidden, context["attention_mask"]

    def encode_history(self, data: Dict) -> torch.Tensor:
        hidden, attention_mask = self.encode_history_sequence(data)
        mean = mean_pool(hidden, attention_mask)
        if not self.history_attention_pool:
            return mean
        logits = self.history_pool_score(hidden).squeeze(-1)
        logits = logits.masked_fill(attention_mask == 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        attended = torch.sum(weights.unsqueeze(-1) * hidden, dim=1)
        return mean + self.history_pool_gate * attended

    def parse_target_codes(self, data: Dict) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        targets = tokenize_batch(self.tokenizer, device, data[self.IPos])
        ids = targets["input_ids"]
        return [
            self._code_table(level)[ids[:, level + 1]]
            for level in range(self.num_sid_tokens)
        ]

    def item_repr(self, codes: List[torch.Tensor]) -> torch.Tensor:
        embs = [emb(code) for emb, code in zip(self.code_embs, codes)]
        return self.item_proj(torch.cat(embs, dim=-1))

    def suffix_embeddings(self, codes: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [self.code_embs[m + 1](codes[m + 1]) for m in range(self.num_suffix_positions)],
            dim=1,
        )

    def prior_sample(
        self, h: torch.Tensor, g_emb: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        mu, logvar = self.prior(h, g_emb)
        mu = mu.unsqueeze(1).expand(-1, self.num_suffix_positions, -1)
        if deterministic:
            return mu
        sigma = (0.5 * logvar).exp().mul(self.cfg.prior_noise)
        sigma = sigma.unsqueeze(1).expand(-1, self.num_suffix_positions, -1)
        return mu + sigma * torch.randn_like(mu)

    def flow_forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        g_emb: torch.Tensor,
    ) -> torch.Tensor:
        return self.flow_expert(z, t, h, g_emb)

    def euler_solve(
        self,
        z0: torch.Tensor,
        h: torch.Tensor,
        g_emb: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        z = z0
        for k in range(steps):
            t = torch.full(
                (z.size(0),),
                k / steps,
                device=z.device,
                dtype=z.dtype,
            )
            z = z + self.flow_forward(z, t, h, g_emb) / steps
        return z

    def codebook_logits(self, z1: torch.Tensor) -> List[torch.Tensor]:
        logits = []
        for m in range(self.num_suffix_positions):
            emb = F.normalize(self.code_embs[m + 1].weight, dim=-1)
            vec = F.normalize(z1[:, m], dim=-1)
            logits.append(vec @ emb.t() / self.cfg.tau_code)
        return logits

    # ---------------------------------------------------------------- training

    def fit(self, data: Dict) -> Dict[str, torch.Tensor]:
        if self.expert_score_mode:
            return self.fit_expert_score(data)

        distill_loss = None
        logit_distill_loss = None
        teacher_h = None
        if self.teacher_backbone is not None:
            hidden, attention_mask = self.encode_history_sequence(data)
            h = mean_pool(hidden, attention_mask)
            context = self._tokenize_fast(
                data[self.ISeq], next(self.parameters()).device
            )
            self.teacher_backbone.eval()
            with torch.no_grad():
                teacher_hidden = self.teacher_backbone(
                    input_ids=context["input_ids"],
                    attention_mask=context["attention_mask"],
                    return_dict=True,
                ).last_hidden_state
                teacher_h = mean_pool(teacher_hidden, attention_mask)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            distill_loss = (
                ((hidden - teacher_hidden.detach()) ** 2) * mask
            ).sum() / (mask.sum().clamp_min(1.0) * hidden.size(-1))
        else:
            h = self.encode_history(data)
        codes = self.parse_target_codes(data)
        batch = h.size(0)
        device = h.device

        prefix_logits = self.prefix_head(h)
        prefix_loss = F.cross_entropy(prefix_logits, codes[0])
        parallel_suffix_logits = [head(h) for head in self.suffix_heads]
        parallel_suffix_loss = sum(
            F.cross_entropy(parallel_suffix_logits[m], codes[m + 1])
            for m in range(self.num_suffix_positions)
        )
        if teacher_h is not None:
            with torch.no_grad():
                teacher_prefix_logits = self.teacher_prefix_head(teacher_h)
                teacher_suffix_logits = [head(teacher_h) for head in self.teacher_suffix_heads]
            kd_temperature = float(getattr(self.cfg, "kd_temperature", 2.0))
            logit_distill_loss = F.kl_div(
                F.log_softmax(prefix_logits / kd_temperature, dim=-1),
                F.softmax(teacher_prefix_logits / kd_temperature, dim=-1),
                reduction="batchmean",
            ) * (kd_temperature ** 2)
            for student_logits, teacher_logits in zip(
                parallel_suffix_logits, teacher_suffix_logits
            ):
                logit_distill_loss = logit_distill_loss + F.kl_div(
                    F.log_softmax(student_logits / kd_temperature, dim=-1),
                    F.softmax(teacher_logits / kd_temperature, dim=-1),
                    reduction="batchmean",
                ) * (kd_temperature ** 2)

        # π0.5-style conditioning: feed the high-level semantic prediction
        # into the low-level suffix expert.  This optional differentiable
        # path keeps training and one-expert inference on the same interface.
        if getattr(self.cfg, "train_soft_prefix", False):
            temperature = max(float(getattr(self.cfg, "prefix_temperature", 1.0)), 1e-4)
            prefix_prob = F.softmax(prefix_logits / temperature, dim=-1)
            g_emb = prefix_prob @ self.code_embs[0].weight
        else:
            g_emb = self.code_embs[0](codes[0])
        z1 = self.suffix_embeddings(codes)  # (B, P, d)

        z0 = self.prior_sample(h, g_emb, deterministic=False)

        t = torch.rand(batch, device=device, dtype=h.dtype)
        z_t = (1 - t).view(-1, 1, 1) * z0 + t.view(-1, 1, 1) * z1
        v = self.flow_forward(z_t, t, h, g_emb)
        fm_loss = ((v - (z1 - z0)) ** 2).mean()

        z1_hat = z0 + self.flow_forward(
            z0,
            torch.zeros(batch, device=device, dtype=h.dtype),
            h,
            g_emb,
        )
        q_logits = self.codebook_logits(z1_hat)
        code_loss = sum(
            F.cross_entropy(q_logits[m], codes[m + 1])
            for m in range(self.num_suffix_positions)
        )

        h_proj = F.normalize(self.h_proj(h), dim=-1)
        item_rep = F.normalize(self.item_repr(codes), dim=-1)
        joint_logits = h_proj @ item_rep.t() / self.cfg.tau_joint
        joint_loss = F.cross_entropy(
            joint_logits, torch.arange(batch, device=device)
        )

        rec_loss = (
            self.cfg.lam_fm * fm_loss
            + self.cfg.lam_code * code_loss
            + self.cfg.lam_joint * joint_loss
            + self.cfg.lam_prefix * prefix_loss
            + float(getattr(self.cfg, "lam_parallel", 1.0)) * parallel_suffix_loss
        )
        if distill_loss is not None:
            rec_loss = rec_loss + float(getattr(self.cfg, "lam_distill", 1.0)) * distill_loss
        if logit_distill_loss is not None:
            rec_loss = rec_loss + float(getattr(self.cfg, "lam_logit_distill", 0.0)) * logit_distill_loss
        return {
            "rec_loss": rec_loss,
            "fm_loss": fm_loss,
            "code_loss": code_loss,
            "joint_loss": joint_loss,
            "prefix_loss": prefix_loss,
            "parallel_suffix_loss": parallel_suffix_loss,
            **({"distill_loss": distill_loss} if distill_loss is not None else {}),
            **({"logit_distill_loss": logit_distill_loss} if logit_distill_loss is not None else {}),
        }

    def _parallel_sid_logits(self, h: torch.Tensor) -> List[torch.Tensor]:
        return [self.prefix_head(h)] + [head(h) for head in self.suffix_heads]

    def score_sid_candidates(
        self,
        h: torch.Tensor,
        position_logits: List[torch.Tensor],
        candidate_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch of catalog SIDs for every user in h."""
        scores = h.new_zeros((h.size(0), candidate_codes.size(0)))
        for position, logits in enumerate(position_logits):
            log_probs = F.log_softmax(logits, dim=-1)
            scores = scores + log_probs[:, candidate_codes[:, position]]
        expert_score = self.score_expert.score_candidates(
            h, candidate_codes, self.code_embs[0].weight
        )
        gated_expert_score = self.score_expert.joint_gate * expert_score
        scores = scores + float(getattr(self.cfg, "expert_score_scale", 1.0)) * gated_expert_score
        if bool(getattr(self.cfg, "use_joint", True)):
            candidate_rep = F.normalize(
                self.item_repr(
                    [candidate_codes[:, position] for position in range(self.num_sid_tokens)]
                ),
                dim=-1,
            )
            user_rep = F.normalize(self.h_proj(h), dim=-1)
            energy = user_rep @ candidate_rep.t() / float(getattr(self.cfg, "tau_joint", 0.1))
            scores = scores + (
                float(getattr(self.cfg, "lam_joint_score", 0.1))
                * self.joint_energy_gate
                * energy
            )
        return scores

    def fit_expert_score(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Train the no-distillation model with direct interaction labels."""
        distill_loss = None
        logit_distill_loss = None
        teacher_h = None
        if self.teacher_backbone is not None:
            # Keep the fast encoder as the only input to all recommendation
            # heads, while matching the frozen MTP/T5 representation during
            # training.  This removes the random-feature cold start that
            # otherwise causes the loaded MTP heads to be unusable.
            hidden, attention_mask = self.encode_history_sequence(data)
            h = mean_pool(hidden, attention_mask)
            context = self._tokenize_fast(
                data[self.ISeq], next(self.parameters()).device
            )
            self.teacher_backbone.eval()
            with torch.no_grad():
                teacher_hidden = self.teacher_backbone(
                    input_ids=context["input_ids"],
                    attention_mask=context["attention_mask"],
                    return_dict=True,
                ).last_hidden_state
                teacher_h = mean_pool(teacher_hidden, attention_mask)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            distill_loss = (
                ((hidden - teacher_hidden.detach()) ** 2) * mask
            ).sum() / (mask.sum().clamp_min(1.0) * hidden.size(-1))
        else:
            h = self.encode_history(data)
        codes = torch.stack(self.parse_target_codes(data), dim=1)
        position_logits = self._parallel_sid_logits(h)
        prefix_loss = F.cross_entropy(position_logits[0], codes[:, 0])
        parallel_loss = sum(
            F.cross_entropy(position_logits[position], codes[:, position])
            for position in range(1, self.num_sid_tokens)
        )

        # Keep the in-batch positives first so their diagonal labels remain
        # valid, then add sampled catalog SIDs. This provides harder negatives
        # than a batch-only contrastive loss without materializing the full
        # catalogue for every optimization step.
        negative_count = int(getattr(self.cfg, "expert_negatives", 2048))
        candidate_parts = [codes]
        if negative_count > 0 and self.item_codes.size(0) > 0:
            sample_ids = torch.randint(
                self.item_codes.size(0), (negative_count,), device=h.device
            )
            candidate_parts.append(self.item_codes[sample_ids])
        hard_count = int(getattr(self.cfg, "hard_prefix_negatives", 0))
        if hard_count > 0 and self.item_codes.size(0) > 0:
            rows = torch.randint(h.size(0), (hard_count,), device=h.device)
            prefixes = codes[rows, 0]
            counts = self.prefix_bucket_counts[prefixes].clamp_min(1)
            offsets = (torch.rand(hard_count, device=h.device) * counts).long()
            hard_ids = self.prefix_bucket_ids[prefixes, offsets]
            hard_codes = self.item_codes[hard_ids]
            candidate_parts.append(hard_codes)
        candidate_codes = torch.cat(candidate_parts, dim=0)
        candidate_scores = self.score_sid_candidates(h, position_logits, candidate_codes)
        labels = torch.arange(h.size(0), device=h.device)
        expert_rank_loss = F.cross_entropy(candidate_scores, labels)
        # Keep the same direct item-level alignment signal as MTP.  This is
        # trained from observed interactions and code embeddings only; it is
        # not a teacher-logit or hidden-state distillation term.
        h_proj = F.normalize(self.h_proj(h), dim=-1)
        item_rep = F.normalize(self.item_repr([codes[:, m] for m in range(self.num_sid_tokens)]), dim=-1)
        align_logits = h_proj @ item_rep.t() / float(getattr(self.cfg, "tau_align", 0.1))
        align_loss = F.cross_entropy(align_logits, labels)
        if teacher_h is not None:
            with torch.no_grad():
                teacher_prefix_logits = self.teacher_prefix_head(teacher_h)
                teacher_suffix_logits = [
                    head(teacher_h) for head in self.teacher_suffix_heads
                ]
            kd_temperature = float(getattr(self.cfg, "kd_temperature", 2.0))
            logit_distill_loss = F.kl_div(
                F.log_softmax(position_logits[0] / kd_temperature, dim=-1),
                F.softmax(teacher_prefix_logits / kd_temperature, dim=-1),
                reduction="batchmean",
            ) * (kd_temperature ** 2)
            for student_logits, teacher_logits in zip(
                position_logits[1:], teacher_suffix_logits
            ):
                logit_distill_loss = logit_distill_loss + F.kl_div(
                    F.log_softmax(student_logits / kd_temperature, dim=-1),
                    F.softmax(teacher_logits / kd_temperature, dim=-1),
                    reduction="batchmean",
                ) * (kd_temperature ** 2)
        rec_loss = (
            float(getattr(self.cfg, "lam_prefix", 0.5)) * prefix_loss
            + float(getattr(self.cfg, "lam_parallel", 1.0)) * parallel_loss
            + float(getattr(self.cfg, "lam_expert_rank", 1.0)) * expert_rank_loss
            + float(getattr(self.cfg, "lam_align", 0.1)) * align_loss
        )
        if distill_loss is not None:
            rec_loss = rec_loss + float(getattr(self.cfg, "lam_distill", 1.0)) * distill_loss
        if logit_distill_loss is not None:
            rec_loss = rec_loss + float(getattr(self.cfg, "lam_logit_distill", 0.0)) * logit_distill_loss
        return {
            "rec_loss": rec_loss,
            "prefix_loss": prefix_loss,
            "parallel_suffix_loss": parallel_loss,
            "expert_rank_loss": expert_rank_loss,
            "align_loss": align_loss,
            **({"distill_loss": distill_loss} if distill_loss is not None else {}),
            **({"logit_distill_loss": logit_distill_loss} if logit_distill_loss is not None else {}),
        }

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def score_batch_expert(self, h: torch.Tensor, position_logits: List[torch.Tensor]) -> torch.Tensor:
        scores = self.score_sid_candidates(h, position_logits, self.item_codes)
        # The item alignment head is trained jointly with the SID heads. The
        # previous ExpertScore serving path ignored it even when use_joint was
        # enabled, discarding a useful item-level residual.
        if bool(getattr(self.cfg, "use_joint", True)):
            item_rep = F.normalize(
                self.item_repr([self.item_codes[:, m] for m in range(self.num_sid_tokens)]),
                dim=-1,
            )
            h_proj = F.normalize(self.h_proj(h), dim=-1)
            energy = h_proj @ item_rep.t() / float(self.cfg.tau_joint)
            scores = scores + float(getattr(self.cfg, "lam_joint_score", 0.1)) * energy
        return scores

    @torch.no_grad()
    def score_batch_single_expert(
        self,
        h: torch.Tensor,
        prefix_logits: torch.Tensor,
        flow_steps: Optional[int] = None,
        prefix_mode: Optional[str] = None,
        use_prior: Optional[bool] = None,
        use_joint: Optional[bool] = None,
        energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score the full catalogue with one semantic-conditioned expert.

        This is the direct π0.5-inspired path: predict one high-level
        semantic prefix, then run the small low-level expert once on the
        complete suffix chunk.  It removes the old ``B * top_b`` branch
        expansion while retaining per-item prefix likelihoods in the final
        catalogue score.
        """
        flow_steps = flow_steps if flow_steps is not None else self.cfg.flow_steps
        prefix_mode = prefix_mode or getattr(self.cfg, "prefix_mode", "soft")
        use_prior = use_prior if use_prior is not None else self.cfg.use_prior
        use_joint = use_joint if use_joint is not None else self.cfg.use_joint

        batch = h.size(0)
        device = h.device
        codes = self.item_codes

        logp_prefix = F.log_softmax(prefix_logits, dim=-1)
        if prefix_mode == "argmax":
            branch = prefix_logits.argmax(dim=-1)
            g_emb = self.code_embs[0](branch)
        elif prefix_mode == "soft":
            temperature = max(float(getattr(self.cfg, "prefix_temperature", 1.0)), 1e-4)
            prefix_prob = F.softmax(prefix_logits / temperature, dim=-1)
            g_emb = prefix_prob @ self.code_embs[0].weight
        else:
            raise ValueError(f"unknown prefix_mode={prefix_mode!r}; use 'soft' or 'argmax'")

        if use_prior:
            z0 = self.prior_sample(h, g_emb, deterministic=True)
        else:
            z0 = torch.randn(
                batch,
                self.num_suffix_positions,
                self.dim,
                device=device,
                dtype=h.dtype,
            ).mul_(self.cfg.prior_noise)
        z1_hat = self.euler_solve(z0, h, g_emb, steps=flow_steps)
        q_logits = self.codebook_logits(z1_hat)
        parallel_suffix_logits = [head(h) for head in self.suffix_heads]

        scores = logp_prefix[:, codes[:, 0]]
        for suffix_pos, q in enumerate(q_logits, start=1):
            base_logp = F.log_softmax(
                parallel_suffix_logits[suffix_pos - 1], dim=-1
            )
            scores = (
                scores
                + base_logp[:, codes[:, suffix_pos]]
                + q[:, codes[:, suffix_pos]]
            )

        if use_joint:
            if energy is None:
                item_rep_all = F.normalize(
                    self.item_repr(
                        [codes[:, m] for m in range(self.num_sid_tokens)]
                    ),
                    dim=-1,
                )
                h_proj = F.normalize(self.h_proj(h), dim=-1)
                energy = h_proj @ item_rep_all.t() / self.cfg.tau_joint
            scores = scores + self.cfg.lam_joint_score * energy
        return scores

    @torch.no_grad()
    def score_batch(
        self,
        h: torch.Tensor,
        prefix_logits: torch.Tensor,
        flow_steps: Optional[int] = None,
        top_b: Optional[int] = None,
        use_prior: Optional[bool] = None,
        use_joint: Optional[bool] = None,
        energy: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score every item for a batch of user states.

        ``energy`` optionally injects a precomputed ``(B, N)`` joint energy
        matrix; when ``None`` it is computed from the current parameters.
        """
        flow_steps = flow_steps if flow_steps is not None else self.cfg.flow_steps
        top_b = top_b if top_b is not None else self.cfg.top_b
        use_prior = use_prior if use_prior is not None else self.cfg.use_prior
        use_joint = use_joint if use_joint is not None else self.cfg.use_joint

        batch, num_items = h.size(0), self.item_codes.size(0)
        device = h.device

        logp_prefix = F.log_softmax(prefix_logits, dim=-1)  # (B, V0)
        codes = self.item_codes  # (N, L)

        use_all_branches = top_b <= 0 or top_b >= self.num_codewords
        if use_all_branches:
            branch_ids = torch.arange(
                self.num_codewords, device=device
            ).unsqueeze(0).expand(batch, -1)
        else:
            branch_ids = prefix_logits.topk(top_b, dim=-1).indices  # (B, K)
        K = branch_ids.size(1)

        g_emb = self.code_embs[0](branch_ids)  # (B, K, d)
        h_k = h.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, self.dim)
        g_emb = g_emb.reshape(batch * K, self.dim)

        if use_prior:
            z0 = self.prior_sample(h_k, g_emb, deterministic=True)
        else:
            z0 = torch.randn(
                batch * K, self.num_suffix_positions, self.dim,
                device=device, dtype=h.dtype,
            ).mul_(self.cfg.prior_noise)

        z1_hat = self.euler_solve(z0, h_k, g_emb, steps=flow_steps)
        q_logits = self.codebook_logits(z1_hat)  # list of (B*K, V)
        q2 = q_logits[0].view(batch, K, self.num_codewords)
        q3 = q_logits[1].view(batch, K, self.num_codewords)

        # per-item branch index: branch b = c1(i); or -1 when pruned
        branch_of_item = torch.full(
            (batch, self.num_codewords), -1, device=device, dtype=torch.long
        )
        branch_of_item.scatter_(
            1,
            branch_ids,
            torch.arange(K, device=device).unsqueeze(0).expand(batch, -1),
        )
        item_branch = branch_of_item.gather(1, codes[:, 0].unsqueeze(0).expand(batch, -1))
        valid = item_branch >= 0

        # suffix scores: q2 / q3 share the flat layout (branch, code)
        branch_clamped = item_branch.clamp(min=0)
        q2_flat = q2.reshape(batch, K * self.num_codewords)
        q3_flat = q3.reshape(batch, K * self.num_codewords)
        idx2 = branch_clamped * self.num_codewords + codes[:, 1].unsqueeze(0)
        idx3 = branch_clamped * self.num_codewords + codes[:, 2].unsqueeze(0)
        suffix_score = torch.full(
            (batch, num_items), float("-inf"), device=device, dtype=h.dtype
        )
        suffix_score[valid] = (
            q2_flat.gather(1, idx2)[valid] + q3_flat.gather(1, idx3)[valid]
        )

        prefix_score = torch.full(
            (batch, num_items), float("-inf"), device=device, dtype=h.dtype
        )
        if use_all_branches:
            prefix_score = logp_prefix[:, codes[:, 0]]
        else:
            prefix_score[valid] = logp_prefix.gather(1, branch_clamped)[valid]

        scores = prefix_score + suffix_score

        if use_joint:
            if energy is None:
                item_rep_all = F.normalize(
                    self.item_repr(
                        [codes[:, m] for m in range(self.num_sid_tokens)]
                    ),
                    dim=-1,
                )
                h_proj = F.normalize(self.h_proj(h), dim=-1)
                energy = h_proj @ item_rep_all.t() / self.cfg.tau_joint
            scores = scores + self.cfg.lam_joint_score * energy

        return scores

    @torch.no_grad()
    def recommend_from_full(self, data: Dict) -> torch.Tensor:
        h = self.encode_history(data)
        if self.expert_score_mode:
            return self.score_batch_expert(h, self._parallel_sid_logits(h))
        prefix_logits = self.prefix_head(h)
        if getattr(self.cfg, "single_expert", False):
            return self.score_batch_single_expert(h, prefix_logits)
        return self.score_batch(h, prefix_logits)
