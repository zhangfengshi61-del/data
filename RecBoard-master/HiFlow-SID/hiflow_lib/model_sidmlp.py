"""SID-MLP decoder adapted to the RecBoard Beauty protocol.

The implementation keeps the paper's defining path (a frozen TIGER/T5
encoder, one context readout, and prefix-conditioned MLP heads) while using
RecBoard's three-token SDQ SID.  It is intentionally independent of the
π-SID implementation so the two methods can be evaluated through the same
RecBoard full-ranking evaluator.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import freerec
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Config, T5EncoderModel, T5Tokenizer

from hiflow_lib.common import mean_pool, tokenize_batch


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_dim: int = 0) -> None:
        super().__init__()
        inner = attn_dim or d_model
        if inner % num_heads:
            raise ValueError("attn_dim must be divisible by num_heads")
        self.inner = inner
        self.num_heads = num_heads
        self.head_dim = inner // num_heads
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.out_proj = nn.Linear(inner, d_model, bias=False)

    def precompute_kv(self, hidden: torch.Tensor):
        b, n, _ = hidden.shape
        k = self.k_proj(hidden).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(self, query: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        b = query.size(0)
        q = self.q_proj(query).view(b, self.num_heads, 1, self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        any_valid = torch.isfinite(scores).any(dim=-1, keepdim=True)
        scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        probs = F.softmax(scores, dim=-1)
        probs = torch.where(any_valid, probs, torch.zeros_like(probs))
        out = torch.matmul(probs, v).transpose(1, 2).reshape(b, 1, self.inner)
        return self.out_proj(out).squeeze(1)


def make_mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Sequential:
    blocks: List[nn.Module] = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
    for _ in range(max(0, layers - 1)):
        blocks += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU()]
    blocks.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*blocks)


class SIDMLPRec(freerec.models.SeqRecArch):
    """RecBoard-compatible three-digit SID-MLP."""

    def __init__(self, dataset, cfg, tokenizer: T5Tokenizer, converter,
                 code_of_tok: List[torch.Tensor], item_codes: torch.Tensor):
        super().__init__(dataset)
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.converter = converter
        self.code_of_tok = code_of_tok
        self.num_sid_tokens = len(code_of_tok)
        self.num_codewords = cfg.num_codewords
        self.dim = cfg.embedding_dim

        t5_cfg = T5Config(
            vocab_size=len(tokenizer), d_model=cfg.embedding_dim,
            d_kv=cfg.attention_size, d_ff=cfg.intermediate_size,
            num_layers=cfg.num_layers, num_heads=cfg.num_heads,
            dropout_rate=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.backbone = T5EncoderModel(t5_cfg)
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.cross_attn = CrossAttention(cfg.embedding_dim, cfg.num_heads, cfg.attn_dim)
        self.norm1 = nn.LayerNorm(cfg.embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embedding_dim, cfg.ffn_dim), nn.ReLU(),
            nn.Linear(cfg.ffn_dim, cfg.embedding_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.embedding_dim)
        self.query_proj = nn.Linear(cfg.embedding_dim, cfg.embedding_dim)
        self.sid_embeddings = nn.Embedding(self.num_sid_tokens * self.num_codewords, cfg.embedding_dim)
        self.heads = nn.ModuleList([
            make_mlp(cfg.embedding_dim + level * cfg.embedding_dim,
                     cfg.num_codewords, cfg.head_hidden, cfg.head_layers)
            for level in range(self.num_sid_tokens)
        ])
        self.register_buffer("item_codes", item_codes, persistent=False)
        for level, table in enumerate(code_of_tok):
            self.register_buffer(f"code_of_tok_{level}", table, persistent=False)

    def _code_table(self, level: int) -> torch.Tensor:
        return getattr(self, f"code_of_tok_{level}")

    def format_item_ids(self, field, item_ids: Iterable[int]) -> List[str]:
        return [self.converter.format(item) for item in item_ids]

    def encode_item_text(self, field, items: Iterable[str]) -> str:
        return self.converter.encode(items)

    def encode_context(self, data: Dict) -> torch.Tensor:
        device = next(self.parameters()).device
        tokenized = tokenize_batch(self.tokenizer, device, data[self.ISeq])
        with torch.no_grad():
            hidden = self.backbone(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                return_dict=True,
            ).last_hidden_state
        mask = tokenized["attention_mask"]
        pooled = mean_pool(hidden, mask)
        query = self.query_proj(pooled)
        k, v = self.cross_attn.precompute_kv(hidden)
        ctx = self.norm1(query + self.cross_attn(query, k, v, ~mask.bool()))
        return self.norm2(ctx + self.ffn(ctx))

    def parse_target_codes(self, data: Dict) -> torch.Tensor:
        device = next(self.parameters()).device
        ids = tokenize_batch(self.tokenizer, device, data[self.IPos])["input_ids"]
        return torch.stack([
            self._code_table(level)[ids[:, level + 1]]
            for level in range(self.num_sid_tokens)
        ], dim=1)

    def _emb(self, level: int, codes: torch.Tensor) -> torch.Tensor:
        return self.sid_embeddings(codes + level * self.num_codewords)

    def teacher_forced_logits(self, ctx: torch.Tensor, codes: torch.Tensor):
        prefixes: List[torch.Tensor] = []
        for level in range(self.num_sid_tokens):
            if level == 0:
                inp = ctx
            else:
                inp = torch.cat([ctx] + [self._emb(i, codes[:, i]) for i in range(level)], dim=-1)
            prefixes.append(self.heads[level](inp))
        return prefixes

    def fit(self, data: Dict) -> Dict[str, torch.Tensor]:
        ctx = self.encode_context(data)
        codes = self.parse_target_codes(data)
        logits = self.teacher_forced_logits(ctx, codes)
        ce = sum(F.cross_entropy(logits[i], codes[:, i]) for i in range(self.num_sid_tokens))

        # The same in-batch item alignment used by the RecBoard MTP baseline
        # stabilizes ranking while the conditional heads learn suffix effects.
        user = F.normalize(self.query_proj(ctx), dim=-1)
        item = F.normalize(torch.cat([self._emb(i, codes[:, i]) for i in range(self.num_sid_tokens)], dim=-1), dim=-1)
        # Use a cheap projection because concatenated code embeddings are 3D.
        item = item[:, :self.dim]
        align = F.cross_entropy(user @ item.t() / self.cfg.tau_align,
                                torch.arange(ctx.size(0), device=ctx.device))
        loss = ce + self.cfg.lam_align * align
        return {"rec_loss": loss, "ce_loss": ce, "align_loss": align}

    def _candidate_scores(self, ctx: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # Chunked exact candidate scoring avoids materializing B*N*hidden.
        b, n = ctx.size(0), codes.size(0)
        out = ctx.new_empty((b, n))
        logp1 = F.log_softmax(self.heads[0](ctx), dim=-1)
        chunk = int(getattr(self.cfg, "candidate_chunk", 2048))
        for start in range(0, n, chunk):
            c = codes[start:start + chunk]
            m = c.size(0)
            c1 = self._emb(0, c[:, 0]).unsqueeze(0).expand(b, -1, -1)
            x2 = torch.cat([ctx.unsqueeze(1).expand(-1, m, -1), c1], dim=-1)
            l2 = self.heads[1](x2.reshape(b * m, -1)).view(b, m, -1)
            c2 = self._emb(1, c[:, 1]).unsqueeze(0).expand(b, -1, -1)
            x3 = torch.cat([ctx.unsqueeze(1).expand(-1, m, -1), c1, c2], dim=-1)
            l3 = self.heads[2](x3.reshape(b * m, -1)).view(b, m, -1)
            out[:, start:start + m] = (
                logp1[:, c[:, 0]]
                + F.log_softmax(l2, dim=-1).gather(2, c[:, 1].view(1, m, 1).expand(b, -1, 1)).squeeze(-1)
                + F.log_softmax(l3, dim=-1).gather(2, c[:, 2].view(1, m, 1).expand(b, -1, 1)).squeeze(-1)
            )
        return out

    @torch.no_grad()
    def recommend_from_full(self, data: Dict) -> torch.Tensor:
        ctx = self.encode_context(data)
        return self._candidate_scores(ctx, self.item_codes)

