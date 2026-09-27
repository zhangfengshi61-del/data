"""Shared helpers for the HiFlow-SID pipeline.

Protocol (RecBoard-master comparison contract):
  - Amazon Review 2014 5-core (Beauty / Sports / Toys)
  - chronological per-user leave-two-out, max history 20
  - full-ranking evaluation, seen items masked
  - SID: the SDQ-VAE 3-token SID used by the TIGER-style T5 baseline
    (``<sid_0_*>`` coarse prefix, ``<sid_1_*>`` / ``<sid_2_*>`` fine suffix)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import freerec
import torch
from transformers import T5Tokenizer

HiFlowSID_ROOT = Path(__file__).resolve().parents[1]
SDQ_ROOT = HiFlowSID_ROOT.parent / "SDQ-403"

if str(SDQ_ROOT) not in sys.path:
    sys.path.insert(0, str(SDQ_ROOT))

from converter import SemIDConverter  # noqa: E402

SID_TOKEN_PATTERN = re.compile(r"^<sid_(?P<level>[0-9]+)_(?P<code>[0-9]+)>$")

DATASET_OF = {
    "Beauty": "Amazon2014Beauty_550_LOU",
    "Sports": "Amazon2014Sports_550_LOU",
    "Toys": "Amazon2014Toys_550_LOU",
}


def sid_vocab_path(dataset: str) -> Path:
    ds = DATASET_OF[dataset]
    return SDQ_ROOT / "logs" / "SDQ" / ds / "vae" / "sid_vocab.json"


def data_root() -> Path:
    return SDQ_ROOT / "data"


def build_dataset(dataset: str) -> "freerec.data.datasets.RecDataSet":
    name = DATASET_OF[dataset]
    cls = getattr(freerec.data.datasets, name, None)
    if cls is not None:
        return cls(root=str(data_root()))
    return freerec.data.datasets.RecDataSet(
        str(data_root()), name, tasktag=freerec.data.tags.TaskTags.NEXTITEM
    )


def item_count(dataset: "freerec.data.datasets.RecDataSet") -> int:
    return dataset.fields[freerec.data.tags.ITEM, freerec.data.tags.ID].count


def make_tokenizer_converter(
    sid_vocab_file: str,
    item_count: int,
) -> Tuple[T5Tokenizer, SemIDConverter, List[torch.Tensor], torch.Tensor]:
    """Build the SID tokenizer/converter plus code lookup tables.

    Returns
    -------
    tokenizer : T5Tokenizer
    converter : SemIDConverter
    code_of_tok : List[torch.Tensor]
        ``code_of_tok[level]`` maps a token id to its code index
        (``-1`` when the token does not belong to that level).
    item_codes : torch.Tensor
        Long tensor of shape ``(item_count, num_sid_tokens)``.
    """
    with open(sid_vocab_file, "r", encoding="utf-8") as file:
        raw_vocab: Dict[str, List[str]] = json.load(file)

    tokenizer = T5Tokenizer(vocab=None, extra_ids=0)
    converter = SemIDConverter(raw_vocab, tokenizer)
    num_sid_tokens = 3  # the SDQ-VAE SID always has exactly 3 levels

    code_of_tok: List[torch.Tensor] = []
    for level in range(num_sid_tokens):
        table = torch.full(
            (len(tokenizer),), -1, dtype=torch.long
        )
        for token_str, token_id in tokenizer.get_vocab().items():
            match = SID_TOKEN_PATTERN.fullmatch(token_str)
            if match is None or int(match.group("level")) != level:
                continue
            table[token_id] = int(match.group("code"))
        code_of_tok.append(table)

    item_codes = torch.full(
        (item_count, num_sid_tokens), -1, dtype=torch.long
    )
    for item_key, sid_tokens in raw_vocab.items():
        prefix, raw_id = SemIDConverter.parse(item_key)
        if prefix != "item" or raw_id >= item_count:
            continue
        codes = []
        for token_str in sid_tokens:
            match = SID_TOKEN_PATTERN.fullmatch(token_str)
            if match is None:
                continue
            codes.append(int(match.group("code")))
        if len(codes) != num_sid_tokens:
            raise ValueError(f"unexpected SID length for {item_key}: {sid_tokens}")
        item_codes[raw_id] = torch.tensor(codes, dtype=torch.long)

    assert (item_codes >= 0).all(), "some items are missing SID codes"
    return tokenizer, converter, code_of_tok, item_codes


def tokenize_batch(
    tokenizer: T5Tokenizer,
    device: torch.device,
    texts: Iterable[str],
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
    }


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
