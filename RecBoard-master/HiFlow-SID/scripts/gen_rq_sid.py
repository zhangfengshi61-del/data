"""Generate sid_vocab.json from a TIGER RQ-VAE checkpoint (protocol-aligned).

Item alignment: tiger_impl item_emb.parquet rows are ordered by ItemID
1..N, and the protocol item id i (0-based) is asin -> (1-based id) - 1,
verified identical item_mapping.json between tiger_impl and the protocol
canonical data. So codes row i <-> protocol item i.

Usage:
    python gen_rq_sid.py <Beauty|Sports|Toys> <ckpt_path> <out_json> [item_emb_parquet]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

TIGER_IMPL = Path("/data/fszhang/tiger协同语义训练/experiments/tiger_impl/rqvae")
sys.path.insert(0, str(TIGER_IMPL))

from datasets import EmbDataset  # noqa: E402
from models.rqvae import RQVAE  # noqa: E402

CAT = sys.argv[1] if len(sys.argv) > 1 else "Beauty"
CKPT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_JSON = Path(sys.argv[3]) if len(sys.argv) > 3 else None
EMB_PARQUET = (
    Path(sys.argv[4])
    if len(sys.argv) > 4
    else TIGER_IMPL.parent / "data" / CAT / "item_emb.parquet"
)

if CKPT_PATH is None:
    CKPT_PATH = sorted((TIGER_IMPL / "ckpt" / CAT).glob("*/best_collision_model.pth"))[-1]
if OUT_JSON is None:
    OUT_JSON = Path("/data/fszhang/RecBoard-master/HiFlow-SID/logs") / f"sid_vocab_rqvae_{CAT}.json"

device = torch.device("cuda:0")

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
args = ckpt["args"]
print(f"ckpt: {CKPT_PATH}", flush=True)
print(f"data_path: {args.data_path}", flush=True)

data = EmbDataset(str(EMB_PARQUET))
print(f"items: {len(data)}, dim: {data.dim}", flush=True)

model = RQVAE(
    in_dim=data.dim,
    num_emb_list=args.num_emb_list,
    e_dim=args.e_dim,
    layers=args.layers,
    dropout_prob=args.dropout_prob,
    bn=args.bn,
    loss_type=args.loss_type,
    quant_loss_weight=args.quant_loss_weight,
    kmeans_init=args.kmeans_init,
    kmeans_iters=args.kmeans_iters,
    sk_epsilons=args.sk_epsilons,
    sk_iters=args.sk_iters,
)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

loader = torch.utils.data.DataLoader(
    data, num_workers=4, batch_size=256, shuffle=False, pin_memory=True
)

all_codes: list[list[int]] = [[] for _ in range(len(data))]
all_str: list[str] = ["" for _ in range(len(data))]

for d in tqdm(loader, desc="quantize"):
    d = d.to(device)
    indices = model.get_indices(d, use_sk=False).view(-1, len(args.num_emb_list)).cpu()
    offset = 0
    for row in indices:
        all_codes[offset] = [int(x) for x in row]
        all_str[offset] = str(tuple(int(x) for x in row))
        offset += 1


def collision_groups(all_str):
    groups = {}
    for i, s in enumerate(all_str):
        groups.setdefault(s, []).append(i)
    return [v for v in groups.values() if len(v) > 1]


for vq in model.rq.vq_layers[:-1]:
    vq.sk_epsilon = 0.0

for tt in range(30):
    groups = collision_groups(all_str)
    if not groups:
        break
    print(f"collision round {tt}: {len(groups)} groups", flush=True)
    for group in groups:
        d = data[group].to(device)
        indices = model.get_indices(d, use_sk=True).view(-1, len(args.num_emb_list)).cpu()
        for item, row in zip(group, indices):
            all_codes[item] = [int(x) for x in row]
            all_str[item] = str(tuple(int(x) for x in row))

n = len(all_codes)
unique = len(set(all_str))
print(f"collision rate: {(n - unique) / n:.4f} ({n - unique}/{n})", flush=True)
for level in range(len(args.num_emb_list)):
    usage = len({c[level] for c in all_codes})
    print(f"level {level}: {usage} unique codes used / {args.num_emb_list[level]}", flush=True)

sid_vocab = {
    f"item_{i}": [f"<sid_{level}_{all_codes[i][level]}>" for level in range(len(args.num_emb_list))]
    for i in range(n)
}
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(sid_vocab, ensure_ascii=False))
print(f"saved -> {OUT_JSON}", flush=True)
