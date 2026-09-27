"""One real T5 optimization step, SID round-trip, and beam=30 generation."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "SDQ-403"
sys.path.insert(0, str(BASE))
from train_t5 import TIGERT5, cfg, freerec

dataset = getattr(freerec.data.datasets, cfg.dataset)(root=cfg.root)
model = TIGERT5(dataset).to(cfg.device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
batch = next(iter(model.sure_trainpipe(cfg.maxlen, cfg.batch_size)))
loss = model(batch)["rec_loss"]
assert torch.isfinite(loss)
loss.backward()
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step()
model.eval()
valid = next(iter(model.sure_validpipe(cfg.maxlen, batch_size=16)))
candidates, _ = model._generate_full_candidates(valid)
assert candidates.shape == (16, 30)
assert candidates.min() >= 0 and candidates.max() < model.Item.count
converter = model.converter
for item in range(model.Item.count):
    assert converter.decode(converter.encode(converter.format(item))) == [item]
result = {"passed": True, "train_loss": loss.item(), "batch_size": cfg.batch_size,
          "candidate_shape": list(candidates.shape), "roundtrip_items": model.Item.count,
          "parameters": sum(p.numel() for p in model.parameters()),
          "num_sid_tokens_with_collision_suffix": converter.max_num_sid_tokens,
          "semantic_vocab": str(cfg.sid_vocab_file)}
(ROOT / "SDQ-LARS/smoke_t5_result.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result))
