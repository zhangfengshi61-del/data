"""Small real PyTorch smoke test: baseline training backbone -> saved checkpoint -> inference.

Run on CPU with torch, numpy, tqdm installed:
    PYTHONPATH=ServeSID python -m unittest discover -s ServeSID/tests -p 'test_backbone_cpu_smoke.py' -v
"""
import os
import tempfile
from types import SimpleNamespace
import unittest

import torch

from sid_mlp.kd_baselines import build_baseline_student
from sid_mlp.inference import build_sidmlp_from_checkpoint


class TestThreeBackbones(unittest.TestCase):
    def test_forward_backward_and_strict_checkpoint_reload(self):
        torch.manual_seed(123)
        embedding = torch.randn(1024, 128)
        enc = torch.randn(2, 8, 128)
        mask = torch.tensor([[False] * 8, [False] * 6 + [True] * 2])
        d1 = torch.tensor([3, 7], dtype=torch.long)
        d2 = torch.tensor([256 + 3, 256 + 7], dtype=torch.long)
        d3 = torch.tensor([512 + 3, 512 + 7], dtype=torch.long)

        for arch in ("mlp", "gru", "cnn"):
            with self.subTest(arch=arch):
                args = SimpleNamespace(
                    arch=arch, num_heads=4, ffn_dim=64, attn_dim=32,
                    head_hidden=32, head_layers=1, dropout=0.0,
                    gru_hidden=16, gru_layers=1, conv_dim=16,
                    conv_kernel=3, conv_layers=2, seed=42,
                )
                model = build_baseline_student(args, embedding, torch.device("cpu"))
                model.train()
                logits = model(enc, mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
                self.assertEqual(len(logits), 4)
                self.assertTrue(all(v.shape == (2, 256) for v in logits))
                sum(v.square().mean() for v in logits).backward()
                self.assertTrue(any(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters() if p.requires_grad
                ))

                # Inference must recover architecture solely from checkpoint.
                with tempfile.TemporaryDirectory() as tmp:
                    name = os.path.join(tmp, f"baseline_{arch}_best_val_ndcg10.pth")
                    torch.save({
                        "epoch": 5,
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "model_type": f"SID-{arch.upper()}",
                        "selection_split": "val",
                        "selection_metric": "val_ndcg10",
                        "selection_value": 0.04,
                    }, name)
                    restored, ckpt = build_sidmlp_from_checkpoint(
                        name, embedding, torch.device("cpu"))
                    self.assertEqual(ckpt["args"]["arch"], arch)
                    self.assertEqual(type(restored), type(model))
                    restored.eval()
                    model.eval()
                    with torch.no_grad():
                        before = model(enc, mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
                        after = restored(enc, mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
                    for x, y in zip(before, after):
                        self.assertTrue(torch.allclose(x, y, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
