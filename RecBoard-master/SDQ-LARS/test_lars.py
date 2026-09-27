"""Numerical reference and non-leakage checks; python -m unittest test_lars -v."""
import unittest
import numpy as np
import torch
from sklearn.linear_model import lars_path
from lars_solver import lar_code


class LARSTest(unittest.TestCase):
    def test_matches_sklearn_lar(self):
        rng = np.random.default_rng(2025)
        for dimension, n_atoms in [(12, 8), (16, 32), (128, 256)]:
            d = rng.normal(size=(n_atoms, dimension))
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            y = rng.normal(size=(5, dimension))
            for steps in (1, 2, 3):
                actual = lar_code(torch.tensor(y), torch.tensor(d), steps, ridge=0).numpy()
                for row in range(len(y)):
                    _, _, path = lars_path(d.T, y[row], method="lar", max_iter=steps)
                    np.testing.assert_allclose(actual[row], path[:, -1], atol=1e-8, rtol=1e-6)

    def test_orthogonal_example_and_negative_signs(self):
        d = torch.eye(2, dtype=torch.float64)
        y = torch.tensor([[2., 1.], [-2., 1.]], dtype=torch.float64)
        first = lar_code(y, d, 1, ridge=0)
        torch.testing.assert_close(first, torch.tensor([[1., 0.], [-1., 0.]], dtype=y.dtype))
        torch.testing.assert_close(lar_code(y, d, 2, ridge=0), y)

    def test_zero_and_collinear(self):
        d = torch.tensor([[1., 0., 0.], [1., 0., 0.], [-1., 0., 0.], [0., 1., 0.]])
        y = torch.tensor([[0., 0., 0.], [2., 1., 0.]])
        a = lar_code(y, d, 3)
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a[0], torch.zeros_like(a[0]))
        self.assertLessEqual(((a @ d - y) ** 2).sum().item(), (y ** 2).sum().item())

    def test_outer_dictionary_gradient_only(self):
        torch.manual_seed(9)
        c = torch.randn(16, 8, requires_grad=True)
        y = torch.randn(4, 8, requires_grad=True)
        d = torch.nn.functional.normalize(c, dim=-1)
        a = lar_code(y, d, 3)
        self.assertFalse(a.requires_grad)
        ((a @ d - y.detach()) ** 2).sum().backward()
        self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater((c.grad.abs().sum(1) > 0).sum().item(), 1)

    def test_sdq_hard_path_and_zero_weight_gradient_equivalence(self):
        from types import SimpleNamespace
        from lars_quantizer import LARSStructureDiffusionQuantizer
        from quantizer import StructureDiffusionQuantizer
        from freerec.data.tags import ITEM, ID
        item = SimpleNamespace(count=8, fork=lambda tag: "seq")
        dataset = SimpleNamespace(fields={(ITEM, ID): item})
        dataset.train = lambda: SimpleNamespace(to_seqs=lambda maxlen: [{"seq": tuple(range(8))}])
        torch.manual_seed(42)
        kwargs = dict(dataset=dataset, hidden_size=4, num_codebooks=3,
                      num_codewords=6, sk_epsilons=[0., 0., 0.003], sk_iters=50)
        base = StructureDiffusionQuantizer(**kwargs)
        model = LARSStructureDiffusionQuantizer(**kwargs, sparse_weight=0., bridge_weight=0.)
        model.load_state_dict(base.state_dict(), strict=True)
        base.reset_local_graph(torch.arange(8))
        model.reset_local_graph(torch.arange(8))
        z = torch.randn(8, 4, requires_grad=True)
        q0, loss0, ids0 = base(z)
        q1, loss1, ids1 = model(z)
        torch.testing.assert_close(q0, q1, rtol=0, atol=0)
        torch.testing.assert_close(loss0, loss1, rtol=0, atol=0)
        torch.testing.assert_close(ids0, ids1)
        loss0.backward(retain_graph=True)
        loss1.backward()
        for c0, c1 in zip(base.codebooks, model.codebooks):
            torch.testing.assert_close(c0.weight.grad, c1.weight.grad, rtol=0, atol=0)
        model.sparse_weight = model.bridge_weight = 0.1
        model.loss_scale = 1.
        q2, loss2, ids2 = model(z)
        torch.testing.assert_close(q0, q2, rtol=0, atol=0)
        torch.testing.assert_close(ids0, ids2)
        self.assertGreater(loss2.item(), loss0.item())
        model.zero_grad()
        loss2.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        _, loss3, ids3 = model(z)
        torch.testing.assert_close(loss0, loss3, rtol=0, atol=0)
        torch.testing.assert_close(ids0, ids3)
        self.assertEqual(model.last_diagnostics, [])

    def test_margin_gate_rejects_small_improvements(self):
        from types import SimpleNamespace
        from lars_quantizer import LARSStructureDiffusionQuantizer
        from freerec.data.tags import ITEM, ID
        item = SimpleNamespace(count=8, fork=lambda tag: "seq")
        dataset = SimpleNamespace(fields={(ITEM, ID): item})
        dataset.train = lambda: SimpleNamespace(to_seqs=lambda maxlen: [{"seq": tuple(range(8))}])
        model = LARSStructureDiffusionQuantizer(
            dataset=dataset, hidden_size=4, num_codebooks=3, num_codewords=6,
            sk_epsilons=[0., 0., 0.003], accept_margin=.9, gate_sparse=True)
        model.reset_local_graph(torch.arange(8))
        model.loss_scale = 1.
        model.train()
        _, loss, _ = model(torch.randn(8, 4))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(0 <= x[-1].item() <= 1 for x in model.last_diagnostics))

    def test_stage_budget_and_anchor_are_finite(self):
        from types import SimpleNamespace
        from lars_quantizer import LARSStructureDiffusionQuantizer
        from freerec.data.tags import ITEM, ID
        item = SimpleNamespace(count=8, fork=lambda tag: "seq")
        dataset = SimpleNamespace(fields={(ITEM, ID): item})
        dataset.train = lambda: SimpleNamespace(to_seqs=lambda maxlen: [{"seq": tuple(range(8))}])
        model = LARSStructureDiffusionQuantizer(
            dataset=dataset, hidden_size=4, num_codebooks=3, num_codewords=6,
            lars_steps=[3, 2, 1], stage_weights=[1., .5, .25],
            sparse_weight=.1, bridge_weight=.1, anchor_weight=.01)
        model.reset_local_graph(torch.arange(8))
        model.set_codebook_anchor()
        model.loss_scale = 1.
        _, loss, ids = model(torch.randn(8, 4))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(ids.shape), (8, 3))

    def test_soft_sequence_assignments_are_differentiable(self):
        from types import SimpleNamespace
        from lars_quantizer import LARSStructureDiffusionQuantizer
        from freerec.data.tags import ITEM, ID
        item = SimpleNamespace(count=8, fork=lambda tag: "seq")
        dataset = SimpleNamespace(fields={(ITEM, ID): item})
        dataset.train = lambda: SimpleNamespace(to_seqs=lambda maxlen: [{"seq": tuple(range(8))}])
        model = LARSStructureDiffusionQuantizer(
            dataset=dataset, hidden_size=4, num_codebooks=3, num_codewords=6,
            sequence_weight=.05, sequence_temperature=.05)
        model.reset_local_graph(torch.arange(8))
        model.train()
        _, loss, _ = model(torch.randn(8, 4, requires_grad=True))
        self.assertEqual(len(model.last_soft_assignments), 3)
        self.assertTrue(all(p.requires_grad for p in model.last_soft_assignments))
        loss.backward()
        self.assertTrue(torch.isfinite(model.codebooks[0].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
