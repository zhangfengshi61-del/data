"""CPU regression tests for the four paper-specific HCDA/RRDA controls."""
import unittest

import torch
import torch.nn.functional as F

from sid_mlp.losses import da_cdrs_loss, shuffle_hcda_candidate_scores, serve_kl_loss


class PaperAblationTests(unittest.TestCase):
    def setUp(self):
        self.codes = torch.tensor([[
            [1, 1, 1, 0], [1, 2, 1, 0], [2, 1, 2, 0], [3, 1, 1, 0],
        ]], dtype=torch.long)
        self.teacher = torch.tensor([[
            [-0.3, -0.2, -0.4, -0.2],
            [-0.6, -0.1, -0.2, -0.3],
            [-0.7, -0.4, -0.1, -0.2],
            [-1.2, -0.3, -0.2, -0.4],
        ]])
        self.student = torch.tensor([[
            [-0.9, -0.5, -0.2, -0.5],
            [-0.3, -0.6, -0.4, -0.1],
            [-0.8, -0.1, -0.3, -0.3],
            [-0.7, -0.5, -0.2, -0.2],
        ]], requires_grad=True)
        self.valid = torch.ones(1, 4, dtype=torch.bool)

    def _loss(self, mode):
        return da_cdrs_loss(self.teacher, self.student, self.codes,
                            self.valid, torch.ones(4), n_pos=1,
                            n_neg_hi=4, weight_mode=mode,
                            counterfactual=(mode == "cf"), margin=0.05)

    def test_all_paper_weighting_modes_have_finite_gradients(self):
        results = []
        for mode in ("cf", "uniform_valid", "posres", "absres",
                     "posuni", "collapsed"):
            self.student.grad = None
            loss = self._loss(mode)
            self.assertTrue(torch.isfinite(loss), mode)
            loss.backward()
            self.assertTrue(torch.isfinite(self.student.grad).all(), mode)
            results.append(round(loss.item(), 6))
        self.assertGreater(len(set(results)), 2)

    def test_absres_and_posuni_match_manual_weight_definitions(self):
        ds = self.student[:, :1, :] - self.student[:, 1:, :]
        dt = self.teacher[:, :1, :] - self.teacher[:, 1:, :]
        r = dt[..., :3] - ds[..., :3]
        # All pairs in this fixture diverge from the first SID depth.
        eligible = torch.ones_like(r)
        huber = F.smooth_l1_loss((ds[..., :3] - dt[..., :3]),
                                  torch.zeros_like(r), reduction="none")
        gated = (r > 0).any(-1, keepdim=True)
        abs_w = r.abs() * eligible * gated
        abs_w = abs_w / abs_w.sum(-1, keepdim=True).clamp_min(1e-9)
        pos_w = (r > 0).float()
        pos_w = pos_w / pos_w.sum(-1, keepdim=True).clamp_min(1.0)
        self.assertTrue(torch.allclose(self._loss("absres"),
                          (abs_w * huber).sum(-1).mean(), atol=1e-6))
        self.assertTrue(torch.allclose(self._loss("posuni"),
                          (pos_w * huber).sum(-1).mean(), atol=1e-6))

    def test_collapsed_uses_detached_batch_std_of_summed_residual(self):
        ds = self.student[:, :1, :] - self.student[:, 1:, :]
        dt = self.teacher[:, :1, :] - self.teacher[:, 1:, :]
        residual = (ds[..., :3] - dt[..., :3]).sum(-1)
        scale = residual.detach().std(unbiased=False).clamp_min(1e-4)
        manual = F.smooth_l1_loss(residual / scale,
                                  torch.zeros_like(residual)).mean()
        self.assertTrue(torch.allclose(self._loss("collapsed"), manual, atol=1e-6))

    def test_shuffled_preserves_multiset_and_padding_and_grad(self):
        s = torch.tensor([[-1., -2., -3., -4., -99.]], requires_grad=True)
        t = torch.tensor([[2., 3., 5., 7., -99.]])
        valid = torch.tensor([[True, True, True, True, False]])
        generator = torch.Generator().manual_seed(42)
        shuffled_s, shuffled_t = shuffle_hcda_candidate_scores(
            s, t, valid, generator=generator)
        self.assertEqual(shuffled_s[0, -1].item(), -99.)
        self.assertEqual(sorted(shuffled_s[0, :4].tolist()), [-4., -3., -2., -1.])
        self.assertEqual(sorted(shuffled_t[0, :4].tolist()), [2., 3., 5., 7.])
        # Each pair of scores must be moved by the same permutation.
        for j in range(4):
            orig = [-1., -2., -3., -4.].index(shuffled_s[0, j].item())
            self.assertEqual(shuffled_t[0, j].item(), [2., 3., 5., 7.][orig])
        sid = torch.tensor([[[1, 1, 1, 0], [1, 2, 1, 0],
                             [2, 1, 1, 0], [3, 1, 1, 0], [0, 0, 0, 0]]])
        loss = serve_kl_loss(shuffled_s, shuffled_t, sid, valid)
        loss.backward()
        self.assertIsNotNone(s.grad)
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_zero_valid_pairs_do_not_produce_nan(self):
        valid = torch.zeros_like(self.valid)
        for mode in ("cf", "uniform_valid", "absres", "posuni", "collapsed"):
            loss = da_cdrs_loss(
                self.teacher, self.student, self.codes, valid,
                torch.ones(4), n_pos=1, n_neg_hi=4, weight_mode=mode,
                counterfactual=(mode == "cf"), margin=0.05)
            self.assertTrue(torch.isfinite(loss), mode)
            self.assertEqual(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
