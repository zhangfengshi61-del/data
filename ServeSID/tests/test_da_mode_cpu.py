"""CPU regression checks for all three DA-CDRS weighting paths."""
import unittest

import torch

from sid_mlp.losses import da_cdrs_loss


class TestDACDRSModes(unittest.TestCase):
    def setUp(self):
        self.codes = torch.tensor([[
            [1, 1, 1, 1], [1, 1, 2, 1],
            [1, 2, 1, 1], [2, 1, 1, 1],
        ]], dtype=torch.long)
        self.teacher = torch.tensor([[
            [-.2, -.3, -.1, -.3], [-.1, -.5, -.3, -.2],
            [-.8, -.6, -.2, -.4], [-.7, -.8, -.1, -.4],
        ]])
        self.student = torch.tensor([[
            [-.5, -.4, -.5, -.2], [-.2, -.7, -.1, -.3],
            [-.2, -.4, -.5, -.6], [-.4, -.6, -.4, -.2],
        ]], requires_grad=True)
        self.valid = torch.tensor([[True, True, True, True]])
        self.scale = torch.ones(4)

    def loss(self, mode):
        return da_cdrs_loss(
            self.teacher, self.student, self.codes,
            self.valid, self.scale, n_pos=2, n_neg_hi=4,
            weight_mode=mode, counterfactual=(mode == "cf"),
            margin=.05,
        )

    def test_three_modes_are_finite_and_trainable(self):
        measured = []
        for mode in ("cf", "uniform_valid", "posres"):
            self.student.grad = None
            v = self.loss(mode)
            self.assertTrue(torch.isfinite(v))
            v.backward()
            self.assertIsNotNone(self.student.grad)
            self.assertTrue(torch.isfinite(self.student.grad).all())
            measured.append(float(v))
        self.assertGreater(len(set(round(v, 7) for v in measured)), 1)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown DA-CDRS"):
            self.loss("not-a-valid-weight")
        with self.assertRaisesRegex(ValueError, "counterfactual=True"):
            da_cdrs_loss(
                self.teacher, self.student, self.codes,
                self.valid, self.scale, n_pos=2, n_neg_hi=4,
                weight_mode="cf", counterfactual=False, margin=.05,
            )


if __name__ == "__main__":
    unittest.main()
