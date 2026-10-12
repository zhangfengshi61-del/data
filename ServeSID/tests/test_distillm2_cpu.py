"""CPU regression tests for DistiLLM-2 autograd and codebook probabilities."""
import unittest

import torch
from sid_mlp.constants import DIGIT_OFFSETS, CODEBOOK_SIZE
from sid_mlp.kd_baselines import (
    teacher_codebook_logp,
    contrastive_student_sid_loss,
    _jsd,
    _skew_kl,
)


class DistiLLM2Tests(unittest.TestCase):
    def test_codebook_teacher_normalization_not_full_vocab_slices(self):
        offset_end = max(DIGIT_OFFSETS[:4]) + CODEBOOK_SIZE
        logits = torch.full((2, 4, offset_end + 64), -2.0)
        for d, offset in enumerate(DIGIT_OFFSETS[:4]):
            logits[:, d, offset:offset+CODEBOOK_SIZE] = torch.arange(
                CODEBOOK_SIZE, dtype=torch.float32) / 20.0
            # Outside-codebook logit may be extremely large. It must not
            # change conditional teacher probabilities on this codebook.
            logits[:, d, -1] = 50.0
        parts = teacher_codebook_logp(logits)
        self.assertEqual(len(parts), 4)
        for logp in parts:
            torch.testing.assert_close(logp.exp().sum(-1), torch.ones(2), atol=1e-6, rtol=0)
            self.assertLess(float(_jsd(logp, logp).abs().max()), 1e-6)
            self.assertLess(abs(float(_skew_kl(logp, logp).abs().max())), 1e-6)
        expected = torch.log_softmax(torch.arange(CODEBOOK_SIZE).float()/20, -1)
        torch.testing.assert_close(parts[0][0], expected)

    def test_student_contrastive_term_changes_gradients(self):
        teacher = torch.tensor([[0.1, -0.4, -2.0], [1.0, -0.6, -1.8]])
        layer = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(3))
        x = torch.tensor([[0., .5, -1.], [.4, .2, -.8]])
        scored = layer(x)
        contrastive = contrastive_student_sid_loss(scored, teacher)
        self.assertTrue(contrastive.requires_grad)
        self.assertGreater(float(contrastive), 0)
        base = scored.square().mean()
        grad_base, = torch.autograd.grad(base, layer.weight, retain_graph=True)
        grad_full, = torch.autograd.grad(base + .1*contrastive, layer.weight)
        self.assertGreater(float((grad_full-grad_base).abs().max()), 1e-5)

    def test_teacher_labels_not_in_gradient(self):
        teacher = torch.tensor([[0.1, -0.4, -2.0]], requires_grad=True)
        scores = torch.tensor([[0., .5, -1.]], requires_grad=True)
        v = contrastive_student_sid_loss(scores, teacher)
        self.assertIsNone(torch.autograd.grad(v, teacher, allow_unused=True, retain_graph=True)[0])
        self.assertIsNotNone(torch.autograd.grad(v, scores)[0])

    def test_first_on_policy_depth_contributes_student_gradient(self):
        # DistiLLM-2 on-policy uses skew-KL at all four positions.
        first_logits = torch.randn(2, CODEBOOK_SIZE, requires_grad=True)
        teacher = torch.log_softmax(torch.randn(2, CODEBOOK_SIZE), dim=-1)
        loss = _skew_kl(teacher, torch.log_softmax(first_logits, -1), .9).mean()
        grad, = torch.autograd.grad(loss, first_logits)
        self.assertGreater(float(grad.abs().max()), 1e-7)
        from pathlib import Path
        code = (Path(__file__).resolve().parents[1] /
                "sid_mlp" / "kd_baselines.py").read_text()
        self.assertIn("_skew_kl(pT1o, pS1, alpha)", code)
        self.assertNotIn("_jsd(pT1o, pS1.detach())", code)

    def test_invalid_temperature_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            contrastive_student_sid_loss(
                torch.randn(2, 3), torch.randn(2, 3), temperature=0)


if __name__ == "__main__":
    unittest.main()
