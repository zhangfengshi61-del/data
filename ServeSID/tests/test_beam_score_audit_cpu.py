"""Numerical regression for the teacher-cache score-convention audit."""
import importlib.util
from pathlib import Path
import unittest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_beam_score_convention.py"
spec = importlib.util.spec_from_file_location("beam_score_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BeamScoreAuditTests(unittest.TestCase):
    def setUp(self):
        self.steps = torch.full((2, 3, 4), -0.25)
        self.base = self.steps.sum(-1)
        self.dump = dict(codes=torch.zeros(2, 3, 4, dtype=torch.long),
                         step_logp=self.steps, nvalid=torch.tensor([3, 2]),
                         scores=self.base + torch.tensor([[.2, .2, .2], [.4, .4, 0.]]))

    def test_constant_remainder_is_safe(self):
        result = module.audit(self.dump)
        self.assertEqual(result["nonconstant_remainder_requests"], 0)
        self.assertLess(result["max_within_request_remainder_spread_nats"], 1e-5)

    def test_candidate_dependent_remainder_is_detected(self):
        self.dump["scores"][0, 1] += .15
        result = module.audit(self.dump)
        self.assertEqual(result["nonconstant_remainder_requests"], 1)
        self.assertGreater(result["max_within_request_remainder_spread_nats"], .1)

    def test_inconsistent_dimensions_are_rejected(self):
        self.dump["step_logp"] = self.steps[:, :2]
        with self.assertRaisesRegex(ValueError, "dimensions"):
            module.audit(self.dump)
