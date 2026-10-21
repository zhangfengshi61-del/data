"""Numerical smoke test: nine metrics use a common teacher candidate support."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recompute_paper_mechanisms.py"
spec = importlib.util.spec_from_file_location("paper_mechanisms", SCRIPT)
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)


class MechanismMetricsTests(unittest.TestCase):
    def setUp(self):
        # Tiny actual-array fixture; numbers are intentionally NOT paper data.
        scores = np.array([[0., -.5, -1.]])
        steps = np.repeat(scores[:, :, None] / 4., 4, axis=2)
        self.data = dict(sid=np.array([[[1, 1, 1, 0],
                                         [2, 1, 1, 0],
                                         [3, 1, 1, 0]]]),
                         teacher_scores=scores,
                         student_scores=scores.copy(),
                         teacher_steps=steps,
                         student_steps=steps.copy(),
                         base_steps=steps.copy(),
                         nvalid=np.array([3]),
                         step_scale=np.ones(4))

    def test_identity_has_zero_distribution_and_huber_error(self):
        m = metric.compute(self.data, topk=1, outsider=3)
        for name in ("item_kl", "hdrift_1", "hdrift_2",
                     "tgd_1", "tgd_2", "pov", "huber_d", "huber_i"):
            self.assertAlmostEqual(m[name], 0., places=9, msg=name)
        self.assertGreater(m["pair_softplus"], 0.)
        self.assertEqual(m["pairs"], 2)

    def test_student_preference_reversal_increases_mismatch(self):
        d = dict(self.data)
        d["student_scores"] = np.array([[-1., -.5, 0.]])
        d["student_steps"] = -self.data["teacher_steps"]
        r = metric.compute(d, topk=1, outsider=3)
        self.assertGreater(r["item_kl"], 0.)
        self.assertGreater(r["hdrift_1"], 0.)
        self.assertGreater(r["tgd_1"], 0.)
        self.assertGreater(r["pov"], 0.)

    def test_unsorted_teacher_support_rejected(self):
        d = dict(self.data)
        d["teacher_scores"] = self.data["teacher_scores"][:, ::-1]
        with self.assertRaisesRegex(ValueError, "sorted"):
            metric.compute(d, topk=1, outsider=3)
