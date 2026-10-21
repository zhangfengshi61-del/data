"""Synthetic unit tests; no values here are experimental paper results."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "scripts" / "recompute_gold_prefix_interpolation.py"
spec = importlib.util.spec_from_file_location("gold_interpolation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GoldPrefixInterpolationTests(unittest.TestCase):
    def test_alpha_zero_and_one(self):
        sid = np.array([[[1, 2, 3, 4], [5, 2, 3, 4], [6, 2, 3, 4]]])
        gt = np.array([[1, 2, 3, 4]])
        teacher = np.array([[0., -.2, -.4]])
        steps = np.repeat(teacher[:, :, None] / 4., 4, axis=-1)
        ps = np.full((1, 4, 256), .2 / 255)
        pt = np.full((1, 4, 256), .2 / 255)
        ps[:, :, 1] = .8
        pt[:, :, 1] = .8
        ps[:, 0, :] = 0.3 / 254.
        pt[:, 0, :] = 0.3 / 254.
        ps[:, 0, 1] = .6
        ps[:, 0, 5] = .1
        pt[:, 0, 1] = .2
        pt[:, 0, 5] = .5
        # Both distributions have rows summing to 1 after adjusting elsewhere.
        pt[:, 0, :] = 0.
        ps[:, 0, :] = 0.
        pt[:, 0, 1], pt[:, 0, 5], pt[:, 0, 6] = .2, .5, .3
        ps[:, 0, 1], ps[:, 0, 5], ps[:, 0, 6] = .6, .1, .3
        data = dict(sid=sid, gold_sid=gt, nvalid=np.array([3]),
                    teacher_scores=teacher, student_steps=steps,
                    teacher_gold_probs=pt, student_gold_probs=ps)
        results = module.evaluate(data, alphas=(0., .5, 1.), topk=1, outsider=3)
        self.assertGreater(results[0]["token_kl"], results[-1]["token_kl"])
        self.assertAlmostEqual(results[-1]["token_kl"], 0., places=9)
        self.assertGreaterEqual(results[0]["candidate_kl"], 0.)
        self.assertEqual(results[0]["requests"], 1)

    def test_rejects_misaligned_probabilities(self):
        data = dict(sid=np.zeros((1, 2, 4), dtype=int),
                    gold_sid=np.zeros((1, 4), dtype=int),
                    nvalid=np.array([2]), teacher_scores=np.zeros((1, 2)),
                    student_steps=np.zeros((1, 2, 4)),
                    student_gold_probs=np.zeros((1, 4, 256)),
                    teacher_gold_probs=np.zeros((1, 4, 256)))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            module.evaluate(data, topk=1, outsider=2)
