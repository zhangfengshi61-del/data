"""Unit tests for reproducible validation-only checkpoint selection."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "sid_mlp" / "validation_selection.py"
spec = importlib.util.spec_from_file_location("val_policy", POLICY)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class TestBestValSelection(unittest.TestCase):
    def test_never_fall_back_to_test(self):
        with self.assertRaisesRegex(RuntimeError, "refusing to fall back"):
            mod.require_validation_split({"test": [1, 2]})
        with self.assertRaisesRegex(RuntimeError, "empty"):
            mod.require_validation_split({"val": [], "test": [1]})
        self.assertEqual(mod.require_validation_split({"val": [1], "test": [2]}), "val")

    def test_selection_full_validation_only(self):
        args = SimpleNamespace(epochs=3, beam_val_max_batches=2, test=False)
        with self.assertRaisesRegex(ValueError, "complete validation"):
            mod.validate_selection_settings(args)
        args.test = True
        mod.validate_selection_settings(args)
        args.beam_val_max_batches = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            mod.validate_selection_settings(args)
        args.beam_val_max_batches = None
        mod.validate_selection_settings(args)

    def test_every_fifth_and_always_final(self):
        self.assertEqual(
            [e for e in range(1, 13) if mod.should_beam_validate(e, 12)],
            [5, 10, 12],
        )
        self.assertEqual(
            [e for e in range(1, 4) if mod.should_beam_validate(e, 3)],
            [3],
        )

    def test_save_only_improving_nonempty_val_metric(self):
        saves = []
        def saver(blob, path):
            saves.append((dict(blob), path))
        kwargs = dict(path="student_best_val_ndcg10.pth",
                      payload={"model_state_dict": {"weight": [1]},
                               "args": {"arch": "gru", "seed": 42}},
                      save=saver)
        best, saved = mod.save_if_best_val_ndcg10(
            metrics={"ndcg10": 0.32, "total": 20}, epoch=5,
            previous_best=float("-inf"), **kwargs)
        self.assertTrue(saved)
        self.assertEqual(best, 0.32)
        self.assertEqual(saves[0][0]["selection_split"], "val")
        self.assertEqual(saves[0][0]["selection_metric"], "val_ndcg10")
        self.assertEqual(saves[0][0]["selection_num_requests"], 20)
        self.assertEqual(saves[0][0]["args"]["arch"], "gru")
        best, saved = mod.save_if_best_val_ndcg10(
            metrics={"ndcg10": 0.32, "total": 20}, epoch=10,
            previous_best=best, **kwargs)
        self.assertFalse(saved)
        self.assertEqual(len(saves), 1)
        best, saved = mod.save_if_best_val_ndcg10(
            metrics={"ndcg10": 0.34, "total": 20}, epoch=12,
            previous_best=best, **kwargs)
        self.assertTrue(saved)
        self.assertEqual(saves[-1][0]["epoch"], 12)
        for metrics in ({"ndcg10": math.nan, "total": 10},
                        {"ndcg10": math.inf, "total": 10},
                        {"ndcg10": 0.44, "total": 0}, {}):
            with self.assertRaises(RuntimeError):
                mod.save_if_best_val_ndcg10(
                    metrics=metrics, epoch=13, previous_best=best, **kwargs)

    def test_trainers_both_use_policy_and_no_test_fallback(self):
        for path in ("train_decoder.py", "kd_baselines.py"):
            text = (ROOT / "sid_mlp" / path).read_text()
            self.assertIn("require_validation_split(", text, path)
            self.assertIn("validate_selection_settings(args)", text, path)
            self.assertIn("save_if_best_val_ndcg10(", text, path)
            self.assertIn("should_beam_validate(", text, path)
            self.assertNotIn('split = "val" if "val" in ', text, path)

    def test_baselines_have_all_architectures(self):
        text = (ROOT / "sid_mlp" / "kd_baselines.py").read_text()
        for ctor in ("SIDMLP", "SIDGRU", "SIDCNN"):
            self.assertIn("model = " + ctor + "(", text)
        self.assertIn('choices=["mlp", "gru", "cnn"]', text)
        self.assertIn('"model_type": f"SID-{arch.upper()}"', text)
        self.assertIn("model._last_ctx = model.encode(enc_seq, enc_mask)", text)
        src = (ROOT / "sid_mlp" / "inference.py").read_text()
        self.assertIn('arch in ("gru", "cnn")', src)
        self.assertIn("model.load_state_dict(state, strict=True)", src)


if __name__ == "__main__":
    unittest.main()
