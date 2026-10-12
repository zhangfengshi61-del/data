"""Paper-loss regression checks. Run: python -m unittest discover -s ServeSID/tests -v"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]


def settings(name):
    result = {}
    for raw in (ROOT / "configs" / name).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


class TestPaperLoss(unittest.TestCase):
    def test_full_objective_coefficients(self):
        cfg = settings("servesid_final.yaml")
        required = {
            "loss": "ce", "d4_teacher": "ce", "pair_w": "0.0",
            "serve_kl": "4", "da_w": "1.0", "da_corr_w": "31.42",
            "da_margin": "0.05", "corr_mode": "cf", "list_topm": "64",
        }
        for name, expected in required.items():
            self.assertEqual(cfg.get(name), expected, name)
        self.assertEqual(float(cfg.get("list_kl", "0")), 0)
        self.assertEqual(float(cfg.get("prefix_kl", "0")), 0)

    def test_plain_pair_only_as_separate_ablation(self):
        pair = settings("abl_hserve_pair.yaml")
        self.assertEqual(float(pair["pair_w"]), 1.0)
        self.assertEqual(float(pair["da_w"]), 0.0)
        self.assertEqual(pair["pair_margin"], "0.05")
        for name in ("abl_daonly.yaml", "abl_hserve_attruni.yaml"):
            cfg = settings(name)
            self.assertEqual(float(cfg["pair_w"]), 0.0, name)
            self.assertEqual(float(cfg["da_w"]), 1.0, name)
            self.assertEqual(cfg["loss"], "ce")
            self.assertEqual(cfg["d4_teacher"], "ce")

    def test_fail_fast_guard_and_architecture_retention(self):
        source = (ROOT / "sid_mlp" / "train_decoder.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        fn = next(
            item for item in tree.body
            if isinstance(item, ast.FunctionDef) and
            item.name == "validate_full_paper_objective"
        )
        env = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<full-loss-check>", "exec"), env)
        validate = env["validate_full_paper_objective"]
        full = dict(serve_kl=4, da_w=1, pair_w=0, loss="ce",
                    d4_teacher="ce", da_margin=.05,
                    list_kl=0, prefix_kl=0, cdrs_w=0, seqkd=False)
        validate(SimpleNamespace(**full))
        with self.assertRaisesRegex(ValueError, "Pair Loss"):
            validate(SimpleNamespace(**(full | {"pair_w": 1})))
        with self.assertRaisesRegex(ValueError, "requires loss=ce"):
            validate(SimpleNamespace(**(full | {"loss": "combined"})))
        with self.assertRaisesRegex(ValueError, "only digit CE"):
            validate(SimpleNamespace(**(full | {"seqkd": True})))
        with self.assertRaisesRegex(ValueError, "pair_margin"):
            validate(SimpleNamespace(serve_kl=4, da_w=0, pair_w=1))
        validate(SimpleNamespace(serve_kl=4, da_w=0, pair_w=1, pair_margin=0.05))
        self.assertIn("loss = loss + lam_da * corr_w * l_da", source)
        for arch in ("SIDGRU", "SIDCNN"):
            self.assertIn("model = " + arch + "(", source)
            self.assertIn("class " + arch, (ROOT / "sid_mlp" / "models.py").read_text())

    def test_full_missing_dump_is_rejected_before_training(self):
        source = (ROOT / "sid_mlp" / "train_decoder.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        pure = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("cache_supervision_required", "validate_cache_supervision")]
        self.assertEqual(len(pure), 2)
        env = {}
        exec(compile(ast.Module(body=pure, type_ignores=[]), "<cache-check>", "exec"), env)
        full = SimpleNamespace(serve_kl=4, da_w=1, pair_w=0, beam_dump=None)
        with self.assertRaisesRegex(ValueError, "Refusing to train"):
            env["validate_cache_supervision"](full)
        full.beam_dump = "/cache/train_beam.pt"
        env["validate_cache_supervision"](full)
        # Digit-only validation is intentionally allowed without extras.
        env["validate_cache_supervision"](SimpleNamespace(
            serve_kl=0, da_w=0, pair_w=0, beam_dump=None))
        self.assertIn("validate_cache_supervision(args)", source)
        self.assertIn("active cached-candidate", source)

    def test_da_negative_sampling_does_not_enable_plain_pair(self):
        source = (ROOT / "sid_mlp" / "train_decoder.py").read_text(encoding="utf-8")
        # A positive da_negs count may select DA candidates independently.
        self.assertIn("n_pos=da_k, n_neg_hi=da_b", source)
        self.assertNotIn("--da_negs requires pair_w>0", source)
        self.assertIn("skipped_batches=0", source)
        self.assertIn("error_if_nonfinite=True", source)

    def test_additive_margin_and_detached_attribution_implementation(self):
        source = (ROOT / "sid_mlp" / "losses.py").read_text(encoding="utf-8")
        self.assertIn("F.softplus(margin - G_S)", source)
        self.assertIn("F.softplus(margin - G_tilde)", source)
        self.assertIn("w = (A / A.sum(-1, keepdim=True).clamp_min(1e-9)).detach()", source)
        self.assertIn("pos = F.relu(dG_T - dG_S)[..., :Dc] * eligible", source)

    def test_teacher_depth_scores_follow_candidate_sorting(self):
        source = (ROOT / "sid_mlp" / "dataset.py").read_text(encoding="utf-8")
        self.assertIn("d['step_logp'] = d['step_logp'].gather(", source)


if __name__ == "__main__":
    unittest.main()
