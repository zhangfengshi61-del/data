"""No-GPU regression checks for cost protocol and Full M/B separation."""
import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_infer_cost.py"


class TestCostAudit(unittest.TestCase):
    def test_benchmark_serialization_helpers(self):
        spec = importlib.util.spec_from_file_location("bench_cost", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        metrics = module.extract_metrics('before\\nMETRICS_JSON {"total": 25, "ndcg10": 0.1}\\n')
        self.assertEqual(metrics["total"], 25)
        self.assertEqual(module.summarize([2, 4])["mean"], 3)
        self.assertIsNotNone(module.summarize([2, 4])["sd"])
        self.assertIsNone(module.summarize([2])["sd"])

    def test_full_m64_b40_invariant(self):
        config = (ROOT / "configs" / "servesid_final.yaml").read_text()
        self.assertIn("list_topm: 64", config)
        self.assertIn("cdrs_neg_end: 40", config)
        self.assertIn("pair_w: 0.0", config)
        src = (ROOT / "sid_mlp" / "train_decoder.py").read_text()
        self.assertIn("actual_m != requested_m", src)
        self.assertIn("outsider_b > requested_m", src)
        self.assertIn("Full ServeSID requires the configured candidate width M", src)

    def test_inference_cost_is_opt_in_and_scope_explicit(self):
        src = (ROOT / "sid_mlp" / "inference.py").read_text()
        ast.parse(src)
        self.assertIn("cost_profile=None", src)
        self.assertIn("persistent_allocated_bytes", src)
        self.assertIn("peak_allocated_bytes", src)
        self.assertIn("student_context", src)
        self.assertIn("frozen_encoder", src)
        self.assertIn("not decoder-only incremental memory", src)
        cli = (ROOT / "sid_mlp" / "cli" / "infer.py").read_text()
        self.assertIn("--cost_json", cli)
        self.assertIn("--cost_warmup_batches", cli)

    def test_legacy_scans_are_explicitly_separate(self):
        p = (ROOT / "verification" / "J4_COST_AND_BM_PROVENANCE.md").read_text()
        self.assertIn("HServe-only", p)
        self.assertIn("HServe + Plain Pair", p)
        self.assertIn("not modern no-direct-pair Full", p)
        self.assertIn("0.6451", p)
        self.assertIn("0.6739", p)


if __name__ == "__main__":
    unittest.main()
