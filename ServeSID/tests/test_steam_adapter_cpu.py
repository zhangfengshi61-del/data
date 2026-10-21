"""Dependency-light smoke test of the actual Steam100K dataset class AST."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest


class StubDataset:
    def __init__(self, config):
        self.config = config
        self.id_mapping = dict(item2id={"[PAD]": 0}, user2id={"[PAD]": 0},
                               id2item=["[PAD]"], id2user=["[PAD]"])
        self.all_item_seqs = {}
        self.item2meta = None

    @property
    def n_users(self):
        return len(self.id_mapping["user2id"])

    @property
    def n_items(self):
        return len(self.id_mapping["item2id"])

    def log(self, *_args):
        pass


path = Path(__file__).resolve().parents[1] / "genrec/datasets/Steam100K/dataset.py"
tree = ast.parse(path.read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Steam100K")
scope = dict(AbstractDataset=StubDataset, os=os, json=json,
             clean_text=lambda x: str(x))
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
Steam100K = scope["Steam100K"]


class SteamAdapterTests(unittest.TestCase):
    def test_accepts_preprocessed_sequences_and_stable_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            (data / "user_item_sequences.jsonl").write_text(
                json.dumps({"user": 10, "items": [200, 100, 300, 200]}) + "\n",
                encoding="utf-8")
            (data / "item_text.jsonl").write_text(
                "".join(json.dumps({"item": i, "title": str(i)}) + "\n"
                        for i in (100, 200, 300)), encoding="utf-8")
            dataset = Steam100K(dict(steam100k_root=tmp, metadata="sentence"))
            self.assertEqual(dataset.all_item_seqs["10"], ["200", "100", "300", "200"])
            self.assertEqual(dataset.id_mapping["id2item"], ["[PAD]", "100", "200", "300"])
            self.assertIn("100", dataset.item2meta)

    def test_rejects_missing_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "user_item_sequences.jsonl").write_text(
                '{"user":"u","items":["a","b","c"]}\n', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                Steam100K(dict(steam100k_root=tmp, metadata="sentence"))

    def test_preserves_explicit_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "user_item_sequences.jsonl").write_text(
                '{"user":"u","items":["a","b","c"]}\n', encoding="utf-8")
            (root / "id_mapping.json").write_text(json.dumps({
                "id2item": ["[PAD]", "c", "a", "b"],
                "id2user": ["[PAD]", "u"]
            }), encoding="utf-8")
            ds = Steam100K(dict(steam100k_root=tmp, metadata="none"))
            self.assertEqual(ds.id_mapping["item2id"]["c"], 1)
