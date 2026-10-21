"""Explicit canonical JSONL adapter for preprocessed Steam-100K sequences.

This is NOT a claim that an unspecified historical Steam benchmark used this
preprocessing. To reproduce a reported checkpoint, the original mapping,
sequence order, item metadata, SID file, teacher cache and splits must match.
"""
import json
import os

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Steam100K(AbstractDataset):
    def __init__(self, config):
        super().__init__(config)
        self.category = config.get("category") or "Steam100K"
        self.data_root = config.get("steam100k_root") or os.environ.get("STEAM100K_ROOT")
        if not self.data_root:
            raise ValueError("Steam100K needs steam100k_root or STEAM100K_ROOT; raw-data processing is not inferred")
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError("Steam100K processed directory missing: " + self.data_root)
        self._load_item_seqs()
        self._build_mapping()
        self._load_item_meta()
        self.log("[DATASET] Steam100K processed: %d users and %d items" %
                 (self.n_users - 1, self.n_items - 1))

    def _load_item_seqs(self):
        path = os.path.join(self.data_root, "user_item_sequences.jsonl")
        if not os.path.isfile(path):
            raise FileNotFoundError("Steam100K requires processed user_item_sequences.jsonl: " + path)
        seen = set()
        with open(path, encoding="utf-8") as f:
            for line_number, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict) or "user" not in row or "items" not in row:
                    raise ValueError("Malformed Steam100K row %d" % line_number)
                user = str(row["user"])
                items = row["items"]
                if user in seen:
                    raise ValueError("Steam100K contains duplicate user: " + user)
                if not isinstance(items, list) or len(items) < 3:
                    raise ValueError("Each Steam100K user requires >=3 chronological interactions: " + user)
                if not all(isinstance(item, (str, int)) and str(item) for item in items):
                    raise ValueError("Non-string/non-integer item ID for Steam100K user " + user)
                seen.add(user)
                # The sequence file MUST already be chronologically ordered.
                self.all_item_seqs[user] = [str(item) for item in items]
        if not self.all_item_seqs:
            raise ValueError("Empty Steam100K sequences: " + path)

    @staticmethod
    def _sort_ids(ids):
        ids = list(ids)
        if all(v.isdigit() for v in ids):
            return sorted(ids, key=lambda v: (int(v), v))
        return sorted(ids)

    def _build_mapping(self):
        item_ids = set()
        for seq in self.all_item_seqs.values():
            item_ids.update(seq)
        item_order = self._sort_ids(item_ids)
        user_order = self._sort_ids(self.all_item_seqs)
        # An explicit mapping file is recommended for checkpoint reproduction.
        path = os.path.join(self.data_root, "id_mapping.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                saved = json.load(f)
            for label, needed in (("item", item_ids), ("user", set(user_order))):
                rev = saved.get("id2" + label)
                if not isinstance(rev, list) or not rev or rev[0] != "[PAD]":
                    raise ValueError("Invalid Steam100K id_mapping.json id2" + label)
                if len(set(rev)) != len(rev) or set(rev[1:]) != needed:
                    raise ValueError("Steam100K id_mapping.json disagrees with sequences: " + label)
                self.id_mapping["id2" + label] = rev
                self.id_mapping[label + "2id"] = {value: i for i, value in enumerate(rev)}
        else:
            for label, order in (("item", item_order), ("user", user_order)):
                self.id_mapping["id2" + label] = ["[PAD]"] + order
                self.id_mapping[label + "2id"] = {value: i for i, value in
                                                 enumerate(self.id_mapping["id2" + label])}

    def _load_item_meta(self):
        path = os.path.join(self.data_root, "item_text.jsonl")
        if not os.path.isfile(path):
            if self.config.get("metadata") == "sentence":
                raise FileNotFoundError("Steam100K sentence metadata requires item_text.jsonl: " + path)
            self.item2meta = {}
            return
        self.item2meta = {}
        with open(path, encoding="utf-8") as f:
            for row in f:
                if not row.strip():
                    continue
                obj = json.loads(row)
                item = str(obj["item"])
                self.item2meta[item] = clean_text(
                    (obj.get("title") or "") + " " + (obj.get("description") or "")).strip()
        if self.config.get("metadata") == "sentence":
            missing = set(self.id_mapping["item2id"]) - set(self.item2meta) - {"[PAD]"}
            if missing:
                raise ValueError("Steam100K lacks text metadata for %d items (e.g. %s)" %
                                 (len(missing), sorted(missing)[0]))
