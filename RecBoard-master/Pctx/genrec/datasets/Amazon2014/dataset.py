"""Amazon-2014 (McAuley 5-core) dataset adapter for the genrec pipeline.

Latte/PSID officially target Amazon-Reviews-2023 and download their processed
splits from HuggingFace. This adapter instead reads the canonical TIGER-protocol
artifacts under ``data/processed/<category>``:

  item_text.jsonl             item id -> metadata
  user_item_sequences.jsonl   user -> full chronological item sequence

The full sequence is exposed through ``all_item_seqs`` so the inherited
``_leave_one_out`` split reproduces our leave-two-out protocol:
  train = seq[:-2], val = seq[:-1], test = seq.
"""
import json
import os

from genrec.dataset import AbstractDataset
from genrec.utils import clean_text


class Amazon2014(AbstractDataset):
    CATEGORIES = ["Beauty", "Sports", "Toys"]

    def __init__(self, config: dict):
        super(Amazon2014, self).__init__(config)

        self.category = config["category"]
        assert self.category in self.CATEGORIES, (
            f"Category {self.category} not available. Available: {self.CATEGORIES}"
        )
        self.log(f"[DATASET] Amazon-2014 5-core for category: {self.category}")

        data_root = config.get("data_root")
        if data_root is None:
            data_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "processed")
            )
        self.data_root = os.path.abspath(data_root)
        self.cache_dir = os.path.join(config["cache_dir"], "Amazon2014", self.category)
        os.makedirs(os.path.join(self.cache_dir, "processed"), exist_ok=True)

        self._load_raw()

    def _metadata_sentence(self, row: dict) -> str:
        title = clean_text(row.get("title", "") or "")
        description = clean_text(row.get("description", "") or "")
        categories = clean_text(row.get("categories", []) or "")
        brand = clean_text(row.get("brand", "") or "")
        parts = [p for p in (title, description, categories, brand) if p]
        return ". ".join(parts) + "."

    def _load_raw(self):
        proc = os.path.join(self.data_root, self.category)
        if not os.path.isdir(proc):
            raise FileNotFoundError(f"Processed data directory not found: {proc}")

        self.item2meta = {}
        with open(os.path.join(proc, "item_text.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.item2meta[str(int(row["item"]))] = self._metadata_sentence(row)

        self.all_item_seqs = {}
        with open(os.path.join(proc, "user_item_sequences.jsonl"), encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                self.all_item_seqs[str(row["user"])] = [str(int(x)) for x in row["items"]]

        for user in sorted(self.all_item_seqs, key=int):
            self.id_mapping["user2id"][user] = len(self.id_mapping["user2id"])
            self.id_mapping["id2user"].append(user)
        for item in sorted(self.item2meta, key=int):
            self.id_mapping["item2id"][item] = len(self.id_mapping["item2id"])
            self.id_mapping["id2item"].append(item)

        assert len(self.id_mapping["id2item"]) - 1 == len(self.item2meta), (
            "Every item must have metadata for sentence tokenization."
        )
        self.log(
            f"[DATASET] Loaded {len(self.all_item_seqs):,} users and "
            f"{len(self.item2meta):,} items from {proc}"
        )
