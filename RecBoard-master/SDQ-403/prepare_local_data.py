#!/usr/bin/env python
"""Build freerec-format ``Processed/Amazon2014*_550_LOU`` data from the
RecBoard-master canonical leave-two-out parquet splits.

The canonical data lives under ``RecBoard-master/data/processed/<Category>``
(``train/valid/test.parquet`` + ``item_text.jsonl``).  freerec expects
``<root>/Processed/Amazon2014<Category>_550_LOU/{train,valid,test,item}.txt``
where each interaction row is ``USER<TAB>ITEM`` with contiguous 0-based ids.

The mapping is exact: ``train.txt`` holds every training interaction
(history + train target), ``valid.txt`` the single valid target per user and
``test.txt`` the single test target per user.  This reproduces the canonical
leave-two-out protocol, so ``train+valid+test`` equals the canonical
interaction count.
"""
import argparse
import json
import os

import pandas as pd

CATEGORIES = ("Beauty", "Sports", "Toys")


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    elif isinstance(value, dict):
        value = " ".join(f"{k}: {v}" for k, v in value.items())
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def load_items(path: str):
    items = {}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = int(record["item"])
            items[item_id] = (
                _clean(record.get("title")),
                _clean(record.get("categories")),
                _clean(record.get("brand")),
            )
    return items


def build(canonical_root: str, out_root: str, category: str) -> None:
    src = os.path.join(canonical_root, category)
    dst = os.path.join(out_root, "Processed", f"Amazon2014{category}_550_LOU")
    os.makedirs(dst, exist_ok=True)

    items = load_items(os.path.join(src, "item_text.jsonl"))
    max_item = max(items)
    if len(items) != max_item:
        raise ValueError(
            f"{category}: item ids are not contiguous 1..N "
            f"(count={len(items)}, max={max_item})"
        )

    n_users = 0
    split_sizes = {}
    for split in ("train", "valid", "test"):
        frame = pd.read_parquet(os.path.join(src, f"{split}.parquet"))
        rows = []
        for record in frame.itertuples(index=False):
            user = int(record.user) - 1
            # train.txt carries the whole train prefix (history + target);
            # valid/test.txt only carry the single held-out target per user,
            # because freerec builds the history from train() for those modes.
            seq = list(record.history) + [int(record.target)] if split == "train" else [int(record.target)]
            for item in seq:
                rows.append((user, int(item) - 1))
        n_users = max(n_users, int(frame["user"].max()))
        split_sizes[split] = len(rows)
        with open(os.path.join(dst, f"{split}.txt"), "w", encoding="utf-8") as file:
            file.write("USER\tITEM\n")
            for user, item in rows:
                file.write(f"{user}\t{item}\n")

    with open(os.path.join(dst, "item.txt"), "w", encoding="utf-8") as file:
        file.write("ITEM\tTITLE\tCATEGORIES\tBRAND\n")
        for token in range(max_item):
            title, categories, brand = items[token + 1]
            file.write(f"{token}\t{title}\t{categories}\t{brand}\n")

    print(
        f"[{category}] users={n_users} items={max_item} "
        f"train={split_sizes['train']} valid={split_sizes['valid']} "
        f"test={split_sizes['test']} -> {dst}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        default="/data/fszhang/RecBoard-master/data/processed",
    )
    parser.add_argument(
        "--out-root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(CATEGORIES))
    args = parser.parse_args()

    for category in args.datasets:
        build(args.canonical_root, args.out_root, category)


if __name__ == "__main__":
    main()
