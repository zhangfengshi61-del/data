"""Fail-closed comparison contract and input fingerprinting."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "SDQ-403"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def baseline_config(dataset, stage):
    description, run_id = ("SDQ", "vae") if stage == "vae" else ("SDQ-T5-VAE", "t5vae")
    return BASE / "logs" / description / f"Amazon2014{dataset}_550_LOU" / run_id / "config.json"


def compare_config(current_path, dataset, stage):
    ref = json.loads(baseline_config(dataset, stage).read_text())
    now = json.loads(Path(current_path).read_text())
    allowed = {"description", "id", "config", "CHECKPOINT_PATH", "LOG_PATH"}
    if stage == "t5":
        allowed.add("sid_vocab_file")
    differences = {k: {"baseline": v, "actual": now.get(k)} for k, v in ref.items()
                   if k not in allowed and now.get(k) != v}
    if differences:
        raise ValueError(f"{dataset}/{stage}: comparison contract mismatch: {differences}")
    return {"passed": True, "checked_fields": sorted(set(ref) - allowed),
            "allowed_differences": sorted(allowed), "baseline": str(baseline_config(dataset, stage)),
            "actual": str(current_path), "method_parameters": {
                k: v for k, v in now.items() if k not in ref}}


def audit_inputs(dataset):
    import pandas as pd
    import torch
    import freerec
    import transformers
    import sklearn
    contract = json.loads((ROOT / "baseline_inputs/PROTOCOL_MANIFEST.json").read_text())
    expected = contract["datasets"][dataset]
    source = ROOT / "data/processed" / dataset
    converted = BASE / "data/Processed" / f"Amazon2014{dataset}_550_LOU"
    report = {"dataset": dataset, "passed": True, "protocol": contract["protocol"], "files": {}}
    total = 0
    for split in ("train", "valid", "test"):
        pq = source / f"{split}.parquet"
        digest = sha256(pq)
        if digest != expected["splits"][split]["file_sha256"]:
            raise ValueError(f"Canonical {dataset}/{split} changed from protocol manifest")
        frame = pd.read_parquet(pq)
        want = [(int(row.user) - 1, int(item) - 1)
                for row in frame.itertuples(index=False)
                for item in (list(row.history) + [row.target] if split == "train" else [row.target])]
        txt = converted / f"{split}.txt"
        observed = list(pd.read_csv(txt, sep="\t").itertuples(index=False, name=None))
        if observed != want:
            raise ValueError(f"freerec {dataset}/{split} rows differ from canonical data")
        if len(frame) != expected["users"]:
            raise ValueError("user count mismatch")
        total += len(want)
        report["files"][str(pq)] = digest
        report["files"][str(txt)] = sha256(txt)
    items = pd.read_csv(converted / "item.txt", sep="\t")
    if items.ITEM.tolist() != list(range(expected["items"])):
        raise ValueError("non-canonical item indexing")
    report.update(users=expected["users"], items=expected["items"], interactions=total)
    paths = [converted / "item.txt", converted / "sentence-t5-xl_title_categories_brand.pkl",
             baseline_config(dataset, "vae"), baseline_config(dataset, "t5"),
             ROOT / "baseline_inputs/PROTOCOL_MANIFEST.json"]
    paths.extend(BASE / name for name in ["train_sdq_vae.py", "train_t5.py", "quantizer.py",
                                         "converter.py", "prepare_local_data.py"])
    paths.extend((ROOT / "SDQ-LARS").glob("*.py"))
    for path in paths:
        report["files"][str(path)] = sha256(path)
    report["versions"] = {"torch": torch.__version__, "freerec": freerec.__version__,
                          "transformers": transformers.__version__, "sklearn": sklearn.__version__}
    return report
