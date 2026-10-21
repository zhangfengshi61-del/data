#!/usr/bin/env python3
"""Verify existence, hashes, and identities of paper experiment artifacts.

This is a provenance check, NOT a reconstruction of missing seeds, a rerun,
or a replacement for independently auditing the evaluation implementation.
See verification/PAPER_CODE_ALIGNMENT.md for the required manifest schema.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest requires nonempty runs list")
    seen = set()
    verified = []
    errors = []
    for index, run in enumerate(runs):
        ident = tuple(str(run.get(key, "")) for key in
                      ("dataset", "arch", "method", "seed"))
        label = "/".join(ident)
        if not all(ident) or ident in seen:
            errors.append("duplicate or incomplete experiment identity: " + label)
        seen.add(ident)
        if not run.get("code_commit") or len(run["code_commit"]) != 40:
            errors.append("missing 40-character code_commit for " + label)
        summary = {"id": label, "artifacts": {}}
        for field in ("checkpoint", "eval_json", "teacher_dump",
                      "semantic_ids", "train_manifest"):
            item = run.get(field)
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                errors.append("missing path/sha256 for " + label + " " + field)
                continue
            path = (manifest_path.parent / item["path"]).resolve()
            if not path.is_file():
                errors.append("missing artifact: " + label + " " + field + " " + str(path))
                continue
            digest = sha256_file(path)
            if digest.lower() != str(item["sha256"]).lower():
                errors.append("SHA256 mismatch: " + label + " " + field)
            summary["artifacts"][field] = {"file": str(path), "sha256": digest}
        verified.append(summary)
    return {"experiment_count": len(runs), "verified_artifacts": verified,
            "errors": errors, "ok": not errors}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()
    report = verify(args.manifest.resolve())
    value = json.dumps(report, indent=2) + "\n"
    print(value)
    if args.output_json:
        args.output_json.write_text(value, encoding="utf-8")
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
