#!/usr/bin/env python3
"""Reproducible student inference cost runner for the paper-aligned ServeSID path.

Examples:
  python ServeSID/scripts/bench_infer_cost.py \
    --checkpoint /models/Beauty/teacher.pth \
    --lens_path /runs/Beauty/gru_final.pth \
    --dataset Amazon2014 --category Beauty --dataset_tag Amazon2014_Beauty \
    --sem_ids /models/Beauty/sem_ids.pt --graph_dir /cache/Beauty \
    --output_dir /results/Beauty_GRU --batch_size 32 --beam_size 50

Each repeat runs an ordinary uninstrumented end-to-end evaluation and
separately a CUDA-synchronized phase profile. Phase times are NOT
treated as unsynchronized production throughput. No teacher inference
benchmark is fabricated by this runner.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def extract_metrics(output):
    for line in output.splitlines():
        if line.startswith("METRICS_JSON "):
            return json.loads(line[len("METRICS_JSON "):])
    raise RuntimeError("Inference did not emit METRICS_JSON:\n" + output[-3000:])


def summarize(values):
    vals = [float(x) for x in values]
    return {"mean": statistics.mean(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else None,
            "runs": vals}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="TIGER checkpoint (encoder source)")
    p.add_argument("--lens_path", required=True, help="Student checkpoint")
    p.add_argument("--dataset", required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--dataset_tag", required=True)
    p.add_argument("--sem_ids", required=True)
    p.add_argument("--graph_dir", required=True)
    p.add_argument("--valid_set")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument("--final_top_n", type=int, default=50)
    p.add_argument("--warmup_batches", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--fp32", action="store_true")
    args = p.parse_args(argv)
    if args.repeats < 1 or args.warmup_batches < 0:
        p.error("repeats must be >= 1 and warmup_batches >= 0")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_code = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_code) + os.pathsep + env.get("PYTHONPATH", "")
    base = [sys.executable, "-m", "sid_mlp.cli.infer",
            "--checkpoint", args.checkpoint, "--lens_path", args.lens_path,
            "--dataset", args.dataset, "--category", args.category,
            "--dataset_tag", args.dataset_tag, "--sem_ids", args.sem_ids,
            "--graph_dir", args.graph_dir,
            "--split", "test", "--batch_size", str(args.batch_size),
            "--beam_size", str(args.beam_size),
            "--final_top_n", str(args.final_top_n)]
    if args.valid_set:
        base += ["--valid_set", args.valid_set]
    if args.fp32:
        base.append("--fp32")

    def run(cmd, logfile):
        finished = subprocess.run(cmd, env=env, cwd=str(repo_code),
                                  text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, check=False)
        logfile.write_text(finished.stdout, encoding="utf-8")
        if finished.returncode:
            raise RuntimeError(f"Inference exited {finished.returncode}: {logfile}\n"
                               + finished.stdout[-3000:])
        return extract_metrics(finished.stdout)

    regular = []
    profiled = []
    for i in range(args.repeats):
        base_metrics = run(base, out_dir / f"run{i+1}_regular.log")
        profile_file = out_dir / f"run{i+1}_profile.json"
        prof_metrics = run(base + ["--cost_json", str(profile_file),
                                  "--cost_warmup_batches", str(args.warmup_batches)],
                           out_dir / f"run{i+1}_profile.log")
        if base_metrics["total"] != prof_metrics["total"]:
            raise RuntimeError("Profile and regular runs evaluated different request counts")
        for field in ("recall10", "ndcg10"):
            if abs(base_metrics[field] - prof_metrics[field]) > 1e-9:
                raise RuntimeError(f"Metric changed between regular/profiled evaluation: {field}")
        prof = json.loads(profile_file.read_text(encoding="utf-8"))
        if prof["measured_requests"] < 1:
            raise RuntimeError("No requests left after profiling warmup; increase dataset size")
        regular.append(base_metrics)
        profiled.append(prof)

    if len({int(r["total"]) for r in regular}) != 1:
        raise RuntimeError("Evaluated request count varies across repetitions")
    if len({str(r["gpu_name"]) for r in profiled}) != 1:
        raise RuntimeError("GPU model varies across repetitions")
    summary = {
        "method": "student_inference",
        "measurement": "run-by-run repeat; full test split; no training",
        "protocol": {"batch_size": args.batch_size,
                     "beam_width": args.beam_size, "final_top_n": args.final_top_n,
                     "warmup_batches_for_sync_profile": args.warmup_batches,
                     "repeats": args.repeats,
                     "gpu": profiled[0]["gpu_name"],
                     "gpu_total_memory_bytes": profiled[0]["gpu_total_memory_bytes"],
                     "dtype": profiled[0]["dtype"],
                     "evaluated_requests": regular[0]["total"],
                     "regular_elapsed_excludes_initial_model_loading": True,
                     "regular_elapsed_includes_dataloader_and_ranking": True,
                     "phase_timer_syncs_after_each_stage": True,
                     "peak_memory_scope": profiled[0]["memory_scope"]},
        "checkpoints": {"student": str(Path(args.lens_path).resolve()),
                        "teacher": str(Path(args.checkpoint).resolve())},
        "results": {
            "unprofiled_requests_per_second": summarize([r["throughput"] for r in regular]),
            "unprofiled_amortized_ms_per_request": summarize(
                [1000.0 * r["elapsed_sec"] / r["total"] for r in regular]),
            "ndcg_at_10": summarize([r["ndcg10"] for r in regular]),
            "synchronized_amortized_ms_per_request": summarize(
                [r["amortized_ms_per_request"] for r in profiled]),
            "full_pipeline_peak_allocated_gib": summarize(
                [r["peak_allocated_bytes"] / (1024**3) for r in profiled]),
            "profiled_stage_ms_per_request": {
                k: summarize([1000.0 * r[k + "_sec"] / r["measured_requests"]
                              for r in profiled])
                for k in ("transfer_and_gt", "frozen_encoder",
                          "student_context", "sid_beam_and_lookup", "cpu_ranking")
            },
        },
    }
    (out_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
