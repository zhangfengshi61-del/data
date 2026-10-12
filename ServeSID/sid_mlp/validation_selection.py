"""Strict validation-only checkpoint selection shared by all SID students.

No inference or model dependencies, so the policy can be tested without GPUs.
"""
import math


def require_validation_split(tokenized_datasets):
    """Never substitute test queries when the validation split is missing."""
    if "val" not in tokenized_datasets:
        raise RuntimeError(
            "A validation split is required for best-Val-NDCG@10 selection; "
            "refusing to fall back to the test split."
        )
    if len(tokenized_datasets["val"]) == 0:
        raise RuntimeError("Validation split is empty; refusing checkpoint selection.")
    return "val"


def validate_selection_settings(args):
    """Reportable selection must use the full validation set, not a debug subset."""
    limit = getattr(args, "beam_val_max_batches", None)
    if limit is not None and int(limit) <= 0:
        raise ValueError("beam_val_max_batches must be positive when provided")
    if limit is not None and not getattr(args, "test", False):
        raise ValueError(
            "Published best-Val checkpoints must score the complete validation set. "
            "Remove --beam_val_max_batches; --test permits debug-only partial evaluation."
        )
    if int(getattr(args, "epochs", 0)) <= 0:
        raise ValueError("epochs must be positive")


def should_beam_validate(epoch, epochs, interval=5):
    if interval < 1:
        raise ValueError("interval must be positive")
    return epoch % interval == 0 or epoch == epochs


def save_if_best_val_ndcg10(*, metrics, epoch, previous_best, path, payload, save):
    """Persist only strictly improved, finite, nonempty validation NDCG@10.

    Caller must supply metrics computed on the validation split and a saver,
    e.g. torch.save. Returns (best_so_far, was_saved).
    """
    if metrics is None or "ndcg10" not in metrics:
        raise RuntimeError("Missing validation NDCG@10; refusing checkpoint selection")
    n = int(metrics.get("total", 0))
    if n <= 0:
        raise RuntimeError("Empty validation recommendation evaluation")
    value = float(metrics["ndcg10"])
    if not math.isfinite(value):
        raise RuntimeError("Non-finite validation NDCG@10")
    if value > float(previous_best):
        snapshot = dict(payload)
        snapshot.update({
            "epoch": int(epoch),
            "selection_split": "val",
            "selection_metric": "val_ndcg10",
            "selection_value": value,
            "selection_num_requests": n,
            "checkpoint_role": "best_val_ndcg10",
        })
        save(snapshot, path)
        return value, True
    return float(previous_best), False
