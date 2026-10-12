#!/usr/bin/env python3
"""Count the trainable SID-student parameters used in ServeSID manuscript.

The pretrained SID embedding is frozen (excluded), and the shared TIGER
encoder is not counted in decoder parameter totals.

Example:
    PYTHONPATH=ServeSID python ServeSID/scripts/count_student_parameters.py
"""
import argparse
import torch
from sid_mlp.models import SIDMLP, SIDGRU, SIDCNN


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_reference_students(embedding):
    """Use model and CLI architecture settings documented in the manuscript."""
    shared = dict(embed_dim=128, head_hidden=512, head_layers=1, dropout=0.0)
    return {
        "MLP": SIDMLP(
            embedding, d_model=128, num_heads=4, ffn_dim=1024,
            attn_dim=384, **shared,
        ),
        "GRU": SIDGRU(
            embedding, gru_hidden=384, gru_layers=1, **shared,
        ),
        "CNN": SIDCNN(
            embedding, conv_dim=256, kernel_size=3, conv_layers=2,
            **shared,
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding-vocab", type=int, default=1024)
    args = ap.parse_args()
    # Vocabulary size affects only the frozen embedding; trainable counts
    # are therefore invariant to the number of SID tokens.
    if args.embedding_vocab < 1024:
        ap.error("--embedding-vocab must cover four 256-token codebooks")
    embedding = torch.zeros((args.embedding_vocab, 128), dtype=torch.float32)
    models = build_reference_students(embedding)
    expected = {"MLP": 1_663_744, "GRU": 2_304_000, "CNN": 1_744_896}
    for name, model in models.items():
        n = count_trainable(model)
        print(f"{name}: {n:,} trainable parameters ({n/1e6:.3f}M)")
        if n != expected[name]:
            raise AssertionError(
                f"{name}: model structure changed; update the manuscript "
                f"(expected {expected[name]:,}, got {n:,})"
            )
    print("PASS: three manuscript parameter counts match current student models")


if __name__ == "__main__":
    main()
