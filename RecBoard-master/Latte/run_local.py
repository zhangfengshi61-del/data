#!/usr/bin/env python
"""Local-protocol runner for Latte/PSID on the Amazon-2014 datasets.

Avoids `uv`, HuggingFace downloads and wandb. The dataset adapter is
``genrec.datasets.Amazon2014`` which reads experiments/data/processed.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Latte")
    parser.add_argument("--dataset", default="Amazon2014")
    parser.add_argument("--category", default="Beauty")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n_latent_tokens", type=int, default=8)
    parser.add_argument("--vq_method", default="rqkmeans")
    parser.add_argument("--aggregation_method", default="agg_max")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--tracker", default="tensorboard")
    parser.add_argument("--smoke_samples", type=int, default=None)
    args, unparsed = parser.parse_known_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from genrec.pipeline import Pipeline

    kwargs = dict(
        category=args.category,
        tracker=args.tracker,
        epochs=args.epochs,
        eval_interval=args.eval_interval,
        patience=args.patience,
        n_latent_tokens=args.n_latent_tokens,
        vq_method=args.vq_method,
        aggregation_method=args.aggregation_method,
        lr=args.lr,
        num_beams=args.num_beams,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        topk=[5, 10, 20],
        metrics=["ndcg", "recall", "mrr"],
        val_metric="ndcg@10",
    )
    if args.smoke_samples:
        kwargs["smoke_samples"] = args.smoke_samples
        kwargs["dataloader_num_workers"] = 0
    if args.data_root:
        kwargs["data_root"] = args.data_root

    pipeline = Pipeline(
        model_name=args.model,
        dataset_name=args.dataset,
        config_dict=kwargs,
    )
    result = pipeline.run()
    print("RESULT", result)


if __name__ == "__main__":
    main()
