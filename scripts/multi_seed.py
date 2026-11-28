#!/usr/bin/env python
"""Measure how much transformer validation performance varies with the seed.

Examples (run from the project root):
    python scripts/multi_seed.py --smoke-test
    python scripts/multi_seed.py --seeds 42 123 456

The official Baly test split is never used for training, checkpoint selection,
or reporting from this script. The smoke test only checks that the pipeline runs;
its scores must not be reported as model performance.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    """Aggregate completed runs; standard deviation is a *sample* SD."""
    if not rows:
        raise ValueError("At least one completed run is required.")

    f1_scores = [float(row["macro_f1"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "n_runs": len(rows),
        "macro_f1_mean": statistics.mean(f1_scores),
        "macro_f1_std": statistics.stdev(f1_scores) if len(rows) > 1 else None,
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_std": statistics.stdev(accuracies) if len(rows) > 1 else None,
    }


def _json_default(value):
    """Support NumPy scalar values returned by scikit-learn metrics."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", default="outputs/multi_seed")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Train 30/validate 15 articles with one seed and one epoch.",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct.")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size and learning rate must be positive.")

    # Keep heavy dependencies out of the module import so aggregation can be
    # tested without installing PyTorch, datasets, or transformers.
    from partisan_classifier.config import Config
    from partisan_classifier.data.loader import load_media_split
    from partisan_classifier.evaluation.metrics import compute_metrics
    from partisan_classifier.models.transformer_model import TransformerClassifier
    from partisan_classifier.utils.logging import set_seed

    cfg = Config.load(args.config)
    model_name = cfg.get("model.name", "roberta-base")
    train_df, valid_df, _ = load_media_split(cfg.get("data.raw_dir", "data/raw"))

    if args.smoke_test:
        train_df = train_df.head(30)
        valid_df = valid_df.head(15)
        seeds = args.seeds[:1]
        epochs = 1
        print("SMOKE TEST: 30 training articles, 15 validation articles, one seed/epoch.")
    else:
        seeds = args.seeds
        epochs = args.epochs

    if train_df.empty or valid_df.empty:
        raise ValueError("Training and validation datasets must be nonempty.")

    output_dir = Path(args.output_dir) / ("smoke" if args.smoke_test else "full")
    if (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"{output_dir / 'summary.json'} already exists; use another --output-dir "
            "to preserve the previous experiment."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    valid_texts = valid_df["text"].tolist()
    valid_labels = valid_df["label"].tolist()

    rows = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"{run_dir} already contains data; use a different --output-dir "
                "to avoid overwriting it."
            )

        print(f"\nTraining seed {seed} ({epochs} epoch(s))...")
        set_seed(seed)
        model = TransformerClassifier(
            model_name=model_name,
            output_dir=str(run_dir),
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
        )
        model.fit(train_texts, train_labels, valid_texts, valid_labels)
        predictions = model.predict(valid_texts)
        metrics = compute_metrics(valid_labels, predictions)

        row = {
            "seed": seed,
            "accuracy": float(metrics["accuracy"]),
            "macro_f1": float(metrics["macro_f1"]),
        }
        rows.append(row)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8"
        )
        print(f"Seed {seed}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}")

        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    aggregate = summarize(rows)
    result = {
        "dataset": "Baly official media validation split",
        "smoke_test": args.smoke_test,
        "model_name": model_name,
        "training_articles": len(train_df),
        "validation_articles": len(valid_df),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": epochs,
        "seeds": seeds,
        "runs": rows,
        "summary": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCompleted {aggregate['n_runs']} run(s).")
    print(f"Mean validation macro-F1: {aggregate['macro_f1_mean']:.4f}")
    if aggregate["macro_f1_std"] is not None:
        print(f"Sample standard deviation: {aggregate['macro_f1_std']:.4f}")
    else:
        print("Standard deviation: n/a (one smoke-test run)")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
