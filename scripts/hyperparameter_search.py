#!/usr/bin/env python
"""
Run a bounded transformer hyperparameter search.

The validation split is used to select hyperparameters.
The test split is intentionally never used.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from partisan_classifier.config import Config
from partisan_classifier.data.loader import load_media_split
from partisan_classifier.models.transformer_model import TransformerClassifier
from partisan_classifier.utils.logging import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run on a tiny subset with one configuration for testing.",
    )
    args = parser.parse_args()

    search_config_path = Path("configs/hyperparameter_search.yaml")

    with search_config_path.open("r", encoding="utf-8") as f:
        search_config = yaml.safe_load(f)

    base_config = Config.load("configs/default.yaml")
    raw_dir = Path(base_config.get("data.raw_dir"))

    if args.smoke_test:
        print(
            "SMOKE TEST: loading only 30 training "
            "and 15 validation articles."
        )

        train_ids = pd.read_csv(
            raw_dir
            / "article-bias-prediction"
            / "data"
            / "splits"
            / "media"
            / "train.tsv",
            sep="\t",
        ).head(30)

        val_ids = pd.read_csv(
            raw_dir
            / "article-bias-prediction"
            / "data"
            / "splits"
            / "media"
            / "valid.tsv",
            sep="\t",
        ).head(15)

        json_dir = (
            raw_dir
            / "article-bias-prediction"
            / "data"
            / "jsons"
        )

        train_id_set = set(train_ids["ID"])
        val_id_set = set(val_ids["ID"])

        train_rows = []
        val_rows = []

        for json_file in json_dir.glob("*.json"):
            if len(train_rows) >= 30 and len(val_rows) >= 15:
                break

            with json_file.open("r", encoding="utf-8") as f:
                article = json.load(f)

            article_id = article.get("ID")

            label = (
                article.get("bias_text")
                or {0: "left", 1: "center", 2: "right"}.get(
                    article.get("bias")
                )
            )

            if article_id in train_id_set and len(train_rows) < 30:
                train_rows.append(
                    {
                        "text": article.get("content", ""),
                        "label": label,
                    }
                )

            elif article_id in val_id_set and len(val_rows) < 15:
                val_rows.append(
                    {
                        "text": article.get("content", ""),
                        "label": label,
                    }
                )

        train_df = pd.DataFrame(train_rows)
        val_df = pd.DataFrame(val_rows)

    else:
        train_df, val_df, _ = load_media_split(
            base_config.get("data.raw_dir")
        )

    learning_rates = search_config["learning_rates"]
    batch_sizes = search_config["batch_sizes"]
    epochs_list = search_config["epochs"]

    if args.smoke_test:
        # The smoke test checks that the entire search pipeline works.
        # It does not need to test every hyperparameter combination.
        combinations = [
            (
                learning_rates[0],
                batch_sizes[0],
                epochs_list[0],
            )
        ]
    else:
        combinations = list(
            itertools.product(
                learning_rates,
                batch_sizes,
                epochs_list,
            )
        )

    results_dir = Path("outputs/hyperparameter_search")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(
        f"Running {len(combinations)} "
        "hyperparameter configurations."
    )

    for run_number, (learning_rate, batch_size, epochs) in enumerate(
        combinations,
        start=1,
    ):
        print(
            f"\nRun {run_number}/{len(combinations)}: "
            f"learning_rate={learning_rate}, "
            f"batch_size={batch_size}, "
            f"epochs={epochs}"
        )

        set_seed(base_config.get("seed", 42))

        run_dir = results_dir / (
            f"lr_{learning_rate}_batch_{batch_size}_epochs_{epochs}"
        )

        model = TransformerClassifier(
            model_name=base_config.get(
                "model.name",
                "roberta-base",
            ),
            output_dir=str(run_dir),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=base_config.get("seed", 42),
        )

        model.fit(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            val_df["text"].tolist(),
            val_df["label"].tolist(),
        )

        metrics_path = run_dir / "validation_metrics.json"

        with metrics_path.open("r", encoding="utf-8") as f:
            validation_history = json.load(f)

        best_macro_f1 = max(
            record["validation_macro_f1"]
            for record in validation_history
        )

        results.append(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "best_validation_macro_f1": best_macro_f1,
            }
        )

    results.sort(
        key=lambda row: row["best_validation_macro_f1"],
        reverse=True,
    )

    results_path = results_dir / "results.csv"

    with results_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "learning_rate",
                "batch_size",
                "epochs",
                "best_validation_macro_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    best = results[0]

    print("\nHyperparameter search complete.")
    print(f"Results saved to: {results_path}")
    print("\nBest configuration:")
    print(f"Learning rate: {best['learning_rate']}")
    print(f"Batch size: {best['batch_size']}")
    print(f"Epochs: {best['epochs']}")
    print(
        "Best validation macro-F1: "
        f"{best['best_validation_macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()