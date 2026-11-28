#!/usr/bin/env python
"""
Analyze classification errors on the official media split.

Produces:
- Per-article predictions
- Misclassified examples
- Confusion matrices
- Per-publisher error rates
- Classification metrics

Use validation for development analysis.
Reserve test for final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from partisan_classifier.data.loader import load_media_split
from partisan_classifier.models.baseline import (
    MajorityClassBaseline,
    TfidfLogisticBaseline,
)
from partisan_classifier.utils.logging import set_seed


LABELS = ["left", "center", "right"]


def analyze_model(
    model_name: str,
    model,
    eval_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Evaluate one model and save detailed error analysis."""

    print(f"\nAnalyzing model: {model_name}")

    y_true = eval_df["label"].tolist()
    y_pred = model.predict(eval_df["text"].tolist())

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # Create a DataFrame containing predictions.
    predictions_df = pd.DataFrame(
        {
            "article_id": eval_df["article_id"].tolist(),
            "publisher": eval_df["publisher"].tolist(),
            "actual_label": y_true,
            "predicted_label": y_pred,
            "text_preview": [
                str(text)[:300]
                for text in eval_df["text"].tolist()
            ],
        }
    )

    predictions_df["correct"] = (
        predictions_df["actual_label"]
        == predictions_df["predicted_label"]
    )

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for every article.
    predictions_df.to_csv(
        model_dir / "predictions.csv",
        index=False,
    )

    # Save only incorrectly classified articles.
    errors_df = predictions_df[
        ~predictions_df["correct"]
    ].copy()

    errors_df.to_csv(
        model_dir / "misclassified_articles.csv",
        index=False,
    )

    print(f"Total articles: {len(predictions_df)}")
    print(f"Misclassified articles: {len(errors_df)}")

    # Save confusion matrix.
    confusion_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )

    confusion_df.to_csv(
        model_dir / "confusion_matrix.csv"
    )

    print("\nConfusion matrix:")
    print(confusion_df)

    # Calculate error rates for each publisher.
    publisher_analysis = (
        predictions_df.groupby("publisher", dropna=False)
        .agg(
            total_articles=("correct", "size"),
            accuracy=("correct", "mean"),
        )
        .reset_index()
    )

    publisher_analysis["error_rate"] = (
        1 - publisher_analysis["accuracy"]
    )

    publisher_analysis = publisher_analysis.sort_values(
        ["error_rate", "total_articles"],
        ascending=[False, False],
    )

    publisher_analysis.to_csv(
        model_dir / "publisher_errors.csv",
        index=False,
    )

    # Identify the most common classification mistakes.
    error_patterns = (
        errors_df.groupby(
            ["actual_label", "predicted_label"]
        )
        .size()
        .sort_values(ascending=False)
    )

    print("\nMost common classification errors:")
    print(error_patterns.head(5))

    # Save summary metrics.
    metrics = {
        "model": model_name,
        "total_articles": len(predictions_df),
        "misclassified_articles": len(errors_df),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_order": LABELS,
    }

    with (model_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {model_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze news-bias classification errors."
    )

    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
        help="Dataset split to analyze (default: valid).",
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Location of the raw datasets.",
    )

    args = parser.parse_args()

    set_seed(42)

    print("Loading dataset...")

    train_df, val_df, test_df = load_media_split(
        args.raw_dir
    )

    eval_df = (
        val_df if args.split == "valid" else test_df
    )

    print(f"Training articles: {len(train_df)}")
    print(f"Evaluation split: {args.split}")
    print(f"Evaluation articles: {len(eval_df)}")

    output_dir = (
        Path("outputs")
        / "error_analysis"
        / args.split
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nTraining baseline models...")

    majority = MajorityClassBaseline()

    majority.fit(
        [],
        train_df["label"].tolist(),
    )

    tfidf = TfidfLogisticBaseline()

    tfidf.fit(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
    )

    models = {
        "majority": majority,
        "tfidf_logreg": tfidf,
    }

    for model_name, model in models.items():
        analyze_model(
            model_name,
            model,
            eval_df,
            output_dir,
        )

    print("\nError analysis completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()