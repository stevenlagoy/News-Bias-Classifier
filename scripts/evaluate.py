#!/usr/bin/env python
"""
CLI wrapper: evaluate trained and baseline models on test / OOD splits,
plus diagnostics on the eval sets themselves.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
"""

import argparse

import pandas as pd

from partisan_classifier.config import Config
from partisan_classifier.data.loader import (
    load_article_bias_prediction,
    load_ood_secondary,
    load_qbias,
)
from partisan_classifier.data.splits import publisher_disjoint_split
from partisan_classifier.evaluation.metrics import compute_metrics
from partisan_classifier.models.baseline import MajorityClassBaseline, TfidfLogisticBaseline
from partisan_classifier.models.pretrained_baseline import PoliticalBiasBertBaseline
from partisan_classifier.models.transformer_model import TransformerClassifier
from partisan_classifier.utils.logging import get_logger

logger = get_logger(__name__)

_LOADERS = {
    "majority": MajorityClassBaseline.load,
    "tfidf_logreg": TfidfLogisticBaseline.load,
    "transformer": TransformerClassifier.load,
}


def describe_text_lengths(eval_sets: dict[str, pd.DataFrame]) -> None:
    """Confirms or rules out a genre/length mismatch between eval sets."""
    for name, df in eval_sets.items():
        lengths = df["text"].str.len()
        logger.info(
            "%s text length: mean=%.0f median=%.0f min=%d max=%d (n=%d)",
            name, lengths.mean(), lengths.median(), lengths.min(), lengths.max(), len(df),
        )


def qbias_topic_breakdown(model, qbias_df: pd.DataFrame) -> None:
    """
    Splits Qbias accuracy by topic tag, to test whether argumentative
    topics (immigration, gun control) carry more partisan signal than
    procedural/economic ones (debt ceiling, budget).
    """
    if "tags" not in qbias_df.columns:
        logger.info("Skipping Qbias topic breakdown: no tags column available.")
        return

    argumentative_keywords = ["Immigration", "Gun Control And Gun Rights", "Abortion", "Protests"]
    procedural_keywords = ["Economic Policy", "Federal Budget", "Government Shutdown", "Banking And Finance"]

    def matches(tags_str: str, keywords: list[str]) -> bool:
        return any(k in str(tags_str) for k in keywords)

    argumentative_df = qbias_df[qbias_df["tags"].apply(lambda t: matches(t, argumentative_keywords))]
    procedural_df = qbias_df[qbias_df["tags"].apply(lambda t: matches(t, procedural_keywords))]

    for label, subset in [("argumentative-topic", argumentative_df), ("procedural-topic", procedural_df)]:
        if len(subset) == 0:
            logger.info("Qbias %s subset: no matching rows.", label)
            continue
        preds = model.predict(subset["text"].tolist())
        metrics = compute_metrics(subset["label"].tolist(), preds)
        logger.info(
            "Qbias %s subset (n=%d): accuracy=%.3f macro_f1=%.3f",
            label, len(subset), metrics["accuracy"], metrics["macro_f1"],
        )


def print_summary_table(results: dict[tuple[str, str], dict]) -> None:
    header = f"{'model':<28} {'eval set':<22} {'accuracy':>9} {'macro_f1':>9}"
    logger.info(header)
    logger.info("-" * len(header))
    for (model_name, set_name), metrics in results.items():
        logger.info(
            "%-28s %-22s %9.3f %9.3f",
            model_name, set_name, metrics["accuracy"], metrics["macro_f1"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    df = load_article_bias_prediction(cfg.get("data.raw_dir"))
    train_df, _, test_df = publisher_disjoint_split(df, seed=cfg.get("seed", 42))

    eval_sets = {
        "in_distribution_test": test_df,
        "qbias_ood": load_qbias(cfg.get("data.raw_dir")),
        "ood_secondary": load_ood_secondary(cfg.get("data.raw_dir")),
    }

    logger.info("Eval set text length diagnostics:")
    describe_text_lengths(eval_sets)

    models = {
        "majority_baseline": MajorityClassBaseline().fit([], train_df["label"].tolist()),
        "political_bias_bert_baseline": PoliticalBiasBertBaseline(),
    }

    checkpoint_dir = cfg.get("output.checkpoint_dir")
    model_type = cfg.get("model.type", "tfidf_logreg")
    if checkpoint_dir:
        models["your_model"] = _LOADERS[model_type](checkpoint_dir)

    results = {}
    for model_name, model in models.items():
        for set_name, eval_df in eval_sets.items():
            preds = model.predict(eval_df["text"].tolist())
            metrics = compute_metrics(eval_df["label"].tolist(), preds)
            results[(model_name, set_name)] = metrics
            logger.info("%s on %s: %s", model_name, set_name, metrics)

    logger.info("Summary:")
    print_summary_table(results)

    logger.info("Qbias topic-stratified breakdown (your_model):")
    if "your_model" in models:
        qbias_topic_breakdown(models["your_model"], eval_sets["qbias_ood"])


if __name__ == "__main__":
    main()