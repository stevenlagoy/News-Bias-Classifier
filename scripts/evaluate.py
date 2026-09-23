#!/usr/bin/env python
"""
CLI wrapper: evaluate a trained model on test / OOD splits.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
"""

import argparse

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

    models = {"political_bias_bert_baseline": PoliticalBiasBertBaseline()}

    checkpoint_dir = cfg.get("output.checkpoint_dir")
    model_type = cfg.get("model.type", "tfidf_logreg")
    if checkpoint_dir:
        models["your_model"] = _LOADERS[model_type](checkpoint_dir)

    # # tfidf_logreg needs the same vectorizer fit on the same train text
    # # before it can transform eval text; refit it here since the
    # # vectorizer itself isn't persisted by TfidfLogisticBaseline.save().
    # featurizer = None
    # if model_type == "tfidf_logreg":
    #     featurizer = TfidfFeaturizer()
    #     featurizer.fit(train_df["text"].tolist())

    for model_name, model in models.items():
        for set_name, eval_df in eval_sets.items():
            # texts = eval_df["text"].tolist()
            # if model_name == "your_model" and featurizer is not None:
            #     inputs = featurizer.transform(texts)
            # else:
            #     inputs = texts
            # preds = model.predict(inputs)
            preds = model.predict(eval_df["text"].tolist())
            metrics = compute_metrics(eval_df["label"].tolist(), preds)
            logger.info("%s on %s: %s", model_name, set_name, metrics)


if __name__ == "__main__":
    main()
