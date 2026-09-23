"""
One-off diagnostic: characterizes the publisher-disjoint split and compares
it against a random split as a control, to isolate whether low accuracy
under the disjoint split is a pipeline bug or a genuine generalization gap.

Usage:
    python scripts/diagnose_split.py --config configs/default.yaml
"""

import argparse

from sklearn.model_selection import train_test_split

from partisan_classifier.config import Config
from partisan_classifier.data.loader import load_article_bias_prediction
from partisan_classifier.data.splits import publisher_disjoint_split
from partisan_classifier.evaluation.metrics import compute_metrics
from partisan_classifier.features.text_features import TfidfFeaturizer
from partisan_classifier.models.baseline import TfidfLogisticBaseline
from partisan_classifier.utils.logging import get_logger

logger = get_logger(__name__)


def describe_splits(train_df, val_df, test_df) -> None:
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info("%s publishers per label:\n%s", name,
                    split_df.groupby("label")["publisher"].nunique())
        logger.info("%s label counts: %s", name,
                    split_df["label"].value_counts().to_dict())


def random_split_control(df) -> dict:
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    featurizer = TfidfFeaturizer()
    X_train = featurizer.fit_transform(train_df["text"].tolist())
    X_val = featurizer.transform(val_df["text"].tolist())

    model = TfidfLogisticBaseline()
    model.fit(X_train, train_df["label"].tolist())
    preds = model.predict(X_val)
    return compute_metrics(val_df["label"].tolist(), preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)

    df = load_article_bias_prediction(cfg.get("data.raw_dir"))
    train_df, val_df, test_df = publisher_disjoint_split(df, seed=cfg.get("seed", 42))

    describe_splits(train_df, val_df, test_df)
    logger.info("Random-split control metrics: %s", random_split_control(df))


if __name__ == "__main__":
    main()