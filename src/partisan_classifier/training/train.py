"""
Training entry point, driven by a config file.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

from partisan_classifier.config import Config
from partisan_classifier.data.loader import load_article_bias_prediction
from partisan_classifier.data.splits import publisher_disjoint_split
from partisan_classifier.evaluation.metrics import compute_metrics
from partisan_classifier.models.baseline import MajorityClassBaseline, TfidfLogisticBaseline
from partisan_classifier.models.transformer_model import TransformerClassifier
from partisan_classifier.utils.logging import get_logger, set_seed

logger = get_logger(__name__)

_MODEL_REGISTRY = {
    "majority": lambda cfg: MajorityClassBaseline(),
    "tfidf_logreg": lambda cfg: TfidfLogisticBaseline(),
    "transformer": lambda cfg: TransformerClassifier(
        model_name=cfg.get("model.name", "roberta-base"),
        epochs=cfg.get("model.epochs", 3),
    ),
}


def train(cfg: Config) -> None:
    """
    Run the full train pipeline: load data, split, featurize, fit, save.
    """
    set_seed(cfg.get("seed", 42))

    logger.info("Loading raw data...")
    df = load_article_bias_prediction(cfg.get("data.raw_dir"))
    print(df["label"].value_counts())
    print(df["text"].str.len().describe())

    train_df, val_df, _ = publisher_disjoint_split(
        df,
        train_frac=cfg.get("split.train_frac", 0.7),
        val_frac=cfg.get("split.val_frac", 0.15),
        test_frac=cfg.get("split.test_frac", 0.15),
        seed=cfg.get("seed", 42),
    )

    model_type = cfg.get("model.type", "tfidf_logreg")
    model = _MODEL_REGISTRY[model_type](cfg)

    logger.info("Training model...")
    # if model_type == "tfidf_logreg":
    #     featurizer = TfidfFeaturizer()
    #     X_train = featurizer.fit_transform(train_df["text"].tolist())
    #     X_val = featurizer.transform(val_df["text"].tolist())
    #     model.fit(X_train, train_df["label"].tolist())
    #     preds = model.predict(X_val)
    # else:
    #     model.fit(train_df["text"].tolist(), train_df["label"].tolist())
    #     preds = model.predict(val_df["text"].tolist())
    model.fit(train_df["text"].tolist(), train_df["label"].tolist())
    preds = model.predict(val_df["text"].tolist())


    logger.info("Evaluating model...")
    metrics = compute_metrics(val_df["label"].tolist(), preds)
    logger.info("Validation metrics: %s", metrics)

    logger.info("Saving...")
    checkpoint_dir = cfg.get("output.checkpoint_dir", "outputs/checkpoints")
    model.save(checkpoint_dir)

    logger.info("Training done!")