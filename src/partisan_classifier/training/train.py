"""
Training entry point, driven by a config file.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

from partisan_classifier.config import Config
from partisan_classifier.data.loader import (
    load_article_bias_prediction,
    load_media_split,
)
from partisan_classifier.evaluation.metrics import compute_metrics
from partisan_classifier.models.baseline import (
    MajorityClassBaseline,
    TfidfLogisticBaseline,
)
from partisan_classifier.utils.logging import get_logger, set_seed

logger = get_logger(__name__)


def _build_transformer(cfg: Config):
    # Import here so tfidf_logreg/majority runs don't require
    # torch/transformers/datasets to be installed.
    from partisan_classifier.models.transformer_model import TransformerClassifier

    return TransformerClassifier(
        model_name=cfg.get("model.name", "roberta-base"),
        output_dir="outputs/transformer",
        epochs=cfg.get("model.epochs", 3),
        batch_size=cfg.get("training.batch_size", 8),
        learning_rate=cfg.get("training.learning_rate", 2.0e-5),
        seed=cfg.get("seed", 42),
    )


_MODEL_REGISTRY = {
    "majority": lambda cfg: MajorityClassBaseline(),
    "tfidf_logreg": lambda cfg: TfidfLogisticBaseline(),
    "transformer": _build_transformer,
}


def train(cfg: Config) -> None:
    """
    Run the full train pipeline: load data, split, fit, evaluate, and save.
    """
    set_seed(cfg.get("seed", 42))

    logger.info("Loading raw data...")

    df = load_article_bias_prediction(
        cfg.get("data.raw_dir")
    )

    print(df["label"].value_counts())
    print(df["text"].str.len().describe())

    # Use the official Baly media-based split.
    # This is the primary publisher-disjoint evaluation setup.
    train_df, val_df, test_df = load_media_split(
        cfg.get("data.raw_dir")
    )

    model_type = cfg.get(
        "model.type",
        "tfidf_logreg",
    )

    model = _MODEL_REGISTRY[model_type](cfg)

    logger.info("Training model...")

    if model_type == "transformer":
        # The transformer uses the validation set during training
        # to calculate macro-F1 after every epoch and select the
        # best checkpoint.
        #
        # The test set is NOT used for training or model selection.
        model.fit(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
            val_df["text"].tolist(),
            val_df["label"].tolist(),
        )
    else:
        # Baseline models keep their original two-argument fit interface.
        model.fit(
            train_df["text"].tolist(),
            train_df["label"].tolist(),
        )

    logger.info("Evaluating model...")

    preds = model.predict(
        val_df["text"].tolist()
    )

    metrics = compute_metrics(
        val_df["label"].tolist(),
        preds,
    )

    logger.info(
        "Validation metrics: %s",
        metrics,
    )

    logger.info("Saving...")

    checkpoint_dir = cfg.get(
        "output.checkpoint_dir",
        "outputs/checkpoints",
    )

    model.save(checkpoint_dir)

    logger.info(
        "Training done! Model saved to %s",
        checkpoint_dir,
    )