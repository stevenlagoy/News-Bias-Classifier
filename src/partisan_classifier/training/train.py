"""
Training entry point, driven by a config file.

Usage (once implemented):
    python scripts/train.py --config configs/default.yaml
"""

from __future__ import annotations

from partisan_classifier.config import Config

def train(cfg: Config) -> None:
    """
    Run the full train pipeline: load data, split, featurize, fit, save.

    TODO:
        1. Load raw dataset(s) via partisan_classifier.data.loader
        2. Publisher-disjoint split via partisan_classifier.data.splits
        3. Featurize via partisan_classifier.features
        4. Fit a model via partisan_classifier.models
        5. Evaluate on val split via partisan_classifier.evaluation
        6. Save checkpoint + metrics to cfg.get("output.checkpoint_dir") /
           cfg.get("output.log_dir")
    """
    raise NotImplementedError
