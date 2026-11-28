#!/usr/bin/env python
"""
CLI wrapper: run training from a config file.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --model transformer
"""

import argparse

from partisan_classifier.config import Config
from partisan_classifier.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model",
        choices=["majority", "tfidf_logreg", "transformer"],
        help="Override the model type from the config file.",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)

    if args.model is not None:
        cfg.raw["model"]["type"] = args.model

    train(cfg)


if __name__ == "__main__":
    main()