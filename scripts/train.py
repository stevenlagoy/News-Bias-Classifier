#!/usr/bin/env python
"""
CLI wrapper: run training from a config file.

Usage:
    python scripts/train.py --config configs/default.yaml
"""

import argparse

from partisan_classifier.config import Config
from partisan_classifier.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    train(cfg)
