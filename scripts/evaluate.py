#!/usr/bin/env python
"""
CLI wrapper: evaluate a trained model on test / OOD splits.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/model.pkl
"""

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    # TODO:
    #   1. Load config + checkpoint
    #   2. Load test split + Qbias OOD split
    #   3. Run partisan_classifier.evaluation.metrics.compute_metrics on both
    #   4. Print / save a comparison report
    raise NotImplementedError
