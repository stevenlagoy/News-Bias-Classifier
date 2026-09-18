"""
Baseline model(s) to sanity-check the pipeline and set a floor for later
approaches to beat.

A majority-class baseline plus a simple TF-IDF + linear classifier are
reasonable starting points before anything more elaborate.
"""

from __future__ import annotations

from typing import Any

from partisan_classifier.models.base import PartisanClassifier

class MajorityClassBaseline(PartisanClassifier):
    """Always predicts the most frequent label seen during fit."""

    def __init__(self) -> None:
        self.majority_label: str | None = None

    def fit(self, X: Any, y: Any) -> "MajorityClassBaseline":
        # TODO: compute the most frequent label in y.
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        if self.majority_label is None:
            raise RuntimeError("Call fit() before predict().")
        return [self.majority_label] * len(X)
