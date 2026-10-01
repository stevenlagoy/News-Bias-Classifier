"""
Common interface for all models, so training/evaluation code doesn't need
to know whether it's driving a classical sklearn model or a fine-tuned
transformer.
"""

from __future__ import annotations

from typing import Any

class PartisanClassifier:
    """Base interface: predicts left / center / right given article features."""

    def fit(self, X: Any, y: Any) -> "PartisanClassifier":
        raise NotImplementedError

    def predict(self, X: Any) -> Any:
        raise NotImplementedError

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "PartisanClassifier":
        raise NotImplementedError
