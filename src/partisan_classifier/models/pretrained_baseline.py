"""
Zero-shot baseline using bucketresearch/politicalBiasBERT, used only for
comparison against your own trained models. Not trained on your data.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import pipeline
import pandas as pd

from partisan_classifier.models.base import PartisanClassifier

_LABEL_MAP = {"LEFT": "left", "CENTER": "center", "RIGHT": "right"}


class PoliticalBiasBertBaseline(PartisanClassifier):
    def __init__(self, max_chars: int = 2000, batch_size: int = 32) -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model="bucketresearch/politicalBiasBERT",
            truncation=True,
            device=device,
            batch_size=batch_size,
        )
        self.max_chars = max_chars

    def fit(self, X: Any, y: Any) -> "PoliticalBiasBertBaseline":
        return self  # pretrained; nothing to fit

    def predict(self, X: list[str]) -> list[str]:
        texts = [str(t)[: self.max_chars] if pd.notna(t) else "" for t in X]
        preds = self.pipe(
            texts,
            truncation=True,
            max_length=512,
            padding=True
        )
        return [_LABEL_MAP[p["label"]] for p in preds]

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("Not needed for the baseline comparison.")

    def save(self, path: str) -> None:
        pass  # nothing to persist; model is loaded from HF Hub each time

    @classmethod
    def load(cls, path: str) -> "PoliticalBiasBertBaseline":
        return cls()