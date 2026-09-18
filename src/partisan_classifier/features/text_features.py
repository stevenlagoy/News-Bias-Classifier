"""
Turn raw article text into model input features.

Kept separate from models/ so the same featurization can be swapped between
a classical pipeline (e.g. TF-IDF) and a transformer tokenizer without
touching model code.
"""

from __future__ import annotations

from typing import Any

class Featurizer:
    """Base interface all featurizers should implement."""

    def fit(self, texts: list[str]) -> "Featurizer":
        raise NotImplementedError

    def transform(self, texts: list[str]) -> Any:
        raise NotImplementedError

    def fit_transform(self, texts: list[str]) -> Any:
        return self.fit(texts).transform(texts)


class TfidfFeaturizer(Featurizer):
    """TODO: wrap sklearn.feature_extraction.text.TfidfVectorizer."""

    def fit(self, texts: list[str]) -> "Featurizer":
        raise NotImplementedError

    def transform(self, texts: list[str]) -> Any:
        raise NotImplementedError
