"""
Turn raw article text into model input features.

Kept separate from models/ so the same featurization can be swapped between
a classical pipeline (e.g. TF-IDF) and a transformer tokenizer without
touching model code.
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


class Featurizer:
    """Base interface all featurizers should implement."""

    def fit(self, texts: list[str]) -> "Featurizer":
        raise NotImplementedError

    def transform(self, texts: list[str]) -> Any:
        raise NotImplementedError

    def fit_transform(self, texts: list[str]) -> Any:
        return self.fit(texts).transform(texts)


class TfidfFeaturizer(Featurizer):
    """Wraps sklearn's TfidfVectorizer for the classical baseline."""

    def __init__(self, max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, stop_words="english"
        )

    def fit(self, texts: list[str]) -> "Featurizer":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]) -> Any:
        return self.vectorizer.transform(texts)

class TransformerFeaturizer(Featurizer):
    """Wraps a HF tokenizer so the transformer path uses the same interface."""

    def __init__(self, model_name: str = "roberta-base", max_length: int = 512):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def fit(self, texts: list[str]) -> "TransformerFeaturizer":
        return self # Pretrained, nothing to fit

    def transform(self, texts: list[str]) -> Any:
        return self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
    