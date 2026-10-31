"""
Baseline model(s) to sanity-check the pipeline and set a floor for later approaches to beat.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

from partisan_classifier.models.base import PartisanClassifier
from partisan_classifier.features.text_features import TfidfFeaturizer


class MajorityClassBaseline(PartisanClassifier):
    """Always predicts the most frequent label seen during fit."""

    def __init__(self) -> None:
        self.majority_label: str | None = None

    def fit(self, X: Any, y: Any) -> "MajorityClassBaseline":
        self.majority_label = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, X: Any) -> Any:
        if self.majority_label is None:
            raise RuntimeError("Call fit() before predict().")
        return [self.majority_label] * len(X)

    def predict_proba(self, X: Any) -> Any:
        raise NotImplementedError("Majority baseline has no meaningful probabilities.")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "majority_label.txt").write_text(self.majority_label or "")

    @classmethod
    def load(cls, path: str) -> "MajorityClassBaseline":
        obj = cls()
        obj.majority_label = (Path(path) / "majority_label.txt").read_text().strip()
        return obj
    

class TfidfLogisticBaseline(PartisanClassifier):
    """
    TF-IDF features + multinomial logistic regression. Owns its own
    featurizer so the saved model is self-contained: predict() accepts
    raw text directly, no external vectorizer needs to be passed around.
    """

    def __init__(self) -> None:
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.featurizer: TfidfFeaturizer | None = None

    def fit(self, X: list[str], y: Any) -> "TfidfLogisticBaseline":
        self.featurizer = TfidfFeaturizer()
        X_vec = self.featurizer.fit_transform(X)
        self.clf.fit(X_vec, y)
        return self

    def predict(self, X: Any) -> Any:
        if self.featurizer is not None and isinstance(X, list):
            X = self.featurizer.transform(X)
        return self.clf.predict(X)

    def predict_proba(self, X: Any) -> Any:
        if self.featurizer is not None and isinstance(X, list):
            X = self.featurizer.transform(X)
        return self.clf.predict_proba(X)

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "featurizer": self.featurizer}, path / "model.joblib")

    @classmethod
    def load(cls, path: str) -> "TfidfLogisticBaseline":
        import joblib
        obj = cls()
        bundle = joblib.load(Path(path) / "model.joblib")
        obj.clf = bundle["clf"]
        obj.featurizer = bundle["featurizer"]
        return obj