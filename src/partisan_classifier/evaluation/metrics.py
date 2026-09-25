"""
Metrics and reporting for the partisanship classifier.

At minimum: accuracy, macro-F1, and a confusion matrix over
{left, center, right}. Report separately on the in-distribution test split
and the Qbias out-of-distribution set.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Return a dict of accuracy, macro-F1, and per-class precision/recall."""
    labels = ["left", "center", "right"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }