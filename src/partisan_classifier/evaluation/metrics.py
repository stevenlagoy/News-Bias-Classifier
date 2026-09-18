"""
Metrics and reporting for the partisanship classifier.

At minimum: accuracy, macro-F1, and a confusion matrix over
{left, center, right}. Report separately on the in-distribution test split
and the Qbias out-of-distribution set.
"""

from __future__ import annotations

from typing import Any

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """
    Return a dict of accuracy, macro-F1, and per-class precision/recall.

    TODO: implement with sklearn.metrics (accuracy_score, f1_score,
    classification_report, confusion_matrix).
    """
    raise NotImplementedError
