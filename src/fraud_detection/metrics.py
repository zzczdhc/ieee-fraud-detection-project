from __future__ import annotations

import math

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": _safe_metric(average_precision_score, y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }

    return {name: float(value) for name, value in metrics.items()}


def _safe_metric(metric_fn, y_true, y_pred_or_score) -> float:
    try:
        return float(metric_fn(y_true, y_pred_or_score))
    except ValueError:
        return math.nan
