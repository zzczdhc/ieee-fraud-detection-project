from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    top_fractions: Sequence[float] = (0.01, 0.05),
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    base_rate = float(np.mean(y_true))

    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_score),
        "average_precision": _safe_metric(average_precision_score, y_true, y_score),
        "log_loss": _safe_metric(log_loss, y_true, y_score),
        "brier_score": _safe_metric(brier_score_loss, y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "base_fraud_rate": base_rate,
    }

    for fraction in top_fractions:
        metrics.update(compute_top_fraction_metrics(y_true, y_score, fraction=fraction))

    return {name: float(value) for name, value in metrics.items()}


def compute_top_fraction_metrics(
    y_true,
    y_score,
    fraction: float,
) -> dict[str, float]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    label = _fraction_label(fraction)
    n_select = max(1, int(math.ceil(len(y_score) * fraction)))
    order = np.argsort(-y_score)
    selected = order[:n_select]
    selected_y = y_true[selected]

    base_rate = float(np.mean(y_true))
    precision_at_fraction = float(np.mean(selected_y)) if len(selected_y) else math.nan
    positives = float(np.sum(y_true))
    recall_at_fraction = float(np.sum(selected_y) / positives) if positives else math.nan
    lift_at_fraction = float(precision_at_fraction / base_rate) if base_rate else math.nan

    return {
        f"precision_at_top_{label}": precision_at_fraction,
        f"recall_at_top_{label}": recall_at_fraction,
        f"lift_at_top_{label}": lift_at_fraction,
    }


def build_curve_frames(y_true, y_score) -> dict[str, pd.DataFrame]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)

    roc_frame = pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": roc_thresholds,
        }
    )
    pr_frame = pd.DataFrame(
        {
            "recall": recall,
            "precision": precision,
            "threshold": np.append(pr_thresholds, np.nan),
        }
    )
    return {"roc": roc_frame, "pr": pr_frame}


def build_threshold_metrics_table(
    y_true,
    y_score,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = tuple(thresholds or np.linspace(0.05, 0.95, 19))

    rows = []
    for threshold in thresholds:
        metrics = compute_classification_metrics(
            y_true,
            y_score,
            threshold=threshold,
            top_fractions=(),
        )
        rows.append({"threshold": float(threshold), **metrics})

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _safe_metric(metric_fn, y_true, y_pred_or_score) -> float:
    try:
        return float(metric_fn(y_true, y_pred_or_score))
    except ValueError:
        return math.nan


def _fraction_label(fraction: float) -> str:
    pct = fraction * 100
    if pct.is_integer():
        return f"{int(pct)}pct"
    return f"{pct:.1f}".replace(".", "p") + "pct"
