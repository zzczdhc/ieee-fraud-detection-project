from __future__ import annotations

from pathlib import Path
import json
import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from .metrics import compute_top_fraction_metrics

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
SCORE_COL = "fraud_score"

DEFAULT_REVIEW_FRACTIONS = (0.01, 0.03, 0.05, 0.10)
DEFAULT_SEGMENT_COLS = (
    "has_identity",
    "addr_missing",
    "email_missing",
    "device_missing",
    "TransactionAmt_bucket",
    "TransactionDay_bucket",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain_top",
    "id_31_browser_top",
)


def build_scored_validation_frame(
    valid_frame: pd.DataFrame,
    y_score: Sequence[float],
    *,
    score_col: str = SCORE_COL,
    target_col: str = TARGET_COL,
    review_fractions: Sequence[float] = DEFAULT_REVIEW_FRACTIONS,
) -> pd.DataFrame:
    """Attach model scores, ranking columns, and top-review flags to validation rows."""
    if target_col not in valid_frame.columns:
        raise KeyError(f"Expected target column `{target_col}` in validation frame.")
    if len(valid_frame) != len(y_score):
        raise ValueError("Validation frame and score vector must have the same length.")

    scored = valid_frame.copy()
    scored[score_col] = np.asarray(y_score, dtype=float)
    scored["score_rank"] = scored[score_col].rank(method="first", ascending=False).astype(int)
    scored["score_percentile"] = scored["score_rank"] / len(scored)

    for fraction in review_fractions:
        label = _fraction_label(fraction)
        selected = _top_fraction_mask(scored[score_col], fraction)
        scored[f"selected_top_{label}"] = selected.astype("int8")
        scored[f"error_type_top_{label}"] = np.select(
            [
                (selected) & (scored[target_col].to_numpy() == 1),
                (selected) & (scored[target_col].to_numpy() == 0),
                (~selected) & (scored[target_col].to_numpy() == 1),
            ],
            ["true_positive", "false_positive", "false_negative"],
            default="true_negative",
        )

    return add_error_segments(scored)


def add_error_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable segment columns for false-positive/false-negative analysis."""
    out = frame.copy()

    identity_cols = [col for col in out.columns if col.startswith("id_")]
    if identity_cols:
        out["has_identity"] = out[identity_cols].notna().any(axis=1).map({True: "has_identity", False: "no_identity"})
    else:
        out["has_identity"] = "identity_unavailable"

    addr_cols = [col for col in ("addr1", "addr2") if col in out.columns]
    out["addr_missing"] = _missing_group_label(out, addr_cols, present_label="addr_present", missing_label="addr_missing")

    email_cols = [col for col in ("P_emaildomain", "R_emaildomain") if col in out.columns]
    out["email_missing"] = _missing_group_label(
        out,
        email_cols,
        present_label="email_present",
        missing_label="email_missing",
    )

    device_cols = [col for col in ("DeviceType", "DeviceInfo", "id_30", "id_31") if col in out.columns]
    out["device_missing"] = _missing_group_label(
        out,
        device_cols,
        present_label="device_present",
        missing_label="device_missing",
    )

    out["row_missing_fraction"] = out.isna().mean(axis=1)
    out["row_missing_bucket"] = _safe_qcut(out["row_missing_fraction"], q=5, prefix="missing")

    if "TransactionAmt" in out.columns:
        out["TransactionAmt_bucket"] = _safe_qcut(
            pd.to_numeric(out["TransactionAmt"], errors="coerce"),
            q=5,
            prefix="amt",
        )
    else:
        out["TransactionAmt_bucket"] = "amt_unavailable"

    if "TransactionDay" in out.columns:
        out["TransactionDay_bucket"] = _safe_qcut(
            pd.to_numeric(out["TransactionDay"], errors="coerce"),
            q=5,
            prefix="day",
        )
    elif "TransactionDT" in out.columns:
        transaction_day = pd.to_numeric(out["TransactionDT"], errors="coerce") // (24 * 3600)
        out["TransactionDay_bucket"] = _safe_qcut(transaction_day, q=5, prefix="day")
    else:
        out["TransactionDay_bucket"] = "day_unavailable"

    out["P_emaildomain_top"] = _top_category_or_other(out, "P_emaildomain", top_n=12)
    out["id_31_browser_top"] = _top_category_or_other(out, "id_31", top_n=12)

    return out


def summarize_review_budgets(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    fractions: Sequence[float] = DEFAULT_REVIEW_FRACTIONS,
) -> pd.DataFrame:
    """Summarize confusion counts and ranking metrics at fixed review budgets."""
    y_true_array = np.asarray(y_true, dtype=int)
    y_score_array = np.asarray(y_score, dtype=float)
    rows = []

    for fraction in fractions:
        selected = _top_fraction_mask(y_score_array, fraction)
        tp = int(np.sum(selected & (y_true_array == 1)))
        fp = int(np.sum(selected & (y_true_array == 0)))
        fn = int(np.sum((~selected) & (y_true_array == 1)))
        tn = int(np.sum((~selected) & (y_true_array == 0)))
        metrics = compute_top_fraction_metrics(y_true_array, y_score_array, fraction=fraction)

        rows.append(
            {
                "review_fraction": float(fraction),
                "review_count": int(np.sum(selected)),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn,
                "precision": _safe_divide(tp, tp + fp),
                "recall": _safe_divide(tp, tp + fn),
                "false_positive_rate": _safe_divide(fp, fp + tn),
                "false_negative_rate": _safe_divide(fn, tp + fn),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize_score_bins(
    scored_frame: pd.DataFrame,
    *,
    score_col: str = SCORE_COL,
    target_col: str = TARGET_COL,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare observed fraud rate across model score buckets."""
    work = scored_frame[[score_col, target_col]].copy()
    work["score_bucket"] = _safe_qcut(work[score_col], q=n_bins, prefix="score")

    summary = (
        work.groupby("score_bucket", dropna=False)
        .agg(
            rows=(target_col, "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
            min_score=(score_col, "min"),
            median_score=(score_col, "median"),
            max_score=(score_col, "max"),
        )
        .reset_index()
    )
    summary["bucket_order"] = summary["score_bucket"].map(_bucket_order)
    return summary.sort_values("bucket_order", ascending=False).drop(columns="bucket_order").reset_index(drop=True)


def summarize_segment_errors(
    scored_frame: pd.DataFrame,
    *,
    segment_cols: Iterable[str] = DEFAULT_SEGMENT_COLS,
    review_fraction: float = 0.03,
    target_col: str = TARGET_COL,
    min_count: int = 50,
) -> pd.DataFrame:
    """Build segment-level false-positive/false-negative diagnostics for a review budget."""
    label = _fraction_label(review_fraction)
    selected_col = f"selected_top_{label}"
    if selected_col not in scored_frame.columns:
        raise KeyError(f"Missing `{selected_col}`. Build the scored frame with this review fraction first.")

    rows = []
    for segment_col in segment_cols:
        if segment_col not in scored_frame.columns:
            continue

        grouped = scored_frame.groupby(segment_col, dropna=False)
        for segment_value, group in grouped:
            if len(group) < min_count:
                continue

            y_true = group[target_col].astype(int).to_numpy()
            selected = group[selected_col].astype(bool).to_numpy()
            tp = int(np.sum(selected & (y_true == 1)))
            fp = int(np.sum(selected & (y_true == 0)))
            fn = int(np.sum((~selected) & (y_true == 1)))

            rows.append(
                {
                    "segment_column": segment_col,
                    "segment_value": _clean_segment_value(segment_value),
                    "rows": int(len(group)),
                    "row_share": float(len(group) / len(scored_frame)),
                    "fraud_count": int(np.sum(y_true)),
                    "fraud_rate": float(np.mean(y_true)),
                    "selected_count": int(np.sum(selected)),
                    "selected_rate": float(np.mean(selected)),
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "precision_within_segment": _safe_divide(tp, tp + fp),
                    "recall_within_segment": _safe_divide(tp, tp + fn),
                    "false_positive_share": _safe_divide(fp, max(1, int(np.sum(scored_frame[selected_col] & (scored_frame[target_col] == 0))))),
                    "false_negative_share": _safe_divide(fn, max(1, int(np.sum((scored_frame[selected_col] == 0) & (scored_frame[target_col] == 1))))),
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    return result.sort_values(
        ["false_negative_share", "false_positive_share", "rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def extract_error_examples(
    scored_frame: pd.DataFrame,
    *,
    review_fraction: float = 0.03,
    score_col: str = SCORE_COL,
    target_col: str = TARGET_COL,
    max_examples: int = 50,
) -> dict[str, pd.DataFrame]:
    """Return the highest-score false positives and highest-score false negatives outside review."""
    label = _fraction_label(review_fraction)
    error_col = f"error_type_top_{label}"
    if error_col not in scored_frame.columns:
        raise KeyError(f"Missing `{error_col}`. Build the scored frame with this review fraction first.")

    display_cols = [
        col
        for col in [
            ID_COL,
            target_col,
            score_col,
            "score_rank",
            "score_percentile",
            error_col,
            "TransactionAmt",
            "ProductCD",
            "card4",
            "card6",
            "P_emaildomain",
            "R_emaildomain",
            "DeviceType",
            "id_31",
            "has_identity",
            "addr_missing",
            "email_missing",
            "device_missing",
            "TransactionAmt_bucket",
            "TransactionDay_bucket",
        ]
        if col in scored_frame.columns
    ]

    false_positives = (
        scored_frame.loc[scored_frame[error_col] == "false_positive", display_cols]
        .sort_values(score_col, ascending=False)
        .head(max_examples)
        .reset_index(drop=True)
    )
    false_negatives = (
        scored_frame.loc[scored_frame[error_col] == "false_negative", display_cols]
        .sort_values(score_col, ascending=False)
        .head(max_examples)
        .reset_index(drop=True)
    )
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def run_error_analysis(
    valid_frame: pd.DataFrame,
    y_score: Sequence[float],
    *,
    output_dir: Path | str,
    model_name: str,
    review_fractions: Sequence[float] = DEFAULT_REVIEW_FRACTIONS,
    primary_review_fraction: float = 0.03,
    min_segment_count: int = 50,
    max_examples: int = 50,
) -> dict[str, pd.DataFrame]:
    """Generate and save error-analysis artifacts for a model's validation scores."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scored = build_scored_validation_frame(
        valid_frame,
        y_score,
        review_fractions=review_fractions,
    )
    review_summary = summarize_review_budgets(
        scored[TARGET_COL].to_numpy(),
        scored[SCORE_COL].to_numpy(),
        fractions=review_fractions,
    )
    score_bins = summarize_score_bins(scored)
    segment_summary = summarize_segment_errors(
        scored,
        review_fraction=primary_review_fraction,
        min_count=min_segment_count,
    )
    examples = extract_error_examples(
        scored,
        review_fraction=primary_review_fraction,
        max_examples=max_examples,
    )

    scored.to_csv(output_path / "validation_scores_with_error_flags.csv", index=False)
    review_summary.to_csv(output_path / "review_budget_summary.csv", index=False)
    score_bins.to_csv(output_path / "score_bucket_summary.csv", index=False)
    segment_summary.to_csv(output_path / "segment_error_summary.csv", index=False)
    examples["false_positives"].to_csv(output_path / "false_positive_examples.csv", index=False)
    examples["false_negatives"].to_csv(output_path / "false_negative_examples.csv", index=False)

    metadata = {
        "model_name": model_name,
        "validation_rows": int(len(scored)),
        "validation_fraud_rate": float(scored[TARGET_COL].mean()),
        "review_fractions": [float(fraction) for fraction in review_fractions],
        "primary_review_fraction": float(primary_review_fraction),
        "min_segment_count": int(min_segment_count),
        "max_examples": int(max_examples),
    }
    (output_path / "error_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return {
        "scored": scored,
        "review_summary": review_summary,
        "score_bins": score_bins,
        "segment_summary": segment_summary,
        **examples,
    }


def _missing_group_label(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    present_label: str,
    missing_label: str,
) -> pd.Series:
    if not cols:
        return pd.Series(f"{missing_label}_unavailable", index=frame.index)
    return frame[cols].isna().any(axis=1).map({True: missing_label, False: present_label})


def _top_category_or_other(frame: pd.DataFrame, col: str, *, top_n: int) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(f"{col}_unavailable", index=frame.index)

    values = frame[col].fillna("Missing").astype(str)
    top_values = set(values.value_counts(dropna=False).head(top_n).index)
    return values.where(values.isin(top_values), other="Other")


def _top_fraction_mask(scores: Sequence[float], fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")
    score_array = np.asarray(scores, dtype=float)
    n_select = max(1, int(math.ceil(len(score_array) * fraction)))
    order = np.argsort(-score_array)
    selected = np.zeros(len(score_array), dtype=bool)
    selected[order[:n_select]] = True
    return selected


def _safe_qcut(series: pd.Series, *, q: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    result = pd.Series(f"{prefix}_missing", index=series.index, dtype="object")
    if valid.sum() < q:
        result.loc[valid] = f"{prefix}_available"
        return result

    try:
        codes = pd.qcut(numeric.loc[valid], q=q, labels=False, duplicates="drop")
    except ValueError:
        result.loc[valid] = f"{prefix}_available"
        return result

    result.loc[valid] = codes.astype(int).map(lambda value: f"{prefix}_q{value + 1}")
    return result


def _fraction_label(fraction: float) -> str:
    pct = fraction * 100
    if pct.is_integer():
        return f"{int(pct)}pct"
    return f"{pct:.1f}".replace(".", "p") + "pct"


def _bucket_order(value: object) -> int:
    if not isinstance(value, str) or "_q" not in value:
        return -1
    try:
        return int(value.rsplit("_q", maxsplit=1)[1])
    except ValueError:
        return -1


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _clean_segment_value(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    return str(value)
