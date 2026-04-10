from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PLOT_COLORS = {
    "navy": "#16324F",
    "blue": "#2D6A8D",
    "teal": "#2A9D8F",
    "gold": "#C8A951",
    "coral": "#C95A49",
    "sand": "#F4EBD0",
    "ink": "#102A43",
    "muted": "#6B7280",
}


def set_plot_theme() -> None:
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=[PLOT_COLORS["navy"], PLOT_COLORS["blue"], PLOT_COLORS["teal"], PLOT_COLORS["coral"]],
    )
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D1D5DB",
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelcolor": PLOT_COLORS["ink"],
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "grid.color": "#E5E7EB",
            "grid.alpha": 0.9,
            "grid.linewidth": 0.8,
            "xtick.color": PLOT_COLORS["ink"],
            "ytick.color": PLOT_COLORS["ink"],
            "text.color": PLOT_COLORS["ink"],
            "legend.frameon": False,
            "figure.autolayout": False,
            "font.size": 11,
        }
    )


def reduce_memory_usage(frame: pd.DataFrame) -> pd.DataFrame:
    optimized = frame.copy()
    for column in optimized.columns:
        series = optimized[column]
        if pd.api.types.is_integer_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            optimized[column] = pd.to_numeric(series, downcast="float")
    return optimized


def standardize_identity_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.copy()
    renamed.columns = [column.replace("-", "_") for column in renamed.columns]
    return renamed


def add_transaction_time_features(
    frame: pd.DataFrame,
    dt_column: str = "TransactionDT",
) -> pd.DataFrame:
    enriched = frame.copy()
    if dt_column not in enriched.columns:
        return enriched

    enriched["TransactionDay"] = (enriched[dt_column] // (24 * 60 * 60)).astype("Int64")
    enriched["TransactionWeek"] = (enriched["TransactionDay"] // 7).astype("Int64")
    enriched["TransactionHour"] = ((enriched[dt_column] // 3600) % 24).astype("Int64")
    return enriched


def infer_feature_family(column: str) -> str:
    if column in {"TransactionID", "isFraud", "TransactionDT", "TransactionAmt"}:
        return "core"
    if column.startswith("card"):
        return "card"
    if column.startswith("addr"):
        return "address"
    if column.startswith("dist"):
        return "distance"
    if column.startswith("P_") or column.startswith("R_"):
        return "email"
    if column.startswith("C"):
        return "count_c"
    if column.startswith("D"):
        return "delta_d"
    if column.startswith("M"):
        return "match_m"
    if column.startswith("V"):
        return "anonymous_v"
    if column.startswith("id_"):
        return "identity"
    if column in {"DeviceType", "DeviceInfo"}:
        return "device"
    return "other"


def metric_cards_html(cards: Sequence[dict[str, str]]) -> str:
    card_blocks = []
    for card in cards:
        card_blocks.append(
            f"""
            <div style="flex:1; min-width:180px; background:{PLOT_COLORS['sand']};
                        border:1px solid #E1D8C7; border-radius:18px; padding:18px 20px;">
                <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:{PLOT_COLORS['muted']};">
                    {card['title']}
                </div>
                <div style="font-size:28px; font-weight:700; color:{PLOT_COLORS['ink']}; margin-top:6px;">
                    {card['value']}
                </div>
                <div style="font-size:13px; color:{PLOT_COLORS['muted']}; margin-top:6px;">
                    {card['subtitle']}
                </div>
            </div>
            """
        )

    return """
    <div style="display:flex; flex-wrap:wrap; gap:14px; margin: 12px 0 8px 0;">
        {blocks}
    </div>
    """.format(blocks="".join(card_blocks))


def build_inventory_table(
    train_transaction: pd.DataFrame,
    train_identity: pd.DataFrame,
    test_transaction: pd.DataFrame,
    test_identity: pd.DataFrame,
) -> pd.DataFrame:
    frames = {
        "train_transaction": train_transaction,
        "train_identity": train_identity,
        "test_transaction": test_transaction,
        "test_identity": test_identity,
    }

    rows = []
    for name, frame in frames.items():
        rows.append(
            {
                "table": name,
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "memory_mb": frame.memory_usage(deep=True).sum() / 1024**2,
            }
        )
    return pd.DataFrame(rows)


def compute_family_missingness(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    transaction_missing = (
        transaction_df.drop(columns=["TransactionID", "isFraud"], errors="ignore").isna().mean()
    )

    records = [
        {
            "column": column,
            "missing_rate": float(rate),
            "family": infer_feature_family(column),
            "source": "transaction",
        }
        for column, rate in transaction_missing.items()
    ]

    if identity_df is not None and not identity_df.empty:
        standardized_identity = standardize_identity_columns(identity_df)
        identity_core = standardized_identity.drop(columns=["TransactionID"], errors="ignore")
        coverage = len(standardized_identity) / len(transaction_df)
        identity_missing = 1 - coverage * (1 - identity_core.isna().mean())
        records.extend(
            {
                "column": column,
                "missing_rate": float(rate),
                "family": infer_feature_family(column),
                "source": "identity",
            }
            for column, rate in identity_missing.items()
        )

    return pd.DataFrame(records).sort_values("missing_rate", ascending=False).reset_index(drop=True)


def missingness_signal_table(
    frame: pd.DataFrame,
    target_column: str = "isFraud",
    min_support: int = 5000,
) -> pd.DataFrame:
    if target_column not in frame.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    rows = []
    for column in frame.columns:
        if column in {"TransactionID", target_column}:
            continue
        missing_mask = frame[column].isna()
        missing_count = int(missing_mask.sum())
        present_count = int((~missing_mask).sum())
        if missing_count < min_support or present_count < min_support:
            continue

        fraud_rate_missing = float(frame.loc[missing_mask, target_column].mean())
        fraud_rate_present = float(frame.loc[~missing_mask, target_column].mean())
        rows.append(
            {
                "column": column,
                "family": infer_feature_family(column),
                "missing_count": missing_count,
                "present_count": present_count,
                "fraud_rate_missing": fraud_rate_missing,
                "fraud_rate_present": fraud_rate_present,
                "lift_vs_present": fraud_rate_missing - fraud_rate_present,
            }
        )

    return pd.DataFrame(rows).sort_values("lift_vs_present", ascending=False).reset_index(drop=True)


def category_risk_table(
    frame: pd.DataFrame,
    column: str,
    target_column: str = "isFraud",
    min_count: int = 1000,
    top_n: int = 10,
) -> pd.DataFrame:
    working = frame[[column, target_column]].copy()
    working[column] = working[column].fillna("Missing").astype(str)

    summary = (
        working.groupby(column, observed=False)[target_column]
        .agg(fraud_rate="mean", count="size")
        .reset_index()
    )
    summary = summary.loc[summary["count"] >= min_count].copy()
    if summary.empty:
        return summary

    base_rate = float(frame[target_column].mean())
    summary["share"] = summary["count"] / len(frame)
    summary["lift_vs_base"] = summary["fraud_rate"] - base_rate
    summary = summary.sort_values(["fraud_rate", "count"], ascending=[False, False]).head(top_n)
    return summary.reset_index(drop=True)


def quantile_fraud_table(
    frame: pd.DataFrame,
    column: str,
    target_column: str = "isFraud",
    quantiles: int = 10,
    clip_upper_quantile: float | None = None,
) -> pd.DataFrame:
    working = frame[[column, target_column]].dropna().copy()
    if working.empty:
        return pd.DataFrame(columns=["bucket", "fraud_rate", "count", "median_value"])

    if clip_upper_quantile is not None:
        upper = working[column].quantile(clip_upper_quantile)
        working[column] = working[column].clip(upper=upper)

    working["bucket"] = pd.qcut(working[column], q=quantiles, duplicates="drop")
    summary = (
        working.groupby("bucket", observed=False)
        .agg(
            fraud_rate=(target_column, "mean"),
            count=(target_column, "size"),
            median_value=(column, "median"),
        )
        .reset_index()
    )
    summary["bucket"] = summary["bucket"].astype(str)
    return summary


def merge_identity_features(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    standardized_identity = standardize_identity_columns(identity_df)
    selected_columns = ["TransactionID", *[column for column in columns if column in standardized_identity.columns]]
    return transaction_df.merge(standardized_identity[selected_columns], on="TransactionID", how="left")


def _psi_numeric(train_series: pd.Series, test_series: pd.Series, bins: int = 10) -> float:
    train_values = train_series.dropna()
    test_values = test_series.dropna()
    if train_values.nunique() < 2 or test_values.nunique() < 2:
        return float("nan")

    edges = np.unique(np.quantile(train_values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")

    train_buckets = pd.cut(train_values, bins=edges, include_lowest=True)
    test_buckets = pd.cut(test_values, bins=edges, include_lowest=True)
    categories = train_buckets.cat.categories
    epsilon = 1e-6

    train_pct = train_buckets.value_counts(normalize=True, sort=False).reindex(categories, fill_value=0) + epsilon
    test_pct = test_buckets.value_counts(normalize=True, sort=False).reindex(categories, fill_value=0) + epsilon
    return float(((train_pct - test_pct) * np.log(train_pct / test_pct)).sum())


def _psi_categorical(train_series: pd.Series, test_series: pd.Series, top_n: int = 15) -> float:
    train_values = train_series.fillna("__MISSING__").astype(str)
    test_values = test_series.fillna("__MISSING__").astype(str)
    retained_categories = train_values.value_counts().head(top_n).index.tolist()
    train_values = train_values.where(train_values.isin(retained_categories), "__OTHER__")
    test_values = test_values.where(test_values.isin(retained_categories), "__OTHER__")

    categories = sorted(set(train_values.unique()) | set(test_values.unique()))
    epsilon = 1e-6
    train_pct = train_values.value_counts(normalize=True).reindex(categories, fill_value=0) + epsilon
    test_pct = test_values.value_counts(normalize=True).reindex(categories, fill_value=0) + epsilon
    return float(((train_pct - test_pct) * np.log(train_pct / test_pct)).sum())


def compute_psi_table(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    categorical = set(categorical_columns or [])
    rows = []

    for column in columns:
        if column not in train_df.columns or column not in test_df.columns:
            continue

        if column in categorical or not pd.api.types.is_numeric_dtype(train_df[column]):
            psi = _psi_categorical(train_df[column], test_df[column])
        else:
            psi = _psi_numeric(train_df[column], test_df[column])

        rows.append(
            {
                "column": column,
                "family": infer_feature_family(column),
                "psi": psi,
                "drift_level": classify_drift_level(psi),
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def classify_drift_level(psi: float) -> str:
    if pd.isna(psi):
        return "n/a"
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "moderate"
    if psi >= 0.02:
        return "mild"
    return "stable"
