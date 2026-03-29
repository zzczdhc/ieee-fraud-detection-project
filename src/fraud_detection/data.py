from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TRAIN_IDENTITY_PATH, TRAIN_TRANSACTION_PATH


def load_train_data(
    transaction_path: Path | str = TRAIN_TRANSACTION_PATH,
    identity_path: Path | str = TRAIN_IDENTITY_PATH,
    sample_size: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    transaction_path = Path(transaction_path)
    identity_path = Path(identity_path)

    if not transaction_path.exists():
        raise FileNotFoundError(
            f"Missing transaction file: {transaction_path}. "
            "Download the Kaggle data and place it under data/raw/."
        )

    transaction_df = pd.read_csv(transaction_path)

    if identity_path.exists():
        identity_df = pd.read_csv(identity_path)
        data = transaction_df.merge(identity_df, on="TransactionID", how="left")
    else:
        data = transaction_df

    if sample_size is not None and sample_size < len(data):
        data = data.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str = "isFraud",
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in data.columns:
        raise KeyError(f"Target column '{target_column}' not found.")

    protected_drop_columns = ["TransactionID"]
    if drop_columns:
        protected_drop_columns.extend(drop_columns)

    feature_frame = data.drop(columns=[target_column, *protected_drop_columns], errors="ignore")
    target = data[target_column].copy()
    return feature_frame, target


def summarize_frame(data: pd.DataFrame, target_column: str = "isFraud") -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "n_rows": int(len(data)),
        "n_columns": int(data.shape[1]),
        "missing_fraction": float(data.isna().mean().mean()),
    }
    if target_column in data.columns:
        summary["fraud_rate"] = float(data[target_column].mean())
    return summary
