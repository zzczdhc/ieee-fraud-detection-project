from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

RANDOM_STATE = 42
VALID_SIZE = 0.20


MISSING_DROP_THRESHOLD = 0.98
MISSING_INDICATOR_THRESHOLD = 0.10
HIGH_CARDINALITY_THRESHOLD = 30


@dataclass
class SafePreprocessingArtifacts:
    drop_cols: list[str]
    missing_indicator_cols: list[str]
    high_cardinality_cols: list[str]
    low_cardinality_cols: list[str]
    numeric_cols: list[str]
    frequency_maps: dict
    preprocessor: ColumnTransformer


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("-", "_") for c in df.columns]
    return df


def read_csv_checked(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, nrows=nrows)
    df = standardize_column_names(df)
    return df


def load_raw_tables(nrows: Optional[int] = None):
    train_transaction = read_csv_checked(RAW_DIR / "train_transaction.csv", nrows=nrows)
    train_identity = read_csv_checked(RAW_DIR / "train_identity.csv", nrows=nrows)
    test_transaction = read_csv_checked(RAW_DIR / "test_transaction.csv", nrows=nrows)
    test_identity = read_csv_checked(RAW_DIR / "test_identity.csv", nrows=nrows)
    return train_transaction, train_identity, test_transaction, test_identity


def merge_transaction_and_identity(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    if ID_COL not in transaction_df.columns:
        raise ValueError(f"{ID_COL} missing from transaction table")
    if ID_COL not in identity_df.columns:
        raise ValueError(f"{ID_COL} missing from identity table")

    merged = transaction_df.merge(
        identity_df,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )
    return merged


def add_basic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TIME_COL not in df.columns:
        return df

    seconds = df[TIME_COL].fillna(0)
    df["TransactionHour"] = ((seconds // 3600) % 24).astype("int32")
    df["TransactionDay"] = (seconds // (24 * 3600)).astype("int32")
    df["TransactionWeek"] = (seconds // (7 * 24 * 3600)).astype("int32")
    return df


def downcast_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    int_cols = df.select_dtypes(include=["int64"]).columns
    float_cols = df.select_dtypes(include=["float64"]).columns

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def validate_train_test_schema(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_feature_cols = set(train_df.columns) - {TARGET_COL}
    test_cols = set(test_df.columns)

    only_in_train = sorted(train_feature_cols - test_cols)
    only_in_test = sorted(test_cols - train_feature_cols)

    if only_in_train:
        raise ValueError(f"Columns only in train: {only_in_train[:20]}")
    if only_in_test:
        raise ValueError(f"Columns only in test: {only_in_test[:20]}")


def load_merged_data_safe(nrows: Optional[int] = None):
    train_transaction, train_identity, test_transaction, test_identity = load_raw_tables(nrows=nrows)

    train_df = merge_transaction_and_identity(train_transaction, train_identity)
    test_df = merge_transaction_and_identity(test_transaction, test_identity)

    train_df = add_basic_time_features(train_df)
    test_df = add_basic_time_features(test_df)

    train_df = downcast_numeric_types(train_df)
    test_df = downcast_numeric_types(test_df)

    validate_train_test_schema(train_df, test_df)
    return train_df, test_df


def make_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean().values * 100).round(4),
        "nunique": df.nunique(dropna=True).values,
    })
    return summary.sort_values(
        by=["missing_pct", "nunique"],
        ascending=[False, False],
    ).reset_index(drop=True)


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def choose_drop_cols(df: pd.DataFrame) -> list[str]:
    missing_fraction = df.isna().mean()
    protected = {TIME_COL}
    drop_cols = [
        col for col in df.columns
        if missing_fraction[col] >= MISSING_DROP_THRESHOLD and col not in protected
    ]
    return sorted(drop_cols)


def choose_missing_indicator_cols(df: pd.DataFrame) -> list[str]:
    missing_fraction = df.isna().mean()
    indicator_cols = [
        col for col in df.columns
        if missing_fraction[col] >= MISSING_INDICATOR_THRESHOLD
    ]
    return sorted(indicator_cols)


def add_missing_indicators(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return df

    indicator_df = pd.DataFrame(
        {
            f"{col}__missing": df[col].isna().astype("int8")
            for col in valid_cols
        },
        index=df.index,
    )

    return pd.concat([df, indicator_df], axis=1)


def fit_frequency_maps(df: pd.DataFrame, categorical_cols: list[str]) -> dict:
    frequency_maps = {}
    for col in categorical_cols:
        frequency_maps[col] = df[col].value_counts(normalize=True, dropna=False).to_dict()
    return frequency_maps


def apply_frequency_encoding(df: pd.DataFrame, frequency_maps: dict) -> pd.DataFrame:
    df = df.copy()
    for col, freq_map in frequency_maps.items():
        if col in df.columns:
            df[f"{col}__freq"] = df[col].map(freq_map).fillna(0.0).astype("float32")
    return df


def base_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[TARGET_COL, ID_COL], errors="ignore").copy()


def fit_safe_preprocessing(train_df: pd.DataFrame) -> SafePreprocessingArtifacts:
    X_train = base_feature_frame(train_df)

    drop_cols = choose_drop_cols(X_train)
    X_train = X_train.drop(columns=drop_cols, errors="ignore")

    missing_indicator_cols = choose_missing_indicator_cols(X_train)
    X_train = add_missing_indicators(X_train, missing_indicator_cols)

    categorical_cols = get_categorical_columns(X_train)

    high_cardinality_cols = [
        col for col in categorical_cols
        if X_train[col].nunique(dropna=False) >= HIGH_CARDINALITY_THRESHOLD
    ]
    low_cardinality_cols = [
        col for col in categorical_cols
        if col not in high_cardinality_cols
    ]

    frequency_maps = fit_frequency_maps(X_train, high_cardinality_cols)
    X_train = apply_frequency_encoding(X_train, frequency_maps)
    X_train = X_train.drop(columns=high_cardinality_cols, errors="ignore")

    low_cardinality_cols = [col for col in low_cardinality_cols if col in X_train.columns]
    numeric_cols = [col for col in X_train.columns if col not in low_cardinality_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, low_cardinality_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    preprocessor.fit(X_train)

    return SafePreprocessingArtifacts(
        drop_cols=drop_cols,
        missing_indicator_cols=missing_indicator_cols,
        high_cardinality_cols=high_cardinality_cols,
        low_cardinality_cols=low_cardinality_cols,
        numeric_cols=numeric_cols,
        frequency_maps=frequency_maps,
        preprocessor=preprocessor,
    )


def prepare_features_for_transform(df: pd.DataFrame, artifacts: SafePreprocessingArtifacts) -> pd.DataFrame:
    X = base_feature_frame(df)

    X = X.drop(columns=artifacts.drop_cols, errors="ignore")
    X = add_missing_indicators(X, artifacts.missing_indicator_cols)
    X = apply_frequency_encoding(X, artifacts.frequency_maps)
    X = X.drop(columns=artifacts.high_cardinality_cols, errors="ignore")

    expected_cols = artifacts.numeric_cols + artifacts.low_cardinality_cols
    for col in expected_cols:
        if col not in X.columns:
            X[col] = np.nan

    X = X[expected_cols]
    return X


def transform_with_safe_artifacts(df: pd.DataFrame, artifacts: SafePreprocessingArtifacts):
    X_prepared = prepare_features_for_transform(df, artifacts)
    X = artifacts.preprocessor.transform(X_prepared)

    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].to_numpy()

    return X, y


def make_stratified_validation_split(train_df: pd.DataFrame):
    train_part, valid_part = train_test_split(
        train_df,
        test_size=VALID_SIZE,
        stratify=train_df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)


def make_time_validation_split(train_df: pd.DataFrame):
    if TIME_COL not in train_df.columns:
        raise ValueError(f"{TIME_COL} not found in training data")

    ordered = train_df.sort_values(TIME_COL).reset_index(drop=True)
    split_idx = int(len(ordered) * (1.0 - VALID_SIZE))
    train_part = ordered.iloc[:split_idx].copy()
    valid_part = ordered.iloc[split_idx:].copy()
    return train_part.reset_index(drop=True), valid_part.reset_index(drop=True)


def build_safe_report_dict(train_df: pd.DataFrame, test_df: pd.DataFrame, artifacts: SafePreprocessingArtifacts) -> dict:
    train_summary = make_column_summary(train_df)
    test_summary = make_column_summary(test_df)

    report = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "train_fraud_rate": float(train_df[TARGET_COL].mean()),
        "n_drop_cols": len(artifacts.drop_cols),
        "drop_cols": artifacts.drop_cols,
        "n_missing_indicator_cols": len(artifacts.missing_indicator_cols),
        "missing_indicator_cols": artifacts.missing_indicator_cols,
        "n_high_cardinality_cols": len(artifacts.high_cardinality_cols),
        "high_cardinality_cols": artifacts.high_cardinality_cols,
        "n_low_cardinality_cols": len(artifacts.low_cardinality_cols),
        "n_numeric_cols": len(artifacts.numeric_cols),
        "top_20_train_missingness": train_summary.head(20).to_dict(orient="records"),
        "top_20_test_missingness": test_summary.head(20).to_dict(orient="records"),
    }
    return report


def save_safe_reports(train_df: pd.DataFrame, test_df: pd.DataFrame, artifacts: SafePreprocessingArtifacts) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_summary = make_column_summary(train_df)
    test_summary = make_column_summary(test_df)

    train_summary.to_csv(REPORT_DIR / "safe_train_column_summary.csv", index=False)
    test_summary.to_csv(REPORT_DIR / "safe_test_column_summary.csv", index=False)
    train_summary.head(50).to_csv(REPORT_DIR / "safe_train_top_missingness.csv", index=False)
    test_summary.head(50).to_csv(REPORT_DIR / "safe_test_top_missingness.csv", index=False)

    report = build_safe_report_dict(train_df, test_df, artifacts)
    with open(REPORT_DIR / "safe_preprocessing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)