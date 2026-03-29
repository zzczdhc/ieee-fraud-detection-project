from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def drop_high_missing_columns(
    frame: pd.DataFrame,
    threshold: float = 0.95,
    protected_columns: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    protected = set(protected_columns or [])
    missing_fraction = frame.isna().mean()
    dropped_columns = [
        column
        for column, missing_rate in missing_fraction.items()
        if missing_rate >= threshold and column not in protected
    ]
    filtered = frame.drop(columns=dropped_columns, errors="ignore")
    return filtered, dropped_columns


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable features remain after preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor, numeric_features, categorical_features
