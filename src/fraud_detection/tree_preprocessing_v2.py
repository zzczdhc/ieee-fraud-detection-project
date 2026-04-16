from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

M_COLS = [f"M{i}" for i in range(1, 10)]

EXTRA_COUNT_ENCODE_COLS = [
    "ProductCD",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceType",
    "uid1", "uid2", "uid3",
]

GROUP_AMOUNT_COLS = [
    "card1",
    "card4",
    "addr1",
    "uid1",
    "uid2",
]


@dataclass
class TreePreprocessorV2Artifacts:
    drop_cols: list[str]
    object_cols_to_drop: list[str]
    count_encode_cols: list[str]
    count_maps: dict[str, dict]
    missing_indicator_cols: list[str]
    group_amount_maps: dict[str, dict[str, dict]]
    numeric_medians: dict[str, float]
    feature_columns: list[str]
    add_missing_indicators: bool
    add_group_amount_features: bool


def _to_string_key(series: pd.Series) -> pd.Series:
    return series.fillna("__nan__").astype(str)


def _safe_first_token(series: pd.Series) -> pd.Series:
    s = _to_string_key(series).str.lower()
    s = s.str.replace(r"[\(\)/_\-]+", " ", regex=True)
    return s.str.split().str[0]


def add_tree_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("-", "_") for c in df.columns]

    # time features
    if TIME_COL in df.columns:
        seconds = df[TIME_COL].fillna(0)
        transaction_day = (seconds // (24 * 3600)).astype("int32")

        df["TransactionHour"] = ((seconds // 3600) % 24).astype("int32")
        df["TransactionDay"] = transaction_day
        df["TransactionWeek"] = (transaction_day // 7).astype("int32")
        df["TransactionDayOfWeek"] = (transaction_day % 7).astype("int32")

    # amount features
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")

        df["TransactionAmt_log1p"] = np.log1p(amt.clip(lower=0))
        amt_fill = amt.fillna(0)
        frac = amt_fill - np.floor(amt_fill)
        df["TransactionAmt_cents"] = (frac * 1000).round().astype("float32")

    # email features
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            s = _to_string_key(df[col]).str.lower()
            df[f"{col}_prefix"] = s.str.split(".").str[0]
            df[f"{col}_suffix"] = s.str.split(".").str[-1]

    # device / browser / os
    if "DeviceInfo" in df.columns:
        df["DeviceInfo_device"] = _safe_first_token(df["DeviceInfo"])

    if "id_30" in df.columns:
        df["id_30_os"] = _safe_first_token(df["id_30"])

    if "id_31" in df.columns:
        df["id_31_browser"] = _safe_first_token(df["id_31"])

    # M columns
    for col in M_COLS:
        if col in df.columns:
            df[f"{col}_bin"] = df[col].map({"T": 1, "F": 0}).astype("float32")

    # uid-style combos
    if "card1" in df.columns and "card2" in df.columns and "card3" in df.columns and "card5" in df.columns:
        df["uid1"] = (
            _to_string_key(df["card1"]) + "_" +
            _to_string_key(df["card2"]) + "_" +
            _to_string_key(df["card3"]) + "_" +
            _to_string_key(df["card5"])
        )

    if "uid1" in df.columns and "addr1" in df.columns and "addr2" in df.columns:
        df["uid2"] = (
            _to_string_key(df["uid1"]) + "_" +
            _to_string_key(df["addr1"]) + "_" +
            _to_string_key(df["addr2"])
        )

    if "uid2" in df.columns and "P_emaildomain" in df.columns:
        df["uid3"] = _to_string_key(df["uid2"]) + "_" + _to_string_key(df["P_emaildomain"])

    return df


def add_missing_indicator_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return df

    indicator_df = pd.DataFrame(
        {f"{col}__missing": df[col].isna().astype("int8") for col in valid_cols},
        index=df.index,
    )

    return pd.concat([df, indicator_df], axis=1)


def choose_drop_cols(df: pd.DataFrame, missing_threshold: float = 0.999) -> list[str]:
    protected = {TARGET_COL, ID_COL}
    missing_fraction = df.isna().mean()
    nunique = df.nunique(dropna=False)

    drop_cols = [
        col for col in df.columns
        if col not in protected and (missing_fraction[col] >= missing_threshold or nunique[col] <= 1)
    ]
    return sorted(drop_cols)


def choose_missing_indicator_cols(df: pd.DataFrame) -> list[str]:
    protected = {TARGET_COL, ID_COL}
    missing_fraction = df.isna().mean()

    cols = [
        col for col in df.columns
        if col not in protected and 0.05 <= missing_fraction[col] < 0.999
    ]
    return sorted(cols)


def get_object_like_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def fit_count_maps(train_df: pd.DataFrame, columns: list[str]) -> dict[str, dict]:
    maps = {}
    for col in columns:
        if col in train_df.columns:
            key = _to_string_key(train_df[col])
            counts = key.value_counts(dropna=False)
            maps[col] = np.log1p(counts).to_dict()
    return maps


def apply_count_encoding(df: pd.DataFrame, count_maps: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in count_maps.items():
        if col in df.columns:
            key = _to_string_key(df[col])
            df[f"{col}__count"] = key.map(mapping).fillna(0).astype("float32")
    return df


def fit_group_amount_maps(train_df: pd.DataFrame, group_cols: list[str]) -> dict[str, dict[str, dict]]:
    maps = {}

    if "TransactionAmt" not in train_df.columns:
        return maps

    amt = pd.to_numeric(train_df["TransactionAmt"], errors="coerce")

    for col in group_cols:
        if col in train_df.columns:
            key = _to_string_key(train_df[col])
            tmp = pd.DataFrame({"_key": key, "TransactionAmt": amt})
            stats = tmp.groupby("_key")["TransactionAmt"].agg(["mean", "std"])

            maps[col] = {
                "mean": stats["mean"].to_dict(),
                "std": stats["std"].fillna(0).to_dict(),
            }

    return maps


def apply_group_amount_maps(df: pd.DataFrame, group_maps: dict[str, dict[str, dict]]) -> pd.DataFrame:
    df = df.copy()

    if "TransactionAmt" not in df.columns:
        return df

    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")

    for col, stats in group_maps.items():
        if col in df.columns:
            key = _to_string_key(df[col])

            mean_map = key.map(stats["mean"]).astype("float32")
            std_map = key.map(stats["std"]).astype("float32")

            df[f"TransactionAmt_diff_mean_{col}"] = (amt - mean_map).astype("float32")
            df[f"TransactionAmt_to_mean_{col}"] = (amt / mean_map).replace([np.inf, -np.inf], np.nan).astype("float32")
            df[f"TransactionAmt_z_{col}"] = ((amt - mean_map) / std_map.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).astype("float32")

    return df


def fit_tree_preprocessor_v2(
    train_df: pd.DataFrame,
    add_missing_indicators: bool = True,
    add_group_amount_features: bool = True,
    drop_missing_threshold: float = 0.999,
) -> TreePreprocessorV2Artifacts:
    df = add_tree_features_v2(train_df)

    # drop only truly unusable columns
    drop_cols = choose_drop_cols(df, missing_threshold=drop_missing_threshold)
    df = df.drop(columns=drop_cols, errors="ignore")

    # object-like columns
    object_cols = get_object_like_cols(df)

    # count-encode both true string categoricals and selected id-like columns
    extra_cols = [c for c in EXTRA_COUNT_ENCODE_COLS if c in df.columns]
    count_encode_cols = sorted(set(object_cols + extra_cols))

    count_maps = fit_count_maps(df, count_encode_cols)

    if add_group_amount_features:
        group_amount_maps = fit_group_amount_maps(df, [c for c in GROUP_AMOUNT_COLS if c in df.columns])
    else:
        group_amount_maps = {}

    use_missing_indicators = add_missing_indicators
    use_group_amount_features = add_group_amount_features

    if use_missing_indicators:
        missing_indicator_cols = choose_missing_indicator_cols(df)
    else:
        missing_indicator_cols = []

    df = add_missing_indicator_features(df, missing_indicator_cols)
    df = apply_count_encoding(df, count_maps)
    df = apply_group_amount_maps(df, group_amount_maps)

    # drop only original object/string columns after encoding
    df = df.drop(columns=object_cols, errors="ignore")

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    numeric_medians = X.median(numeric_only=True).to_dict()
    feature_columns = X.columns.tolist()

    return TreePreprocessorV2Artifacts(
        drop_cols=drop_cols,
        object_cols_to_drop=object_cols,
        count_encode_cols=count_encode_cols,
        count_maps=count_maps,
        missing_indicator_cols=missing_indicator_cols,
        group_amount_maps=group_amount_maps,
        numeric_medians=numeric_medians,
        feature_columns=feature_columns,
        add_missing_indicators=use_missing_indicators,
        add_group_amount_features=use_group_amount_features,
    )


def transform_tree_preprocessor_v2(
    df: pd.DataFrame,
    artifacts: TreePreprocessorV2Artifacts,
    impute_numeric: bool = False,
) -> pd.DataFrame:
    out = add_tree_features_v2(df)
    out = out.drop(columns=artifacts.drop_cols, errors="ignore")

    if artifacts.add_missing_indicators:
        out = add_missing_indicator_features(out, artifacts.missing_indicator_cols)

    out = apply_count_encoding(out, artifacts.count_maps)

    if artifacts.add_group_amount_features:
        out = apply_group_amount_maps(out, artifacts.group_amount_maps)

    out = out.drop(columns=artifacts.object_cols_to_drop, errors="ignore")

    X = out.drop(columns=[TARGET_COL, ID_COL], errors="ignore").copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    for col in artifacts.feature_columns:
        if col not in X.columns:
            X[col] = np.nan

    X = X[artifacts.feature_columns].copy()

    if impute_numeric:
        fill_values = pd.Series(artifacts.numeric_medians)
        X = X.fillna(fill_values)

    return X
