from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import load_train_data, split_features_target, summarize_frame
from .features import build_preprocessor, drop_high_missing_columns
from .metrics import build_curve_frames, build_threshold_metrics_table, compute_classification_metrics


@dataclass
class ModelSpec:
    name: str
    estimator: object


ENGINEERED_FEATURE_DESCRIPTIONS = {
    "TransactionAmt_Log": "log-scaled transaction amount",
    "TransactionAmt_Cents": "fractional cent pattern from transaction amount",
    "TransactionDay": "transaction day derived from TransactionDT",
    "TransactionWeek": "transaction week derived from TransactionDT",
    "TransactionHour": "transaction hour derived from TransactionDT",
    "TransactionDayOfWeek": "transaction weekday proxy from TransactionDT",
    "uid": "card + address style pseudo-client identifier",
    "uid2": "uid plus purchaser email domain",
    "bank_type": "card network and card type interaction",
    "email_match": "whether purchaser and recipient email domains match",
    "email_pair": "joint purchaser / recipient email pair",
    "DeviceInfoPrefix": "shortened device family prefix",
    "row_missing_count": "number of missing values in the row",
    "row_missing_fraction": "fraction of missing values in the row",
}


def add_competition_tree_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    enriched = frame.copy()
    engineered_features: list[str] = []

    if "TransactionAmt" in enriched.columns:
        amount = enriched["TransactionAmt"].fillna(0)
        enriched["TransactionAmt_Log"] = np.log1p(amount)
        enriched["TransactionAmt_Cents"] = ((amount * 1000) % 1000).round().astype("Int64")
        engineered_features.extend(["TransactionAmt_Log", "TransactionAmt_Cents"])

    if "TransactionDT" in enriched.columns:
        transaction_day = (enriched["TransactionDT"] // (24 * 60 * 60)).astype("Int64")
        enriched["TransactionDay"] = transaction_day
        enriched["TransactionWeek"] = (transaction_day // 7).astype("Int64")
        enriched["TransactionHour"] = ((enriched["TransactionDT"] // 3600) % 24).astype("Int64")
        enriched["TransactionDayOfWeek"] = (transaction_day % 7).astype("Int64")
        engineered_features.extend(
            ["TransactionDay", "TransactionWeek", "TransactionHour", "TransactionDayOfWeek"]
        )

    card_cols = [
        column
        for column in ["card1", "card2", "card3", "card5", "addr1", "addr2"]
        if column in enriched.columns
    ]
    if card_cols:
        enriched["uid"] = _combine_as_string(enriched, card_cols)
        engineered_features.append("uid")

    uid2_cols = [column for column in ["uid", "P_emaildomain"] if column in enriched.columns]
    if uid2_cols:
        enriched["uid2"] = _combine_as_string(enriched, uid2_cols)
        engineered_features.append("uid2")

    bank_cols = [column for column in ["card4", "card6"] if column in enriched.columns]
    if bank_cols:
        enriched["bank_type"] = _combine_as_string(enriched, bank_cols)
        engineered_features.append("bank_type")

    if "P_emaildomain" in enriched.columns and "R_emaildomain" in enriched.columns:
        p_email = enriched["P_emaildomain"].fillna("missing").astype(str)
        r_email = enriched["R_emaildomain"].fillna("missing").astype(str)
        enriched["email_match"] = (p_email == r_email).astype("int8")
        enriched["email_pair"] = p_email + "__" + r_email
        engineered_features.extend(["email_match", "email_pair"])

    if "DeviceInfo" in enriched.columns:
        device_series = enriched["DeviceInfo"].fillna("missing").astype(str)
        enriched["DeviceInfoPrefix"] = device_series.str.split("/", n=1).str[0].str.slice(0, 30)
        engineered_features.append("DeviceInfoPrefix")

    enriched["row_missing_count"] = enriched.isna().sum(axis=1).astype("int32")
    enriched["row_missing_fraction"] = enriched.isna().mean(axis=1).astype("float32")
    engineered_features.extend(["row_missing_count", "row_missing_fraction"])
    return enriched, engineered_features


def build_tree_preprocessor(frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = frame.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_ordinal_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable features remain after preprocessing.")

    return ColumnTransformer(transformers=transformers), numeric_features, categorical_features


def time_based_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
    valid_fraction: float = 0.2,
    time_column: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if time_column not in features.columns:
        raise KeyError(f"Expected '{time_column}' in features for time-based validation.")
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1.")

    ordered_index = features[time_column].sort_values(kind="mergesort").index
    split_idx = int(len(ordered_index) * (1 - valid_fraction))
    train_index = ordered_index[:split_idx]
    valid_index = ordered_index[split_idx:]

    return (
        features.loc[train_index].copy(),
        features.loc[valid_index].copy(),
        target.loc[train_index].copy(),
        target.loc[valid_index].copy(),
    )


def fit_tree_model_benchmark(
    sample_size: int | None = 80_000,
    valid_fraction: float = 0.2,
    random_state: int = 42,
    missing_threshold: float = 0.98,
) -> dict[str, object]:
    raw_data = load_train_data(sample_size=None, random_state=random_state)
    raw_dataset_summary = summarize_frame(raw_data)
    raw_features, target = split_features_target(raw_data)

    enriched_data, engineered_feature_names = add_competition_tree_features(raw_data)
    if sample_size is not None and sample_size < len(enriched_data):
        enriched_data = _time_spaced_sample(
            enriched_data,
            sample_size=sample_size,
            random_state=random_state,
        )

    dataset_summary = summarize_frame(enriched_data)
    features, target = split_features_target(enriched_data)
    x_train, x_valid, y_train, y_valid = time_based_validation_split(
        features,
        target,
        valid_fraction=valid_fraction,
    )

    x_train, dropped_columns = drop_high_missing_columns(
        x_train,
        threshold=missing_threshold,
        protected_columns=["TransactionDT", "uid", "uid2"],
    )
    x_valid = x_valid.drop(columns=dropped_columns, errors="ignore")

    feature_audit = build_feature_audit(
        raw_feature_columns=raw_features.columns.tolist(),
        final_feature_columns=x_train.columns.tolist(),
        engineered_feature_names=engineered_feature_names,
        dropped_columns=dropped_columns,
    )

    logistic_result = fit_logistic_reference(
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        random_state=random_state,
    )

    tree_result = fit_tree_ensemble_models(
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        random_state=random_state,
    )

    comparison = pd.concat(
        [logistic_result["leaderboard"], tree_result["leaderboard"]],
        ignore_index=True,
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    prediction_frame = pd.concat(
        [logistic_result["prediction_frame"], tree_result["prediction_frame"]],
        axis=1,
    )
    curves = {**logistic_result["curve_frames"], **tree_result["curve_frames"]}
    threshold_tables = {**logistic_result["threshold_tables"], **tree_result["threshold_tables"]}
    feature_importances = tree_result["feature_importances"]

    tree_columns = [column for column in tree_result["prediction_frame"].columns]
    if tree_columns:
        prediction_frame["TreeEnsembleMean"] = tree_result["prediction_frame"][tree_columns].mean(axis=1)
        ensemble_auc = roc_auc_score(y_valid, prediction_frame["TreeEnsembleMean"])
        ensemble_metrics = compute_classification_metrics(y_valid, prediction_frame["TreeEnsembleMean"])
        comparison = pd.concat(
            [
                comparison,
                pd.DataFrame(
                    [
                        {
                            "model": "TreeEnsembleMean",
                            "family": "ensemble",
                            "roc_auc": ensemble_metrics["roc_auc"],
                            "average_precision": ensemble_metrics["average_precision"],
                            "precision_at_top_1pct": ensemble_metrics["precision_at_top_1pct"],
                            "recall_at_top_1pct": ensemble_metrics["recall_at_top_1pct"],
                            "precision_at_top_5pct": ensemble_metrics["precision_at_top_5pct"],
                            "recall_at_top_5pct": ensemble_metrics["recall_at_top_5pct"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).sort_values("roc_auc", ascending=False).reset_index(drop=True)
        curves["TreeEnsembleMean"] = build_curve_frames(y_valid, prediction_frame["TreeEnsembleMean"])
        threshold_tables["TreeEnsembleMean"] = build_threshold_metrics_table(
            y_valid,
            prediction_frame["TreeEnsembleMean"],
        )
    else:
        ensemble_auc = np.nan

    return {
        "raw_dataset_summary": raw_dataset_summary,
        "dataset_summary": dataset_summary,
        "sample_rows": int(len(enriched_data)),
        "train_rows": int(len(x_train)),
        "validation_rows": int(len(x_valid)),
        "missing_threshold": float(missing_threshold),
        "dropped_high_missing_columns": dropped_columns,
        "feature_audit": feature_audit,
        "engineered_feature_names": engineered_feature_names,
        "comparison": comparison,
        "prediction_frame": prediction_frame,
        "curve_frames": curves,
        "threshold_tables": threshold_tables,
        "feature_importances": feature_importances,
        "y_valid": y_valid,
        "ensemble_roc_auc": float(ensemble_auc),
    }


def fit_logistic_reference(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    random_state: int = 42,
) -> dict[str, object]:
    preprocessor, _, _ = build_preprocessor(x_train)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(x_train, y_train)

    valid_scores = pipeline.predict_proba(x_valid)[:, 1]
    metrics = compute_classification_metrics(y_valid, valid_scores)

    leaderboard = pd.DataFrame(
        [
            {
                "model": "LogisticRegression",
                "family": "linear_baseline",
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "precision_at_top_1pct": metrics["precision_at_top_1pct"],
                "recall_at_top_1pct": metrics["recall_at_top_1pct"],
                "precision_at_top_5pct": metrics["precision_at_top_5pct"],
                "recall_at_top_5pct": metrics["recall_at_top_5pct"],
            }
        ]
    )

    return {
        "leaderboard": leaderboard,
        "prediction_frame": pd.DataFrame({"LogisticRegression": valid_scores}, index=y_valid.index),
        "curve_frames": {"LogisticRegression": build_curve_frames(y_valid, valid_scores)},
        "threshold_tables": {"LogisticRegression": build_threshold_metrics_table(y_valid, valid_scores)},
    }


def fit_tree_ensemble_models(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    random_state: int = 42,
) -> dict[str, object]:
    preprocessor, _, _ = build_tree_preprocessor(x_train)
    x_train_encoded = preprocessor.fit_transform(x_train)
    x_valid_encoded = preprocessor.transform(x_valid)
    feature_names = list(preprocessor.get_feature_names_out())

    positive_count = max(1, int(y_train.sum()))
    negative_count = max(1, int(len(y_train) - positive_count))
    scale_pos_weight = negative_count / positive_count

    results = []
    prediction_frame = pd.DataFrame(index=y_valid.index)
    curve_frames: dict[str, dict[str, pd.DataFrame]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}
    feature_importances: dict[str, pd.DataFrame] = {}

    for model_spec in _build_tree_model_specs(scale_pos_weight=scale_pos_weight, random_state=random_state):
        estimator = model_spec.estimator
        estimator.fit(x_train_encoded, y_train)
        valid_scores = estimator.predict_proba(x_valid_encoded)[:, 1]
        metrics = compute_classification_metrics(y_valid, valid_scores)

        results.append(
            {
                "model": model_spec.name,
                "family": "tree_model",
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "precision_at_top_1pct": metrics["precision_at_top_1pct"],
                "recall_at_top_1pct": metrics["recall_at_top_1pct"],
                "precision_at_top_5pct": metrics["precision_at_top_5pct"],
                "recall_at_top_5pct": metrics["recall_at_top_5pct"],
            }
        )
        prediction_frame[model_spec.name] = valid_scores
        curve_frames[model_spec.name] = build_curve_frames(y_valid, valid_scores)
        threshold_tables[model_spec.name] = build_threshold_metrics_table(y_valid, valid_scores)
        feature_importances[model_spec.name] = _extract_feature_importance(estimator, feature_names)

    leaderboard = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    return {
        "leaderboard": leaderboard,
        "prediction_frame": prediction_frame,
        "curve_frames": curve_frames,
        "threshold_tables": threshold_tables,
        "feature_importances": feature_importances,
    }


def build_feature_audit(
    raw_feature_columns: list[str],
    final_feature_columns: list[str],
    engineered_feature_names: list[str],
    dropped_columns: list[str],
) -> pd.DataFrame:
    engineered_in_final = [column for column in engineered_feature_names if column in final_feature_columns]
    raw_in_final = [column for column in final_feature_columns if column not in engineered_feature_names]

    rows = [
        {
            "stage": "raw_features_before_engineering",
            "count": len(raw_feature_columns),
            "note": "all predictors after merging transaction and identity tables",
        },
        {
            "stage": "engineered_features_added",
            "count": len(engineered_feature_names),
            "note": "new columns created for time, uid-style interactions, email, and missingness",
        },
        {
            "stage": "dropped_for_high_missingness",
            "count": len(dropped_columns),
            "note": "columns removed because missing rate exceeded the threshold",
        },
        {
            "stage": "final_features_used",
            "count": len(final_feature_columns),
            "note": "columns actually passed into the model pipeline",
        },
        {
            "stage": "raw_features_retained",
            "count": len(raw_in_final),
            "note": "original Kaggle features kept after filtering",
        },
        {
            "stage": "engineered_features_retained",
            "count": len(engineered_in_final),
            "note": "newly engineered features that survived filtering",
        },
    ]
    return pd.DataFrame(rows)


def build_engineered_feature_catalog(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": feature_name,
                "description": ENGINEERED_FEATURE_DESCRIPTIONS.get(feature_name, "engineered feature"),
            }
            for feature_name in feature_names
        ]
    )


def _build_tree_model_specs(scale_pos_weight: float, random_state: int) -> list[ModelSpec]:
    model_specs: list[ModelSpec] = [
        ModelSpec(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                max_depth=14,
                min_samples_leaf=25,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="ExtraTrees",
            estimator=ExtraTreesClassifier(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    ]

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model_specs.append(
            ModelSpec(
                name="HistGradientBoosting",
                estimator=HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=8,
                    max_iter=300,
                    min_samples_leaf=100,
                    random_state=random_state,
                ),
            )
        )
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier

        model_specs.append(
            ModelSpec(
                name="XGBoost",
                estimator=XGBClassifier(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    min_child_weight=5,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        model_specs.append(
            ModelSpec(
                name="LightGBM",
                estimator=LGBMClassifier(
                    n_estimators=600,
                    learning_rate=0.05,
                    num_leaves=64,
                    max_depth=-1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=50,
                    reg_lambda=1.0,
                    objective="binary",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            )
        )
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        model_specs.append(
            ModelSpec(
                name="CatBoost",
                estimator=CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=8,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                    verbose=False,
                    random_seed=random_state,
                ),
            )
        )
    except Exception:
        pass

    return model_specs


def _extract_feature_importance(estimator: object, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    frame = pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float)}
    )
    return frame.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def _make_ordinal_encoder() -> OrdinalEncoder:
    try:
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
    except TypeError:
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)


def _combine_as_string(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    pieces = [frame[column].fillna("missing").astype(str) for column in columns]
    combined = pieces[0]
    for piece in pieces[1:]:
        combined = combined + "__" + piece
    return combined


def _time_spaced_sample(frame: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    if "TransactionDT" not in frame.columns:
        return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    ordered = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=sample_size, dtype=int)
    sampled = ordered.iloc[np.unique(positions)].copy()
    return sampled.reset_index(drop=True)
