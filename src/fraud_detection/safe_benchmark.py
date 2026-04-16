from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data_prep_safe import (
    SafePreprocessingArtifacts,
    fit_safe_preprocessing,
    load_merged_data_safe,
    make_time_validation_split,
    save_safe_reports,
    transform_with_safe_artifacts,
)
from .metrics import build_curve_frames, build_threshold_metrics_table, compute_classification_metrics
from .train import fit_baseline_experiment
from .tree_models import fit_tree_model_benchmark


@dataclass
class SafeModelSpec:
    name: str
    estimator: object


def fit_safe_pipeline_benchmark(
    sample_size: int | None = 80_000,
    random_state: int = 42,
    save_reports: bool = True,
) -> dict[str, object]:
    train_df, test_df = load_merged_data_safe(nrows=None)
    if sample_size is not None and sample_size < len(train_df):
        train_df = _time_spaced_sample(train_df, sample_size=sample_size)

    if save_reports:
        artifacts_for_report = fit_safe_preprocessing(train_df)
        save_safe_reports(train_df, test_df, artifacts_for_report)

    train_part, valid_part = make_time_validation_split(train_df)
    artifacts = fit_safe_preprocessing(train_part)
    x_train, y_train = transform_with_safe_artifacts(train_part, artifacts)
    x_valid, y_valid = transform_with_safe_artifacts(valid_part, artifacts)

    positive_count = max(1, int(np.sum(y_train)))
    negative_count = max(1, int(len(y_train) - positive_count))
    scale_pos_weight = negative_count / positive_count

    results = []
    prediction_frame = pd.DataFrame(index=valid_part.index)
    curve_frames: dict[str, dict[str, pd.DataFrame]] = {}
    threshold_tables: dict[str, pd.DataFrame] = {}

    for model_spec in _build_safe_model_specs(scale_pos_weight=scale_pos_weight, random_state=random_state):
        estimator = model_spec.estimator
        estimator.fit(x_train, y_train)
        valid_scores = estimator.predict_proba(x_valid)[:, 1]
        metrics = compute_classification_metrics(y_valid, valid_scores)

        results.append(
            {
                "model": model_spec.name,
                "family": "safe_pipeline",
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

    leaderboard = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    if not prediction_frame.empty:
        prediction_frame["SafeTreeEnsembleMean"] = prediction_frame.mean(axis=1)
        ensemble_metrics = compute_classification_metrics(y_valid, prediction_frame["SafeTreeEnsembleMean"])
        leaderboard = pd.concat(
            [
                leaderboard,
                pd.DataFrame(
                    [
                        {
                            "model": "SafeTreeEnsembleMean",
                            "family": "safe_pipeline_ensemble",
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
        curve_frames["SafeTreeEnsembleMean"] = build_curve_frames(y_valid, prediction_frame["SafeTreeEnsembleMean"])
        threshold_tables["SafeTreeEnsembleMean"] = build_threshold_metrics_table(
            y_valid,
            prediction_frame["SafeTreeEnsembleMean"],
        )

    preprocessing_audit = build_safe_preprocessing_audit(train_part, artifacts)
    return {
        "sample_rows": int(len(train_df)),
        "train_rows": int(len(train_part)),
        "validation_rows": int(len(valid_part)),
        "leaderboard": leaderboard,
        "prediction_frame": prediction_frame,
        "curve_frames": curve_frames,
        "threshold_tables": threshold_tables,
        "preprocessing_audit": preprocessing_audit,
        "artifacts": artifacts,
        "y_valid": y_valid,
    }


def fit_cross_pipeline_comparison(
    sample_size: int = 80_000,
    random_state: int = 42,
) -> dict[str, object]:
    baseline = fit_baseline_experiment(
        sample_size=sample_size,
        random_state=random_state,
        missing_threshold=0.95,
    )

    baseline_metrics = baseline["metrics"]
    baseline_row = pd.DataFrame(
        [
            {
                "model": "BaselineLogisticRandomSplit",
                "family": "baseline_notebook",
                "roc_auc": baseline_metrics["roc_auc"],
                "average_precision": baseline_metrics["average_precision"],
                "precision_at_top_1pct": baseline_metrics["precision_at_top_1pct"],
                "recall_at_top_1pct": baseline_metrics["recall_at_top_1pct"],
                "precision_at_top_5pct": baseline_metrics["precision_at_top_5pct"],
                "recall_at_top_5pct": baseline_metrics["recall_at_top_5pct"],
            }
        ]
    )

    tree_benchmark = fit_tree_model_benchmark(
        sample_size=sample_size,
        valid_fraction=0.2,
        random_state=random_state,
        missing_threshold=0.98,
    )
    safe_benchmark = fit_safe_pipeline_benchmark(
        sample_size=sample_size,
        random_state=random_state,
        save_reports=True,
    )

    comparison = pd.concat(
        [
            baseline_row,
            tree_benchmark["comparison"],
            safe_benchmark["leaderboard"],
        ],
        ignore_index=True,
    ).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    method_summary = pd.DataFrame(
        [
            {
                "pipeline": "baseline_notebook",
                "split": "random stratified split",
                "preprocessing": "median impute + one-hot + scaling",
                "special_handling": "drop very-missing columns",
            },
            {
                "pipeline": "03_tree_models",
                "split": "time-based validation",
                "preprocessing": "median impute + ordinal encoding for trees",
                "special_handling": "competition-style engineered features",
            },
            {
                "pipeline": "safe_pipeline",
                "split": "time-based validation",
                "preprocessing": "missing indicators + frequency encoding + one-hot",
                "special_handling": "schema checks and safer categorical handling",
            },
        ]
    )

    strengths_gaps = pd.DataFrame(
        [
            {
                "pipeline": "baseline_notebook",
                "what_improves": "simple and easy to explain; good teaching baseline",
                "what_is_missing": "random split is optimistic for fraud data; weak handling of high-cardinality categories",
            },
            {
                "pipeline": "03_tree_models",
                "what_improves": "best alignment with competition logic through time split and engineered interactions",
                "what_is_missing": "less train/test hygiene than safe pipeline; no explicit schema audit",
            },
            {
                "pipeline": "safe_pipeline",
                "what_improves": "better robustness for missingness and high-cardinality categoricals; safer train/test consistency",
                "what_is_missing": "currently less domain-specific than uid-style competition features",
            },
        ]
    )

    return {
        "comparison": comparison,
        "baseline_row": baseline_row,
        "tree_benchmark": tree_benchmark,
        "safe_benchmark": safe_benchmark,
        "method_summary": method_summary,
        "strengths_gaps": strengths_gaps,
    }


def build_safe_preprocessing_audit(
    train_part: pd.DataFrame,
    artifacts: SafePreprocessingArtifacts,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "drop_high_missing_columns",
                "count": len(artifacts.drop_cols),
                "note": "removed columns with missing rate >= 98%",
            },
            {
                "stage": "add_missing_indicators",
                "count": len(artifacts.missing_indicator_cols),
                "note": "binary flags for columns with missing rate >= 10%",
            },
            {
                "stage": "frequency_encoded_high_cardinality",
                "count": len(artifacts.high_cardinality_cols),
                "note": "high-cardinality categorical columns replaced by frequency encodings",
            },
            {
                "stage": "one_hot_low_cardinality",
                "count": len(artifacts.low_cardinality_cols),
                "note": "low-cardinality categoricals kept for one-hot encoding",
            },
            {
                "stage": "numeric_columns_after_safe_prep",
                "count": len(artifacts.numeric_cols),
                "note": "numeric columns passed into the safe preprocessor",
            },
            {
                "stage": "time_split_rows",
                "count": len(train_part),
                "note": "rows used to fit preprocessing artifacts",
            },
        ]
    )


def _build_safe_model_specs(scale_pos_weight: float, random_state: int) -> list[SafeModelSpec]:
    model_specs: list[SafeModelSpec] = [
        SafeModelSpec(
            name="SafeLogistic",
            estimator=LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=random_state,
            ),
        ),
        SafeModelSpec(
            name="SafeRandomForest",
            estimator=RandomForestClassifier(
                n_estimators=250,
                max_depth=14,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
    ]

    try:
        from xgboost import XGBClassifier

        model_specs.append(
            SafeModelSpec(
                name="SafeXGBoost",
                estimator=XGBClassifier(
                    n_estimators=450,
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
            SafeModelSpec(
                name="SafeLightGBM",
                estimator=LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=64,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=50,
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

    return model_specs


def _time_spaced_sample(frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if "TransactionDT" not in frame.columns:
        return frame.sample(n=sample_size, random_state=42).reset_index(drop=True)

    ordered = frame.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, num=sample_size, dtype=int)
    return ordered.iloc[np.unique(positions)].reset_index(drop=True)
