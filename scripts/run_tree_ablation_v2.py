from pathlib import Path
import sys
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

# adapt these imports to your repo if names differ
from src.fraud_detection.data_prep_safe import (
    load_merged_data_safe,
    make_time_validation_split,
)
from src.fraud_detection.tree_preprocessing_v2 import (
    fit_tree_preprocessor_v2,
    transform_tree_preprocessor_v2,
)


def top5_metrics(y_true, y_score, frac=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    k = max(1, int(len(y_true) * frac))
    idx = np.argsort(-y_score)[:k]

    precision_at_k = y_true[idx].mean()
    recall_at_k = y_true[idx].sum() / max(1, y_true.sum())

    return precision_at_k, recall_at_k


def evaluate_variant(train_part, valid_part, name, add_missing_indicators, add_group_amount_features):
    artifacts = fit_tree_preprocessor_v2(
        train_part,
        add_missing_indicators=add_missing_indicators,
        add_group_amount_features=add_group_amount_features,
        drop_missing_threshold=0.999,
    )

    # For XGBoost, leave numeric NaN values as NaN
    X_train = transform_tree_preprocessor_v2(train_part, artifacts, impute_numeric=False)
    X_valid = transform_tree_preprocessor_v2(valid_part, artifacts, impute_numeric=False)

    y_train = train_part["isFraud"].values
    y_valid = valid_part["isFraud"].values

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    valid_proba = model.predict_proba(X_valid)[:, 1]

    precision_at_5, recall_at_5 = top5_metrics(y_valid, valid_proba, frac=0.05)

    metrics = {
        "variant": name,
        "roc_auc": float(roc_auc_score(y_valid, valid_proba)),
        "average_precision": float(average_precision_score(y_valid, valid_proba)),
        "precision_at_top_5pct": float(precision_at_5),
        "recall_at_top_5pct": float(recall_at_5),
        "n_features": int(X_train.shape[1]),
    }
    return metrics


def main():
    train_df, _ = load_merged_data_safe()
    train_part, valid_part = make_time_validation_split(train_df)

    results = []

    # 1. core v2 without missing indicators or group amount features
    results.append(
        evaluate_variant(
            train_part, valid_part,
            name="v2_core",
            add_missing_indicators=False,
            add_group_amount_features=False,
        )
    )

    # 2. add missing indicators
    results.append(
        evaluate_variant(
            train_part, valid_part,
            name="v2_plus_missing_flags",
            add_missing_indicators=True,
            add_group_amount_features=False,
        )
    )

    # 3. full version
    results.append(
        evaluate_variant(
            train_part, valid_part,
            name="v2_full",
            add_missing_indicators=True,
            add_group_amount_features=True,
        )
    )

    output_path = PROJECT_ROOT / "outputs" / "tree_ablation_v2_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()
