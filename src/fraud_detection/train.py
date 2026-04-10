from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import OUTPUTS_DIR, ensure_outputs_dir


def fit_baseline_experiment(
    sample_size: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    missing_threshold: float = 0.95,
) -> dict[str, object]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    from .data import load_train_data, split_features_target, summarize_frame
    from .features import build_preprocessor, drop_high_missing_columns
    from .metrics import build_threshold_metrics_table, compute_classification_metrics

    data = load_train_data(sample_size=sample_size, random_state=random_state)
    dataset_summary = summarize_frame(data)

    features, target = split_features_target(data)
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    x_train, dropped_columns = drop_high_missing_columns(
        x_train,
        threshold=missing_threshold,
    )
    x_valid = x_valid.drop(columns=dropped_columns, errors="ignore")

    preprocessor, numeric_features, categorical_features = build_preprocessor(x_train)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)

    validation_scores = pipeline.predict_proba(x_valid)[:, 1]
    metrics = compute_classification_metrics(y_valid, validation_scores)
    threshold_table = build_threshold_metrics_table(y_valid, validation_scores)
    best_f1_row = threshold_table.sort_values("f1", ascending=False).iloc[0].to_dict()

    return {
        "pipeline": pipeline,
        "dataset_summary": dataset_summary,
        "x_train": x_train,
        "x_valid": x_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "validation_scores": validation_scores,
        "dropped_high_missing_columns": dropped_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "metrics": metrics,
        "threshold_table": threshold_table,
        "best_f1_threshold": best_f1_row,
    }


def run_baseline(
    sample_size: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    missing_threshold: float = 0.95,
    output_path: Path | str | None = None,
) -> dict[str, object]:
    experiment = fit_baseline_experiment(
        sample_size=sample_size,
        test_size=test_size,
        random_state=random_state,
        missing_threshold=missing_threshold,
    )

    results: dict[str, object] = {
        "dataset_summary": experiment["dataset_summary"],
        "train_rows": int(len(experiment["x_train"])),
        "validation_rows": int(len(experiment["x_valid"])),
        "dropped_high_missing_columns": experiment["dropped_high_missing_columns"],
        "n_numeric_features": len(experiment["numeric_features"]),
        "n_categorical_features": len(experiment["categorical_features"]),
        "metrics": experiment["metrics"],
        "best_f1_threshold": experiment["best_f1_threshold"],
        "threshold_table": experiment["threshold_table"].to_dict(orient="records"),
    }

    output_file = _resolve_output_path(output_path)
    output_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _resolve_output_path(output_path: Path | str | None) -> Path:
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    ensure_outputs_dir()
    return OUTPUTS_DIR / "baseline_metrics.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a baseline fraud detector.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional number of rows to sample from the training data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splitting and sampling.",
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.95,
        help="Drop columns whose missing fraction is at least this value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for metrics output.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    results = run_baseline(
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.random_state,
        missing_threshold=args.missing_threshold,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
