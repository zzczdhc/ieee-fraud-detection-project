from __future__ import annotations

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.config import ensure_outputs_dir


FINAL_MODEL_PARAMS = {
    "n_estimators": 800,
    "max_depth": 7,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 3,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "gamma": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the final cost-sensitive XGBoost model and save validation error-analysis tables.",
    )
    parser.add_argument("--sample-size", type=int, default=80_000)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--pos-weight-multiplier", type=float, default=0.6)
    parser.add_argument("--primary-review-fraction", type=float, default=0.03)
    parser.add_argument("--min-segment-count", type=int, default=50)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ensure_outputs_dir() / "error_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from src.fraud_detection.cost_sensitive_experiments import (
            fit_cost_sensitive_xgboost,
            prepare_cost_sensitive_data,
        )
        from src.fraud_detection.error_analysis import (
            DEFAULT_REVIEW_FRACTIONS,
            run_error_analysis,
        )
    except ImportError as exc:
        raise SystemExit(
            "Could not import the cost-sensitive XGBoost pipeline. "
            "Install the project requirements first, including `xgboost`."
        ) from exc

    data_bundle = prepare_cost_sensitive_data(
        sample_size=args.sample_size,
        random_state=args.random_state,
        nrows=args.nrows,
    )

    result = fit_cost_sensitive_xgboost(
        data_bundle,
        pos_weight_multiplier=args.pos_weight_multiplier,
        param_overrides=FINAL_MODEL_PARAMS,
        label="kaggle_style_more_capacity",
        return_artifacts=True,
    )

    artifacts = run_error_analysis(
        data_bundle.valid_part,
        result["valid_scores"],
        output_dir=args.output_dir,
        model_name="kaggle_style_more_capacity",
        review_fractions=DEFAULT_REVIEW_FRACTIONS,
        primary_review_fraction=args.primary_review_fraction,
        min_segment_count=args.min_segment_count,
        max_examples=args.max_examples,
    )

    print(f"Saved error-analysis artifacts to {args.output_dir}")
    print(artifacts["review_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
