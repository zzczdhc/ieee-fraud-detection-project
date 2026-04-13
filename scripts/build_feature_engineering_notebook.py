from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "05_feature_engineering_for_tree_models.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # 05 Feature Engineering For Tree Models

            This notebook focuses only on the feature engineering ideas used before training tree models. The goal is to make the transformations explicit and easy to discuss in a report.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import importlib
            import sys
            import warnings

            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from IPython.display import display

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import src.fraud_detection.eda as eda_module
            import src.fraud_detection.data as data_module
            import src.fraud_detection.tree_models as tree_models_module

            importlib.reload(eda_module)
            importlib.reload(data_module)
            importlib.reload(tree_models_module)

            from src.fraud_detection.eda import PLOT_COLORS, set_plot_theme
            from src.fraud_detection.data import load_train_data, split_features_target
            from src.fraud_detection.tree_models import (
                add_competition_tree_features,
                build_engineered_feature_catalog,
                build_feature_audit,
            )

            warnings.filterwarnings("ignore")
            set_plot_theme()
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.float_format", "{:,.4f}".format)

            SAMPLE_SIZE = 50_000
            RANDOM_STATE = 42
            """
        ),
        code_cell(
            """
            raw_data = load_train_data(sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)
            enriched_data, engineered_feature_names = add_competition_tree_features(raw_data)

            raw_features, _ = split_features_target(raw_data)
            engineered_features, _ = split_features_target(enriched_data)

            new_columns = [column for column in engineered_features.columns if column not in raw_features.columns]
            feature_audit = build_feature_audit(
                raw_feature_columns=raw_features.columns.tolist(),
                final_feature_columns=engineered_features.columns.tolist(),
                engineered_feature_names=engineered_feature_names,
                dropped_columns=[],
            )
            feature_catalog = build_engineered_feature_catalog(new_columns)

            display(feature_audit.style.hide(axis="index"))
            display(feature_catalog.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Missingness Signal Of Engineered Features"),
        code_cell(
            """
            rows = []
            for column in new_columns:
                rows.append(
                    {
                        "feature": column,
                        "missing_rate": enriched_data[column].isna().mean(),
                        "n_unique": enriched_data[column].nunique(dropna=True),
                    }
                )

            engineered_summary = pd.DataFrame(rows).sort_values("missing_rate", ascending=False).reset_index(drop=True)
            display(engineered_summary.style.format({"missing_rate": "{:.2%}", "n_unique": "{:,.0f}"}).hide(axis="index"))
            """
        ),
        markdown_cell("## Fraud Rate By Selected Engineered Features"),
        code_cell(
            """
            selected_columns = [column for column in ["email_match", "TransactionHour", "bank_type"] if column in enriched_data.columns]

            for column in selected_columns:
                working = enriched_data[[column, "isFraud"]].copy()
                working[column] = working[column].fillna("Missing").astype(str)

                summary = (
                    working.groupby(column, observed=False)["isFraud"]
                    .agg(fraud_rate="mean", count="size")
                    .reset_index()
                    .sort_values(["count", "fraud_rate"], ascending=[False, False])
                    .head(12)
                )

                fig, ax = plt.subplots(figsize=(12, 4))
                sns.barplot(data=summary, x=column, y="fraud_rate", color=PLOT_COLORS["coral"], ax=ax)
                ax.set_title(f"{column}: fraud rate by category")
                ax.set_xlabel("")
                ax.set_ylabel("fraud_rate")
                ax.tick_params(axis="x", rotation=30)
                plt.tight_layout()
                plt.show()

                display(summary)
            """
        ),
        markdown_cell("## Example Rows"),
        code_cell(
            """
            preview_columns = [
                column
                for column in [
                    "TransactionDT",
                    "TransactionAmt",
                    "card1",
                    "card2",
                    "addr1",
                    "P_emaildomain",
                    "R_emaildomain",
                    "uid",
                    "uid2",
                    "bank_type",
                    "email_match",
                    "TransactionHour",
                    "row_missing_count",
                    "row_missing_fraction",
                    "isFraud",
                ]
                if column in enriched_data.columns
            ]

            display(enriched_data[preview_columns].head(10))
            """
        ),
    ]

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(nbf.writes(build_notebook()), encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
