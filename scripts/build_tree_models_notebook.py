from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "03_tree_models.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # 03 Tree Models

            This notebook is the stricter follow-up to the logistic baseline. It keeps the baseline notebook's reporting style, but upgrades the modeling side in three ways:

            1. switch from a random split to a **time-based validation split**
            2. add **competition-style feature engineering** inspired by the 1st-place solution
            3. compare a **linear reference model** against several **tree ensembles**
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import importlib
            import sys
            import warnings

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            from IPython.display import display

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            import src.fraud_detection.eda as eda_module
            import src.fraud_detection.tree_models as tree_models_module

            importlib.reload(eda_module)
            importlib.reload(tree_models_module)

            from src.fraud_detection.eda import PLOT_COLORS, set_plot_theme
            from src.fraud_detection.tree_models import (
                build_engineered_feature_catalog,
                fit_tree_model_benchmark,
            )

            warnings.filterwarnings("ignore")
            set_plot_theme()
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.float_format", "{:,.4f}".format)

            SAMPLE_SIZE = 80_000
            VALID_FRACTION = 0.20
            MISSING_THRESHOLD = 0.98
            RANDOM_STATE = 42
            """
        ),
        code_cell(
            """
            benchmark = fit_tree_model_benchmark(
                sample_size=SAMPLE_SIZE,
                valid_fraction=VALID_FRACTION,
                missing_threshold=MISSING_THRESHOLD,
                random_state=RANDOM_STATE,
            )

            comparison = benchmark["comparison"].copy()
            threshold_tables = benchmark["threshold_tables"]
            curve_frames = benchmark["curve_frames"]
            prediction_frame = benchmark["prediction_frame"].copy()
            y_valid = np.asarray(benchmark["y_valid"])
            feature_audit = benchmark["feature_audit"].copy()
            engineered_feature_catalog = build_engineered_feature_catalog(
                benchmark["engineered_feature_names"]
            )

            run_summary = pd.DataFrame(
                [
                    ["sample_size", benchmark["sample_rows"]],
                    ["train_rows", benchmark["train_rows"]],
                    ["validation_rows", benchmark["validation_rows"]],
                    ["missing_threshold", benchmark["missing_threshold"]],
                    ["tree_ensemble_auc", benchmark["ensemble_roc_auc"]],
                ],
                columns=["item", "value"],
            )

            core_metrics = comparison[
                ["model", "family", "roc_auc", "average_precision", "precision_at_top_1pct", "recall_at_top_1pct"]
            ].copy()
            core_metrics = core_metrics.rename(
                columns={
                    "roc_auc": "ROC-AUC",
                    "average_precision": "PR-AUC",
                    "precision_at_top_1pct": "Precision@Top1%",
                    "recall_at_top_1pct": "Recall@Top1%",
                }
            )

            display(run_summary.style.format({"value": "{:,.4f}"}).hide(axis="index"))
            display(core_metrics.style.format({
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
                "Precision@Top1%": "{:.2%}",
                "Recall@Top1%": "{:.2%}",
            }).hide(axis="index"))
            """
        ),
        markdown_cell("## Feature Audit"),
        code_cell(
            """
            display(feature_audit.style.hide(axis="index"))
            display(engineered_feature_catalog.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Model Comparison"),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(
                data=comparison,
                x="roc_auc",
                y="model",
                hue="family",
                dodge=False,
                palette={
                    "linear_baseline": PLOT_COLORS["gold"],
                    "tree_model": PLOT_COLORS["navy"],
                    "ensemble": PLOT_COLORS["coral"],
                },
                ax=axes[0],
            )
            axes[0].set_title("ROC-AUC comparison")
            axes[0].set_xlabel("roc_auc")
            axes[0].set_ylabel("")
            axes[0].set_xlim(max(0.5, comparison["roc_auc"].min() - 0.03), 1.0)
            axes[0].legend(title="")

            metric_plot = comparison.melt(
                id_vars=["model", "family"],
                value_vars=["roc_auc", "average_precision", "precision_at_top_5pct", "recall_at_top_5pct"],
                var_name="metric",
                value_name="value",
            )
            sns.lineplot(
                data=metric_plot,
                x="metric",
                y="value",
                hue="model",
                style="model",
                markers=True,
                dashes=False,
                linewidth=2,
                ax=axes[1],
            )
            axes[1].set_title("Ranking and top-risk comparison")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("score")
            axes[1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Curves"),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for model_name in comparison["model"]:
                frame_pair = curve_frames[model_name]
                axes[0].plot(frame_pair["roc"]["fpr"], frame_pair["roc"]["tpr"], linewidth=2, label=model_name)
                axes[1].plot(frame_pair["pr"]["recall"], frame_pair["pr"]["precision"], linewidth=2, label=model_name)

            axes[0].plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1)
            axes[0].set_title("ROC curves")
            axes[0].set_xlabel("false positive rate")
            axes[0].set_ylabel("true positive rate")
            axes[0].legend()

            axes[1].set_title("PR curves")
            axes[1].set_xlabel("recall")
            axes[1].set_ylabel("precision")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Threshold Sweep"),
        code_cell(
            """
            threshold_rows = []
            for model_name in comparison["model"]:
                frame = threshold_tables[model_name].copy()
                frame["model"] = model_name
                threshold_rows.append(frame[["model", "threshold", "precision", "recall", "f1"]])

            threshold_plot = pd.concat(threshold_rows, ignore_index=True)
            selected_models = comparison["model"].head(min(3, len(comparison))).tolist()

            fig, axes = plt.subplots(len(selected_models), 1, figsize=(14, 4 * len(selected_models)), sharex=True)
            if len(selected_models) == 1:
                axes = [axes]

            for ax, model_name in zip(axes, selected_models):
                model_frame = threshold_plot.loc[threshold_plot["model"] == model_name].melt(
                    id_vars=["model", "threshold"],
                    value_vars=["precision", "recall", "f1"],
                    var_name="metric",
                    value_name="value",
                )
                sns.lineplot(
                    data=model_frame,
                    x="threshold",
                    y="value",
                    hue="metric",
                    palette=[PLOT_COLORS["navy"], PLOT_COLORS["coral"], PLOT_COLORS["teal"]],
                    linewidth=2,
                    ax=ax,
                )
                best_f1_row = threshold_tables[model_name].sort_values("f1", ascending=False).iloc[0]
                ax.axvline(best_f1_row["threshold"], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1)
                ax.set_title(f"{model_name}: threshold sweep")
                ax.set_ylabel("metric")
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.legend(title="")

            axes[-1].set_xlabel("threshold")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Score Distribution"),
        code_cell(
            """
            display_models = comparison["model"].head(min(3, len(comparison))).tolist()
            fig, axes = plt.subplots(len(display_models), 1, figsize=(14, 4 * len(display_models)), sharex=True)
            if len(display_models) == 1:
                axes = [axes]

            for ax, model_name in zip(axes, display_models):
                score_frame = pd.DataFrame(
                    {
                        "score": prediction_frame[model_name],
                        "Class": np.where(y_valid == 1, "fraud", "legit"),
                    }
                )
                sns.histplot(
                    data=score_frame,
                    x="score",
                    hue="Class",
                    bins=50,
                    stat="density",
                    common_norm=False,
                    element="step",
                    fill=False,
                    linewidth=1.6,
                    palette=[PLOT_COLORS["navy"], PLOT_COLORS["coral"]],
                    ax=ax,
                )
                ax.set_title(f"{model_name}: validation score distribution")
                ax.set_xlabel("predicted fraud probability")
                ax.set_ylabel("density")

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Feature Importance"),
        code_cell(
            """
            for model_name, importance_frame in benchmark["feature_importances"].items():
                if importance_frame.empty:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=importance_frame.sort_values("importance", ascending=True),
                    x="importance",
                    y="feature",
                    color=PLOT_COLORS["blue"],
                    ax=ax,
                )
                ax.set_title(f"{model_name}: top feature importances")
                ax.set_xlabel("importance")
                ax.set_ylabel("")
                plt.tight_layout()
                plt.show()

                display(importance_frame)
            """
        ),
        markdown_cell(
            """
            ## What Changed From The Baseline

            Compared with `02_baseline_logistic_regression.ipynb`, this notebook becomes more competition-oriented in four ways:

            - the validation split respects **time order**
            - we compare **multiple tree families**, not just one sklearn model
            - we add **UID-style and missingness features** inspired by the 1st-place writeup
            - we keep the baseline-style diagnostics: metric tables, curves, threshold sweeps, and score distributions

            If you want to inspect the feature engineering itself in more detail, open the companion notebook `05_feature_engineering_for_tree_models.ipynb`.
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
