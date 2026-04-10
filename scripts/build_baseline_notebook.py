from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "02_baseline_logistic_regression.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # 02 Baseline Logistic Regression

            This notebook trains a first-pass fraud classifier using the reusable pipeline in `src/` and evaluates it with metrics that are actually useful for an imbalanced fraud problem.

            The main goals here are:

            - establish a transparent baseline before trying stronger models
            - inspect `ROC` and `PR` behavior instead of relying on accuracy
            - compare operating thresholds rather than assuming `0.5` is the right cutoff
            - measure how much fraud we can capture in the top-risk slice of transactions
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import sys
            import warnings

            import numpy as np
            import pandas as pd
            import seaborn as sns
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            from IPython.display import HTML, display
            from sklearn.metrics import confusion_matrix

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent

            if str(PROJECT_ROOT) not in sys.path:
                sys.path.append(str(PROJECT_ROOT))

            from src.fraud_detection.eda import PLOT_COLORS, metric_cards_html, set_plot_theme
            from src.fraud_detection.metrics import build_curve_frames
            from src.fraud_detection.train import fit_baseline_experiment

            warnings.filterwarnings("ignore")
            set_plot_theme()
            pd.set_option("display.max_columns", 120)
            sns.set_context("talk")

            SAMPLE_SIZE = 50_000
            MISSING_THRESHOLD = 0.95
            RANDOM_STATE = 42
            """
        ),
        code_cell(
            """
            experiment = fit_baseline_experiment(
                sample_size=SAMPLE_SIZE,
                missing_threshold=MISSING_THRESHOLD,
                random_state=RANDOM_STATE,
            )

            metrics = experiment["metrics"]
            threshold_table = experiment["threshold_table"].copy()
            curve_frames = build_curve_frames(experiment["y_valid"], experiment["validation_scores"])
            best_f1_row = threshold_table.sort_values("f1", ascending=False).iloc[0]

            y_valid = np.asarray(experiment["y_valid"])
            validation_scores = np.asarray(experiment["validation_scores"])
            y_pred_default = (validation_scores >= 0.5).astype(int)
            cm = confusion_matrix(y_valid, y_pred_default, labels=[0, 1], normalize="true")

            score_frame = pd.DataFrame(
                {
                    "score": validation_scores,
                    "Class": np.where(y_valid == 1, "Fraud", "Legitimate"),
                }
            )

            run_summary = pd.DataFrame(
                [
                    {"item": "Sample size", "value": f"{SAMPLE_SIZE:,} rows"},
                    {"item": "Train rows", "value": f"{len(experiment['x_train']):,}"},
                    {"item": "Validation rows", "value": f"{len(experiment['x_valid']):,}"},
                    {"item": "Dropped high-missing columns", "value": f"{len(experiment['dropped_high_missing_columns']):,}"},
                    {"item": "Numeric features", "value": f"{len(experiment['numeric_features']):,}"},
                    {"item": "Categorical features", "value": f"{len(experiment['categorical_features']):,}"},
                ]
            )

            top_risk_table = pd.DataFrame(
                [
                    {
                        "slice": "Top 1% scores",
                        "precision": metrics["precision_at_top_1pct"],
                        "recall": metrics["recall_at_top_1pct"],
                        "lift": metrics["lift_at_top_1pct"],
                    },
                    {
                        "slice": "Top 5% scores",
                        "precision": metrics["precision_at_top_5pct"],
                        "recall": metrics["recall_at_top_5pct"],
                        "lift": metrics["lift_at_top_5pct"],
                    },
                ]
            )

            print("Baseline fit complete.")
            print(f"Validation ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"Validation PR-AUC (Average Precision): {metrics['average_precision']:.4f}")
            """
        ),
        markdown_cell(
            """
            ## Baseline Snapshot

            For an imbalanced fraud problem, a baseline is only useful if it answers two questions:

            1. **How well does the model rank risky transactions overall?**
            2. **How much fraud can we capture when we only act on the highest-risk slice?**

            That is why this notebook emphasizes `PR-AUC`, `ROC-AUC`, threshold tradeoffs, and top-risk capture metrics.
            """
        ),
        code_cell(
            """
            cards = [
                {
                    "title": "ROC-AUC",
                    "value": f"{metrics['roc_auc']:.3f}",
                    "subtitle": "overall ranking quality",
                },
                {
                    "title": "PR-AUC",
                    "value": f"{metrics['average_precision']:.3f}",
                    "subtitle": "average precision on an imbalanced target",
                },
                {
                    "title": "Recall @ 0.5",
                    "value": f"{metrics['recall']:.1%}",
                    "subtitle": "default threshold sensitivity",
                },
                {
                    "title": "Recall @ Top 5%",
                    "value": f"{metrics['recall_at_top_5pct']:.1%}",
                    "subtitle": "fraud captured in highest-risk 5%",
                },
                {
                    "title": "Precision @ Top 1%",
                    "value": f"{metrics['precision_at_top_1pct']:.1%}",
                    "subtitle": "purity of the top-ranked slice",
                },
                {
                    "title": "Best F1 Threshold",
                    "value": f"{best_f1_row['threshold']:.2f}",
                    "subtitle": "best threshold inside the notebook sweep",
                },
            ]

            display(HTML(metric_cards_html(cards)))

            display(run_summary.style.hide(axis="index"))

            display(
                top_risk_table.style
                .format({"precision": "{:.2%}", "recall": "{:.2%}", "lift": "{:.2f}x"})
                .hide(axis="index")
            )
            """
        ),
        markdown_cell(
            """
            ## Core Curves And Threshold Tradeoffs

            `ROC` can look optimistic on imbalanced data, so we keep it but do not treat it as the only decision metric. `PR` is more revealing because it directly exposes the precision-recall tradeoff on the fraud class.
            """
        ),
        code_cell(
            """
            threshold_plot = threshold_table.melt(
                id_vars="threshold",
                value_vars=["precision", "recall", "f1", "balanced_accuracy"],
                var_name="metric",
                value_name="value",
            )

            fig, axes = plt.subplots(2, 2, figsize=(18, 12))

            axes[0, 0].plot(curve_frames["roc"]["fpr"], curve_frames["roc"]["tpr"], color=PLOT_COLORS["navy"], linewidth=2.5)
            axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1)
            axes[0, 0].set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.3f})")
            axes[0, 0].set_xlabel("False positive rate")
            axes[0, 0].set_ylabel("True positive rate")

            axes[0, 1].plot(curve_frames["pr"]["recall"], curve_frames["pr"]["precision"], color=PLOT_COLORS["coral"], linewidth=2.5)
            axes[0, 1].axhline(metrics["base_fraud_rate"], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1, label="Fraud base rate")
            axes[0, 1].set_title(f"Precision-Recall Curve (AP = {metrics['average_precision']:.3f})")
            axes[0, 1].set_xlabel("Recall")
            axes[0, 1].set_ylabel("Precision")
            axes[0, 1].legend(loc="lower left")

            sns.lineplot(
                data=threshold_plot,
                x="threshold",
                y="value",
                hue="metric",
                palette=[PLOT_COLORS["navy"], PLOT_COLORS["coral"], PLOT_COLORS["teal"], PLOT_COLORS["gold"]],
                linewidth=2.3,
                ax=axes[1, 0],
            )
            axes[1, 0].axvline(best_f1_row["threshold"], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1)
            axes[1, 0].set_title("Threshold Sweep")
            axes[1, 0].set_xlabel("Decision threshold")
            axes[1, 0].set_ylabel("Metric value")
            axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))

            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                cbar=False,
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["True Legit", "True Fraud"],
                ax=axes[1, 1],
            )
            axes[1, 1].set_title("Normalized Confusion Matrix at Threshold = 0.50")
            axes[1, 1].set_xlabel("")
            axes[1, 1].set_ylabel("")

            plt.suptitle("Baseline evaluation views", fontsize=22, fontweight="bold", y=1.02)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Score Separation

            The score distributions below help us see whether the logistic model is meaningfully separating fraud from legitimate traffic, even if the two classes still overlap.
            """
        ),
        code_cell(
            """
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.histplot(
                data=score_frame,
                x="score",
                hue="Class",
                bins=50,
                stat="density",
                common_norm=False,
                element="step",
                fill=False,
                linewidth=2,
                palette=[PLOT_COLORS["navy"], PLOT_COLORS["coral"]],
                ax=ax,
            )
            ax.set_title("Validation Score Distribution by Class")
            ax.set_xlabel("Predicted fraud probability")
            ax.set_ylabel("Density")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Threshold Table

            The threshold sweep is useful because the default cutoff of `0.5` is rarely the best operating point for fraud detection. We usually care about a tradeoff between review capacity, recall, and alert precision.
            """
        ),
        code_cell(
            """
            display(
                threshold_table.style
                .format(
                    {
                        "threshold": "{:.2f}",
                        "roc_auc": "{:.3f}",
                        "average_precision": "{:.3f}",
                        "log_loss": "{:.3f}",
                        "brier_score": "{:.3f}",
                        "precision": "{:.2%}",
                        "recall": "{:.2%}",
                        "f1": "{:.2%}",
                        "balanced_accuracy": "{:.2%}",
                        "positive_prediction_rate": "{:.2%}",
                        "base_fraud_rate": "{:.2%}",
                    }
                )
                .hide(axis="index")
            )
            """
        ),
        markdown_cell(
            """
            ## Recommended Metric Set

            For the rest of this project, a good metric stack is:

            - **Primary ranking metric:** `PR-AUC` / average precision
            - **Secondary ranking metric:** `ROC-AUC`
            - **Threshold metrics:** precision, recall, F1, and balanced accuracy at a chosen cutoff
            - **Operational metrics:** recall and precision in the top-risk slice, such as top `1%` and top `5%`

            This gives us both a model-comparison view and an operations-facing view. It also sets up a clean transition to tree-based models, where we can reuse the exact same evaluation lenses.
            """
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13",
            },
        },
    )
    return notebook


def main() -> None:
    notebook = build_notebook()
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"Wrote notebook to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
