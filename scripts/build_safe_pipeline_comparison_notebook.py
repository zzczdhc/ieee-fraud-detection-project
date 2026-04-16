from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "06_safe_pipeline_comparison.ipynb"


def markdown_cell(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code_cell(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    cells = [
        markdown_cell(
            """
            # 06 Safe Pipeline Comparison

            This notebook evaluates the new `run_data_check_safe` / `data_prep_safe` / `eda_safe` workflow against the existing project pipelines.

            It answers four questions:

            1. What does the new safe preprocessing actually change?
            2. How is it different from the baseline notebook?
            3. How does it differ from `03_tree_models.ipynb`?
            4. Does the safer preprocessing improve model quality, robustness, or both?
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
            import src.fraud_detection.safe_benchmark as safe_benchmark_module

            importlib.reload(eda_module)
            importlib.reload(safe_benchmark_module)

            from src.fraud_detection.eda import PLOT_COLORS, set_plot_theme
            from src.fraud_detection.safe_benchmark import fit_cross_pipeline_comparison

            warnings.filterwarnings("ignore")
            set_plot_theme()
            pd.set_option("display.max_columns", 200)
            pd.set_option("display.float_format", "{:,.4f}".format)

            SAMPLE_SIZE = 80_000
            RANDOM_STATE = 42
            """
        ),
        code_cell(
            """
            comparison_bundle = fit_cross_pipeline_comparison(
                sample_size=SAMPLE_SIZE,
                random_state=RANDOM_STATE,
            )

            comparison = comparison_bundle["comparison"].copy()
            method_summary = comparison_bundle["method_summary"].copy()
            strengths_gaps = comparison_bundle["strengths_gaps"].copy()
            safe_audit = comparison_bundle["safe_benchmark"]["preprocessing_audit"].copy()

            display(method_summary.style.hide(axis="index"))
            display(safe_audit.style.hide(axis="index"))
            display(
                comparison[
                    ["model", "family", "roc_auc", "average_precision", "precision_at_top_1pct", "recall_at_top_1pct"]
                ]
                .style
                .format(
                    {
                        "roc_auc": "{:.4f}",
                        "average_precision": "{:.4f}",
                        "precision_at_top_1pct": "{:.2%}",
                        "recall_at_top_1pct": "{:.2%}",
                    }
                )
                .hide(axis="index")
            )
            """
        ),
        markdown_cell("## Where The Safe Pipeline Helps"),
        code_cell(
            """
            display(strengths_gaps.style.hide(axis="index"))
            """
        ),
        markdown_cell("## Performance Comparison"),
        code_cell(
            """
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            sns.barplot(
                data=comparison,
                x="roc_auc",
                y="model",
                hue="family",
                dodge=False,
                ax=axes[0],
            )
            axes[0].set_title("ROC-AUC across project pipelines")
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
            axes[1].set_title("Ranking and top-risk metrics")
            axes[1].set_xlabel("")
            axes[1].set_ylabel("score")
            axes[1].tick_params(axis="x", rotation=20)

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell("## Safe Pipeline Curves"),
        code_cell(
            """
            safe_curves = comparison_bundle["safe_benchmark"]["curve_frames"]
            safe_models = comparison_bundle["safe_benchmark"]["leaderboard"]["model"].tolist()

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            for model_name in safe_models:
                frame_pair = safe_curves[model_name]
                axes[0].plot(frame_pair["roc"]["fpr"], frame_pair["roc"]["tpr"], linewidth=2, label=model_name)
                axes[1].plot(frame_pair["pr"]["recall"], frame_pair["pr"]["precision"], linewidth=2, label=model_name)

            axes[0].plot([0, 1], [0, 1], linestyle="--", color=PLOT_COLORS["muted"], linewidth=1)
            axes[0].set_title("Safe pipeline ROC curves")
            axes[0].set_xlabel("false positive rate")
            axes[0].set_ylabel("true positive rate")
            axes[0].legend()

            axes[1].set_title("Safe pipeline PR curves")
            axes[1].set_xlabel("recall")
            axes[1].set_ylabel("precision")
            axes[1].legend()

            plt.tight_layout()
            plt.show()
            """
        ),
        markdown_cell(
            """
            ## Reading The Results

            A practical interpretation guide:

            - If `safe_pipeline` beats the original baseline logistic model, then the extra data checks and categorical handling are helping in a real measurable way.
            - If `03_tree_models` still wins overall, that usually means the competition-style engineered features are more valuable than safer generic preprocessing alone.
            - If the safe pipeline is slightly worse but more stable to explain, that can still be useful for the report because it shows a cleaner data engineering workflow.
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
