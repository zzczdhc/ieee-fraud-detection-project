# IEEE-CIS Fraud Detection Project

This repository contains our Applied Machine Learning group project on fraud detection using the IEEE-CIS Fraud Detection dataset. We compare several machine learning approaches and study practical challenges such as severe class imbalance, missing data, and model evaluation under imbalanced settings.

## Project Goals

- Build a strong baseline for fraud detection.
- Compare linear models, tree-based models, and imbalance-aware methods.
- Analyze the impact of missing values and feature preprocessing.
- Evaluate models with metrics that are meaningful for imbalanced classification.

## Repository Structure

```text
ieee-fraud-detection-project/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_logistic_regression.ipynb
│   ├── 03_tree_models.ipynb
│   └── 04_sampling_and_evaluation.ipynb
```

## Dataset

The project is based on the IEEE-CIS Fraud Detection dataset from Kaggle:
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

Raw data files are not tracked in this repository. See [data/README.md](data/README.md) for expected file placement.

## Planned Workflow

1. Exploratory data analysis to understand feature distributions, missingness, and target imbalance.
2. Baseline logistic regression with standard preprocessing.
3. Tree-based models such as Random Forest, XGBoost, or LightGBM.
4. Sampling strategies and evaluation focused on ROC-AUC, PR-AUC, recall, and precision.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Notes

- Keep large raw datasets out of git.
- Add experiment results and conclusions to the notebooks or a future report.
- Update `requirements.txt` if the project stack changes.
