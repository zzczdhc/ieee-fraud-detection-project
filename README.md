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
├── scripts/
│   └── train_baseline.py
└── src/
    └── fraud_detection/
        ├── config.py
        ├── data.py
        ├── features.py
        ├── metrics.py
        └── train.py
```

## Dataset

The project is based on the IEEE-CIS Fraud Detection dataset from Kaggle:
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)

Raw data files are not tracked in this repository. See [data/README.md](data/README.md) for expected file placement.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Data

If you use the Kaggle CLI, the raw files can be downloaded into `data/raw/` like this:

```bash
mkdir -p data/raw
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

If you download the dataset manually from Kaggle, place the CSV files under `data/raw/`.

## Planned Workflow

1. Exploratory data analysis to understand feature distributions, missingness, and target imbalance.
2. Baseline logistic regression with standard preprocessing.
3. Tree-based models such as Random Forest, XGBoost, or LightGBM.
4. Sampling strategies and evaluation focused on ROC-AUC, PR-AUC, recall, and precision.

## Quick Start

```bash
jupyter notebook
```

## Run The Baseline Model

The repository now includes a starter training pipeline for a logistic regression baseline with:

- transaction and identity table merge
- high-missing-column filtering
- numeric median imputation
- categorical mode imputation plus one-hot encoding
- stratified train/validation split
- imbalance-aware training with `class_weight="balanced"`
- JSON metric export

Example command:

```bash
python -m src.fraud_detection.train --sample-size 50000
```

Or equivalently:

```bash
python scripts/train_baseline.py --sample-size 50000
```

This writes metrics to `outputs/baseline_metrics.json`.

## Notebook Guide

- `01_eda.ipynb`: missingness, class imbalance, and feature inspection
- `02_baseline_logistic_regression.ipynb`: reproduce and analyze the baseline pipeline
- `03_tree_models.ipynb`: compare tree-based models
- `04_sampling_and_evaluation.ipynb`: test resampling and threshold strategies

## Suggested Team Questions

- How severe is the fraud class imbalance, and how does it affect metric choice?
- Which families of features have the highest missingness?
- Does sampling improve recall without collapsing precision?
- Do tree-based models outperform a well-preprocessed linear baseline?

## Notes

- Keep large raw datasets out of git.
- Keep reusable logic in `src/` instead of duplicating code across notebooks.
- Add experiment results and conclusions to the notebooks or a future report.
- Update `requirements.txt` if the project stack changes.
