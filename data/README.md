# Data Directory

Place the IEEE-CIS Fraud Detection dataset files in this directory after downloading them from Kaggle.

Expected files typically include:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

Recommended layout:

```text
data/
├── README.md
└── raw/
    ├── train_transaction.csv
    ├── train_identity.csv
    ├── test_transaction.csv
    ├── test_identity.csv
    └── sample_submission.csv
```

Do not commit raw dataset files to git.
