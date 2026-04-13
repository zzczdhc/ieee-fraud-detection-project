from __future__ import annotations

from .data_prep_safe import (
    load_merged_data_safe,
    fit_safe_preprocessing,
    save_safe_reports,
)


def run_safe_eda_pipeline(nrows=None):
    train_df, test_df = load_merged_data_safe(nrows=nrows)
    artifacts = fit_safe_preprocessing(train_df)
    save_safe_reports(train_df, test_df, artifacts)
    return train_df, test_df, artifacts