from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_detection.eda_safe import run_safe_eda_pipeline


def main():
    print("Running safe data checks...")
    train_df, test_df, artifacts = run_safe_eda_pipeline(nrows=50000)

    print("Done.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Dropped columns: {len(artifacts.drop_cols)}")
    print(f"Missingness indicator columns: {len(artifacts.missing_indicator_cols)}")
    print(f"High-cardinality categorical columns: {len(artifacts.high_cardinality_cols)}")


if __name__ == "__main__":
    main()