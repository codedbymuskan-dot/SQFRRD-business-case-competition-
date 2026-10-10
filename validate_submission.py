"""Validate a generated submission against the competition sample format."""
import argparse
import numpy as np
import pandas as pd

EXPECTED_COLUMNS = ["id", "p10", "p50", "p90"]


def validate(sample_path: str, submission_path: str) -> None:
    sample = pd.read_csv(sample_path)
    sub = pd.read_csv(submission_path)

    if list(sub.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Expected columns {EXPECTED_COLUMNS}, got {list(sub.columns)}")
    if len(sub) != len(sample):
        raise ValueError(f"Row count mismatch: sample={len(sample)}, submission={len(sub)}")
    if sub["id"].isna().any():
        raise ValueError("Submission contains missing IDs")
    if sub["id"].duplicated().any():
        raise ValueError("Submission contains duplicate IDs")
    if not sub["id"].astype(str).equals(sample["id"].astype(str)):
        raise ValueError("IDs or row order do not exactly match sample_submission.csv")

    values = sub[["p10", "p50", "p90"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Quantile predictions contain NaN or infinite values")
    if (values[:, 0] > values[:, 1]).any() or (values[:, 1] > values[:, 2]).any():
        raise ValueError("Quantiles cross: expected p10 <= p50 <= p90")

    print("PASS: submission schema, row count, IDs, finite values, and quantile ordering are valid.")
    print(f"Rows validated: {len(sub)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True, help="Path to sample_submission.csv")
    parser.add_argument("--submission", required=True, help="Path to generated submission.csv")
    args = parser.parse_args()
    validate(args.sample, args.submission)
