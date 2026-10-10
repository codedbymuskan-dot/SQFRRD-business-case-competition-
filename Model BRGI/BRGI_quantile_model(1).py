"""
Baseline quantile forecasting script.
Target assumption: district credit-to-deposit ratio = credit / deposit * 100.
Verify this assumption against the official competition brief before use.
"""
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

SEED = 42
QUANTILES = (0.10, 0.50, 0.90)

def quarter_index(q):
    m = re.fullmatch(r"(\d{4})Q([1-4])", str(q))
    if not m:
        return np.nan
    return int(m.group(1)) * 4 + int(m.group(2)) - 1

def build_history(bank):
    df = bank.copy()
    df["quarter_idx"] = df["quarter"].map(quarter_index)
    df["target"] = np.where(df["deposit"] > 0, df["credit"] / df["deposit"] * 100, np.nan)
    df = df.sort_values(["district_id", "quarter_idx"])
    # Treat explicitly nulled/glitch rows as missing observations for target training.
    if "glitch_nulled" in df:
        df.loc[df["glitch_nulled"].fillna(0).astype(bool), "target"] = np.nan
    for lag in (1, 2, 3, 4, 8):
        df[f"lag_{lag}"] = df.groupby("district_id")["target"].shift(lag)
    df["rolling_mean_4"] = df.groupby("district_id")["target"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).mean()
    )
    df["rolling_std_4"] = df.groupby("district_id")["target"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).std()
    )
    df["year"] = df["quarter_idx"] // 4
    df["quarter_of_year"] = df["quarter_idx"] % 4
    return df

FEATURES = [
    "offices", "deposit", "credit", "rrb_adjusted", "bank_gap",
    "glitch_nulled", "hdfc_merger_window", "lag_1", "lag_2", "lag_3",
    "lag_4", "lag_8", "rolling_mean_4", "rolling_std_4", "year",
    "quarter_of_year"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()

    bank = pd.read_csv(args.bank)
    sample = pd.read_csv(args.sample)
    hist = build_history(bank)
    train = hist.dropna(subset=["target"]).copy()

    # Learn one-step-ahead relationships from rows whose lag features are known.
    # Features include the current row's banking variables and prior target values.
    X = train[FEATURES].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0)
    X = X.fillna(medians)
    y = train["target"].clip(lower=0)

    # Build prediction features for every district in the sample using the latest
    # available row as the base. The sample IDs define the required output order.
    latest = hist.sort_values("quarter_idx").groupby("district_id", as_index=False).tail(1)
    latest = latest.set_index("district_id", drop=False)
    rows = []
    for sid in sample["id"].astype(str):
        # Expected ID pattern: DISTRICT_YYYYQn_hN
        m = re.fullmatch(r"(.+?)_(\d{4}Q[1-4])_h(\d+)", sid)
        if not m:
            raise ValueError(f"Unrecognized sample id: {sid}")
        district, forecast_quarter, horizon = m.group(1), m.group(2), int(m.group(3))
        if district in latest.index:
            row = latest.loc[district].copy()
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1].copy()
        else:
            # Some submission IDs may refer to districts absent from the history.
            # Use median latest-district features as a transparent fallback.
            row = latest[FEATURES].median(numeric_only=True)
        row["year"] = quarter_index(forecast_quarter) // 4
        row["quarter_of_year"] = quarter_index(forecast_quarter) % 4
        # Horizon is included as a feature by adjusting the time index; for this
        # simple baseline, the same latest available banking covariates are used.
        row["year"] = row["year"] + (horizon - 1) // 4
        rows.append({**{f: row.get(f, np.nan) for f in FEATURES}, "id": sid})

    pred_frame = pd.DataFrame(rows)
    XP = pred_frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(medians)
    out = sample[["id"]].copy()
    for q in QUANTILES:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=180, max_depth=2,
            learning_rate=0.035, random_state=SEED
        )
        model.fit(X, y)
        out[f"p{int(q*100)}"] = np.maximum(0, model.predict(XP))

    # Ensure valid, non-crossing quantiles.
    vals = np.sort(out[["p10", "p50", "p90"]].to_numpy(), axis=1)
    out[["p10", "p50", "p90"]] = vals
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} with {len(out)} rows.")
    print(out.head())

if __name__ == "__main__":
    main()
