# Methodology (baseline draft)

## Data inputs
- `bank_quarterly.csv`: district-quarter banking observations and quality/event flags.
- `district_graph.csv`: district-neighbour links and distances.
- `district_lineage.csv`: district lineage and bank-level shift metadata.
- `sample_submission.csv`: required output IDs and quantile columns.

## Current baseline assumption — MUST VERIFY
The starter code assumes the forecast target is the district credit-to-deposit ratio:

`target = credit / deposit * 100`

This assumption is inferred from the available columns, not confirmed by the official competition statement. Do not present it as the official target until verified against the brief. The current starter model does not yet incorporate graph or lineage features.

## Features
- Banking variables: offices, deposit, credit, RRB adjustment, and data/event flags.
- District-specific lagged target values (1, 2, 3, 4, and 8 quarters).
- Trailing four-quarter mean and standard deviation, calculated using prior observations.
- Calendar features derived from the forecast quarter.

## Model
Three separate gradient-boosting regressors are fitted with quantile loss for the 10th, 50th, and 90th percentiles. Predictions are clipped to non-negative values and sorted to avoid quantile crossing.

## Validation and limitations
`validation/validate_submission.py` checks submission structure, IDs/order, finite predictions, and quantile ordering. This is a format check, not a model-quality evaluation. A time-based backtest and the official evaluation metric must be implemented after the target and metric are confirmed.

## Reproducibility
A fixed random seed is used. Keep competition data outside the Git repository unless the competition rules explicitly permit redistribution.
