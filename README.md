# SQFRRD Business Case Competition

A reproducible baseline project for the SQFRRD business case competition.

## Repository contents
- `src/forecast.py` — baseline feature engineering, quantile models, and submission generation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — excludes local data, notebook checkpoints, and generated artifacts.

## Data files
Keep the competition-provided CSV files in a local `data/` directory (or use the Kaggle input directory inside Kaggle):
- `bank_quarterly.csv`
- `district_graph.csv`
- `district_lineage.csv`
- `sample_submission.csv`

The data files are intentionally not committed to this repository. Follow the competition rules before sharing any supplied data.

## Baseline approach
The script derives the district credit-to-deposit ratio (`credit / deposit * 100`), creates district-level lag features and calendar features, and fits separate quantile gradient-boosting models for the 10th, 50th, and 90th percentiles. It then writes predictions in the exact `id,p10,p50,p90` layout of the sample submission.

**Important:** This is a baseline, not a claim of leaderboard performance. Before final submission, verify the competition's target definition, forecast horizon interpretation, allowed data, and evaluation metric against the official problem statement. If the target is not the credit-to-deposit ratio, update the target construction accordingly.

## Run locally
```bash
pip install -r requirements.txt
python src/forecast.py   --bank data/bank_quarterly.csv   --sample data/sample_submission.csv   --output submission.csv
```

## Kaggle
Upload/open the notebook in the competition's Kaggle environment, read the input CSVs from `/kaggle/input/...`, and save the final file to `/kaggle/working/submission.csv`. Check that row order and IDs match `sample_submission.csv` exactly.

## Reproducibility
- Fixed random seed.
- No external network access required.
- Generated predictions are clipped to non-negative values and quantiles are sorted to prevent crossing.
