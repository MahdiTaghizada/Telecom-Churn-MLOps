# Dataset inspection

## Source and identification

- Source: [IBM Telco Customer Churn sample dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)
- License/provenance: IBM sample data for an illustrative, fictional telecom company. The source repository does not provide a clear license file alongside this CSV, so the raw file should not be redistributed without confirming IBM's applicable terms.
- Download timestamp and SHA256: `data/raw/dataset_metadata.json`
- Raw file: `data/raw/Telco-Customer-Churn.csv`

## Observed structure

The downloaded file contains 7,043 rows and 21 columns. `Churn` is the binary
target (`Yes`/`No`) and `customerID` is an identifier excluded from model
features. The model uses 19 feature columns: four numeric columns and fifteen
categorical columns.

The observed target balance is:

| Churn | Rows |
|---|---:|
| No | 5,174 |
| Yes | 1,869 |

## Observed data issues and handling

- `TotalCharges` is stored as an object/string column even though it represents
  a numeric amount. Eleven rows contain a whitespace-only value. The
  preprocessing step converts it with numeric coercion, turning whitespace
  into missing values, then fits a median imputer on training data only.
- The downloaded file has no pandas-recognized null values and no duplicate
  rows. This is recorded rather than assumed; validation still rejects missing
  identifiers/targets and unexpected target values.
- All required columns and target values are validated before training. A
  missing required column, empty dataset, invalid target, or missing target or
  identifier raises a clear error.
- `customerID` is retained for traceability but never passed to the model.
- Categorical values are one-hot encoded with `handle_unknown="ignore"`, so a
  previously unseen category at inference does not crash the pipeline.

The complete observed dtypes, categorical values, missing-value counts, target
balance, source URL, download timestamp, and SHA256 are stored in the generated
metadata JSON. Re-run `uv run python -m telecom_churn.train` to refresh the
download/inspection metadata when the raw file is absent.
