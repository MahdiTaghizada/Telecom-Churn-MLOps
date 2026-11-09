# Telecom Churn Model Card

## Intended use

This model estimates the probability that a customer in the IBM Telco Customer
Churn sample will churn. It is intended for portfolio demonstration, model
evaluation, and decision-support prototyping.

The model is decision support, not an automatic business decision. A human
reviewer must consider context before any retention action.

## Training data and features

The source is IBM's public Telco Customer Churn sample dataset. The dataset is
synthetic company data, not internal operator data or real customer records.
The target is `Churn` (`Yes`/`No`). Customer identifiers are excluded. The
feature set contains tenure, charges, service attributes, contract, payment
method, and related customer profile fields.

## Model and measured performance

The selected model is the Logistic Regression pipeline, selected by validation
F1. The recorded held-out test metrics are:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 0.7374 | 0.5034 | 0.7834 | 0.6130 | 0.8415 |

These values are from `reports/model_comparison.json` and are not guarantees
for another dataset or production population. The registered model is
`telecom-churn-model`, version `6`, with the `champion` alias in the recorded
MLflow run.

Precision is approximately `0.50`: among customers flagged as high risk,
about half do not actually churn, which represents false warnings. Recall is
approximately `0.78`: about 78% of customers who actually churn are detected.
For a retention campaign, higher recall can reduce missed at-risk customers,
but it also increases false positives, unnecessary outreach, and campaign
cost. The appropriate balance depends on the business threshold and must be
validated with labeled operational data.

## Out-of-scope use

- Do not use the model to make automatic decisions for real customers.
- Do not use it on another operator's data without retraining and evaluating it
  on that operator's appropriately governed data.

## Limitations and bias risks

- The sample is synthetic and may not represent a real operator or current
  customer population.
- Performance can vary across customer groups and operating conditions.
- Demographic and service-related variables may encode historical or sampling
  bias; fairness was not comprehensively evaluated.
- The probability is not a causal explanation and does not establish that a
  customer will churn.
- Data drift detection is not concept-drift detection and does not prove that
  model quality has declined.

## Monitoring

The API exposes Prometheus-format request, error, prediction, and latency
metrics. Drift reports compare the feature-only training split with control or
explicitly synthetic current data. New labeled data requires a separate model
evaluation before quality conclusions are made.
