# Model comparison

## Validation metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.7473 | 0.5157 | 0.7888 | 0.6237 | 0.8361 |
| random_forest | 0.7878 | 0.5964 | 0.6203 | 0.6081 | 0.8279 |

Both models were compared using the validation split. The test split was
not used for model or hyperparameter selection.

## Selected model and test metrics

Selected model: **logistic_regression**.

The model was selected because it had the highest validation F1 (0.6237); Accuracy alone was not used as the selection criterion.

The following test metrics were calculated **once, only after model selection**, for **logistic_regression**:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.7374 | 0.5034 | 0.7834 | 0.6130 | 0.8415 |

## Precision/Recall business trade-off

Recall measures how many customers who actually churned were identified. Higher recall can help a retention team avoid missing at-risk customers, but it also creates more false positives and potentially unnecessary outreach.
Precision measures how many customers flagged as churn-risk actually churned. Higher precision can reduce wasted intervention effort, but a precision-first threshold may miss customers who would have benefited from retention support.
This run prioritizes the balance represented by validation F1; logistic_regression achieved validation recall 0.7888 and precision 0.5157.

## Confusion matrices

- [logistic_regression confusion matrix](./logistic_regression/logistic_regression_confusion_matrix.png)
- [random_forest confusion matrix](./random_forest/random_forest_confusion_matrix.png)

The machine-readable source for these values is [model_comparison.json](./model_comparison.json).
