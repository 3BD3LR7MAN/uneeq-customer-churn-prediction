# Customer Churn Prediction — Model Report

## Executive summary

This project predicts which Telco customers are at risk of churn so a retention team can prioritize outreach. The final model is the **Logistic Regression selected reference model**, selected as the simpler reference model because it achieved the stronger validation PR-AUC compared with the tuned Random Forest. The operating threshold was selected on the validation split to target recall of **70%**, then evaluated once on an unseen test split.

## Data and preprocessing

- Dataset: IBM Telco Customer Churn educational dataset
- Rows used: **7,043**
- Source columns: **21**
- Churned customers: **1,869 (26.5%)**
- Split: **60% train / 20% validation / 20% test**, stratified and seeded with 42
- `customerID` excluded from modeling
- `TotalCharges` converted to numeric; **11** blank values were replaced with 0 because those records have `tenure = 0`
- Numeric imputation/scaling and categorical one-hot encoding are fitted inside the pipeline

## Model comparison

| Model | Validation PR-AUC |
|---|---:|
| Logistic Regression (selected reference model) | 0.678 |
| Tuned Random Forest comparison | 0.673 |

**Interpretation:** The tuned Random Forest did not improve validation PR-AUC, so the Logistic Regression selected reference model was retained because it was slightly stronger and simpler.

## Threshold and test results

The selected validation threshold is **0.6477**. Validation recall was **70.3%**, meeting the 70% target on validation. On the unseen test set, recall was **66.0%**, so the target did **not** generalize fully to test data.

| Metric | Test result |
|---|---:|
| Precision | 0.559 |
| Recall | 0.660 |
| F1 | 0.605 |
| PR-AUC / Average Precision | 0.624 |
| ROC-AUC | 0.832 |

Confusion matrix, with rows as actual and columns as predicted:

```text
[[840 195]
 [127 247]]
```

The threshold flags **442** of 1,409 test customers, correctly identifies **247** churners, and misses **127** churners. A business should attach a retention action and cost to each flag before using this in production.

## Threshold trade-off on validation

The table below shows the precision/recall trade-off at several candidate thresholds. The selected threshold is chosen on validation to prioritize the stated recall goal; the test set remains reserved for final evaluation.

| Threshold | Precision | Recall | F1 | Flagged customers |
|---:|---:|---:|---:|---:|
| 0.5000 | 0.518 | 0.821 | 0.635 | 593 |
| 0.6000 | 0.564 | 0.738 | 0.640 | 489 |
| 0.6477 | 0.594 | 0.703 | 0.644 | 443 |
| 0.7000 | 0.637 | 0.639 | 0.638 | 375 |

## Explainability and EDA

Permutation importance on validation identifies the features most associated with predictive performance. The top features are:

- tenure: 0.271
- InternetService: 0.075
- Contract: 0.066
- TotalCharges: 0.057
- MonthlyCharges: 0.041

Additional EDA charts cover churn rate by contract, internet service, payment method, and tenure band. Importance is predictive rather than causal.

## Limitations and responsible use

This is an educational model on a public dataset. The test recall is below the validation target, so the operating point requires further calibration on current company data. Monitor drift, subgroup performance, intervention cost, and fairness before production use. Do not use customer identifiers or predictions as automatic grounds for adverse decisions.
