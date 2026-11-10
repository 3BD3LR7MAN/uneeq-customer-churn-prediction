# Customer Churn Prediction Pipeline

A reproducible machine-learning project for prioritizing Telco customers at risk of churn. It includes a Colab notebook, a local training CLI, a saved model, evaluation reports, explainability outputs, and Power BI-ready exports.

## Results

The final model is the Logistic Regression reference model, selected because its validation PR-AUC (0.678) slightly exceeded the tuned Random Forest comparison (0.673). It is called a reference model because it is the simpler benchmark that won model selection—not because it is an unfinished model. The threshold 0.6477 was selected on validation for a 70% recall target. On the unseen test set: precision 0.559, recall 0.660, F1 0.605, PR-AUC 0.624, and ROC-AUC 0.832. The validation target did not fully generalize to test, so this result requires further calibration before operational use.

Read the reviewer-facing report: [`reports/Model_Report_EN.md`](reports/Model_Report_EN.md). The Arabic report is available at [`reports/Model_Report_AR.md`](reports/Model_Report_AR.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_pipeline.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --output . \
  --iterations 12 \
  --cv 3 \
  --target-recall 0.70
```

To regenerate Power BI exports:

```bash
python src/export_powerbi_assets.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --model models/churn_pipeline.joblib \
  --output reports/powerbi
```

## Deliverables

- `notebooks/Customer_Churn_Pipeline_Colab.ipynb` — Colab walkthrough
- `src/train_pipeline.py` — reproducible training and evaluation
- `src/export_powerbi_assets.py` — holdout predictions for Power BI
- `models/churn_pipeline.joblib` — selected model and decision threshold
- `reports/Model_Report_EN.md` — English model report
- `reports/Model_Report_AR.md` — Arabic model report
- `reports/figures/` — class distribution, tenure, category, tenure-band, confusion-matrix, and importance charts
- `reports/threshold_tradeoff.csv` — validation operating-point comparison for threshold selection
- `DATA_SOURCES.md` — dataset citation and redistribution guidance
- `reports/powerbi/` — test predictions, confusion matrix, and model metrics

## Business interpretation

At the selected threshold, the holdout evaluation flags 442 customers, correctly identifies 247 churners, and misses 127 churners. A real retention workflow should define the intervention cost and expected customer value, then recalibrate the threshold on current data. The model is a prioritization aid, not an automatic decision system.


## Data note

The included CSV is the IBM Telco Customer Churn educational dataset. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for citation and redistribution guidance. Review the current terms before publishing the raw file; if redistribution is not permitted, remove the CSV from the public repository and document how reviewers can obtain it.
