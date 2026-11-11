# Customer Churn Prediction Pipeline

A reproducible educational machine-learning project that compares Logistic Regression with a tuned Random Forest, selects a recall-oriented threshold on validation data, and exports aggregate Power BI metrics.

## Demo video

[Watch the narrated customer-churn demo on Google Drive](https://drive.google.com/file/d/1yXLvK6oFT_6QYG4tiOYj223wN8CiTvfF/view?usp=sharing)

The video explains the EDA, baseline and improved model comparison, validation threshold selection, held-out test results, business interpretation, and limitations.

## Results

The selected model is Logistic Regression because its validation PR-AUC (0.678) slightly exceeded the tuned Random Forest comparison (0.673). The threshold 0.6477 was selected on validation for a 70% recall target. On the held-out test set, precision is 0.559, recall 0.660, F1 0.605, PR-AUC 0.624, and ROC-AUC 0.832. The validation recall target did not fully generalize to test and requires further calibration before operational use.

See [`reports/Model_Report_EN.md`](reports/Model_Report_EN.md) and [`reports/Model_Report_AR.md`](reports/Model_Report_AR.md) for the evaluation details.

## Data and privacy

The IBM Telco CSV, fitted model artifact, and row-level Power BI prediction export are not included in this repository copy. Dataset redistribution terms were not independently confirmed, and the source contains a customer identifier field. Obtain the educational dataset from an authorized source only after reviewing its terms, then place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` for local training. The pipeline excludes `customerID` from features, and the Power BI exporter omits it from row-level output. Locally generated raw data, models, and row-level exports are ignored by Git. Do not publish those files without permission and a privacy review.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`data/README.md`](data/README.md) for details.

## Quick start

After obtaining an authorized local dataset file:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train_pipeline.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --output . \
  --iterations 12 \
  --cv 3 \
  --target-recall 0.70
```

To regenerate aggregate Power BI outputs after training:

```bash
python src/export_powerbi_assets.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --model models/churn_pipeline.joblib \
  --output reports/powerbi
```

The row-level export is local only, excludes `customerID`, and is ignored by Git. The checked-in Power BI files contain aggregate metrics and confusion-matrix counts only.

## Deliverables

- `notebooks/Customer_Churn_Pipeline_Colab.ipynb` — Colab walkthrough
- `src/train_pipeline.py` — training, threshold selection, evaluation, and report generation
- `src/export_powerbi_assets.py` — local Power BI export; customer identifiers are omitted
- `reports/Model_Report_EN.md` and `reports/Model_Report_AR.md` — reviewer-facing reports
- `reports/figures/` — EDA, confusion matrix, and importance charts
- `reports/threshold_tradeoff.csv` — validation operating-point comparison
- `reports/powerbi/` — aggregate confusion matrix and model metrics

## Business interpretation and limitations

At the selected threshold, the held-out test evaluation flags 442 customers, correctly identifies 247 churners, and misses 127 churners. A real retention workflow should attach intervention costs and customer value, recalibrate on current data, and monitor subgroup performance and drift. This model is a prioritization aid, not an automatic decision system.

## Software license

No open-source license is granted for this project's code at this time. Public visibility does not grant permission to reuse or redistribute it. The IBM educational dataset has separate source terms and is not licensed by this repository. Add a project license only after confirming the copyright holder and their authority to grant it.
