#!/usr/bin/env python3
"""Export deterministic holdout predictions and metrics for Power BI visuals."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to original churn CSV')
    parser.add_argument('--model', required=True, help='Path to churn_pipeline.joblib')
    parser.add_argument('--output', required=True, help='Output directory for Power BI CSVs')
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing = df['TotalCharges'].isna()
    if missing.any() and not (df.loc[missing, 'tenure'] == 0).all():
        raise ValueError('Some blank TotalCharges rows have nonzero tenure; inspect before export.')
    df.loc[missing, 'TotalCharges'] = 0.0

    X = df.drop(columns=['Churn', 'customerID'])
    y = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=42, stratify=y
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    package = joblib.load(args.model)
    model = package['model']
    threshold = float(package['threshold'])
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    exported = X_test.copy()
    exported.insert(0, 'customerID', df.loc[X_test.index, 'customerID'])
    exported['Actual_Churn'] = y_test.map({0: 'No', 1: 'Yes'})
    exported['Churn_Probability'] = probabilities
    exported['Predicted_Churn'] = np.where(predictions == 1, 'Yes', 'No')
    exported['Decision_Threshold'] = threshold
    exported['Outcome'] = np.select(
        [(y_test.to_numpy() == 1) & (predictions == 1),
         (y_test.to_numpy() == 0) & (predictions == 1),
         (y_test.to_numpy() == 1) & (predictions == 0)],
        ['TP', 'FP', 'FN'], default='TN'
    )
    exported.to_csv(out / 'PowerBI_test_predictions.csv', index=False)

    cm_rows = []
    for actual in ['No', 'Yes']:
        for predicted in ['No', 'Yes']:
            count = int(((exported['Actual_Churn'] == actual) & (exported['Predicted_Churn'] == predicted)).sum())
            cm_rows.append({'Actual_Churn': actual, 'Predicted_Churn': predicted, 'Count': count})
    pd.DataFrame(cm_rows).to_csv(out / 'PowerBI_confusion_matrix.csv', index=False)

    y_binary = y_test.to_numpy()
    metrics = [
        ('Decision threshold', threshold),
        ('Precision', precision_score(y_binary, predictions, zero_division=0)),
        ('Recall', recall_score(y_binary, predictions, zero_division=0)),
        ('F1', f1_score(y_binary, predictions, zero_division=0)),
        ('PR-AUC / Average Precision', average_precision_score(y_binary, probabilities)),
        ('ROC-AUC', roc_auc_score(y_binary, probabilities)),
    ]
    pd.DataFrame(metrics, columns=['Metric', 'Value']).assign(Context='Test holdout').to_csv(
        out / 'PowerBI_model_metrics.csv', index=False
    )

    print(f'Test rows: {len(exported)}; threshold: {threshold:.6f}')
    print('Exports:', ', '.join(p.name for p in out.glob('PowerBI_*.csv')))
    print(pd.DataFrame(cm_rows).pivot(index='Actual_Churn', columns='Predicted_Churn', values='Count'))


if __name__ == '__main__':
    main()
