#!/usr/bin/env python3
"""Reproducible Telco Customer Churn pipeline for local Python or Google Colab."""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
TARGET_RECALL = 0.70


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def metrics_at_threshold(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


def choose_threshold(y_true, probabilities, target_recall: float) -> tuple[float, dict, bool]:
    thresholds = np.unique(np.r_[probabilities, 0.5])
    candidates = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        candidates.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
        })
    table = pd.DataFrame(candidates)
    eligible = table[table["recall"] >= target_recall]
    target_met = not eligible.empty
    if target_met:
        row = eligible.sort_values(["precision", "f1", "threshold"], ascending=[False, False, False]).iloc[0]
    else:
        row = table.sort_values(["f1", "precision"], ascending=[False, False]).iloc[0]
    return float(row["threshold"]), row.to_dict(), target_met


def save_eda_plots(df: pd.DataFrame, figures_dir: Path) -> None:
    """Save compact, reviewer-friendly EDA charts before model training."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="Churn", order=["No", "Yes"], color="#4472C4")
    ax.set(title="Churn class counts", xlabel="Churn", ylabel="Customers")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, multiple="stack", palette={"No": "#4472C4", "Yes": "#ED7D31"})
    plt.title("Customer tenure by churn status")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(figures_dir / "tenure_by_churn.png", dpi=160)
    plt.close()

    categorical_columns = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, len(categorical_columns), figsize=(16, 4.5))
    for ax, column in zip(axes, categorical_columns):
        rates = (df.groupby(column, observed=False)["Churn"]
                   .apply(lambda values: (values == "Yes").mean())
                   .sort_values(ascending=False).reset_index(name="churn_rate"))
        sns.barplot(data=rates, x="churn_rate", y=column, ax=ax, color="#ED7D31")
        ax.set_title(f"Churn rate by {column}")
        ax.set_xlabel("Churn rate")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_rate_by_category.png", dpi=160)
    plt.close()

    eda = df.copy()
    eda["tenure_group"] = pd.cut(
        eda["tenure"], bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"]
    )
    tenure_rates = (eda.groupby("tenure_group", observed=False)["Churn"]
                      .apply(lambda values: (values == "Yes").mean())
                      .reset_index(name="churn_rate"))
    plt.figure(figsize=(7, 4.5))
    sns.barplot(data=tenure_rates, x="tenure_group", y="churn_rate", color="#ED7D31")
    plt.title("Churn rate by tenure band")
    plt.xlabel("Tenure (months)")
    plt.ylabel("Churn rate")
    plt.gca().yaxis.set_major_formatter(lambda value, position: f"{value:.0%}")
    plt.tight_layout()
    plt.savefig(figures_dir / "churn_rate_by_tenure_band.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to Telco churn CSV")
    parser.add_argument("--output", required=True, help="Project/output directory")
    parser.add_argument("--iterations", type=int, default=12, help="RandomizedSearch iterations (default: 12)")
    parser.add_argument("--cv", type=int, default=3, help="CV folds for tuning (default: 3)")
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    args = parser.parse_args()

    project_dir = Path(args.output).expanduser().resolve()
    reports_dir = project_dir / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = project_dir / "models"
    for directory in [reports_dir, figures_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    required = {"Churn", "TotalCharges"}
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    original_rows = len(df)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    totalcharges_missing_after_conversion = int(df["TotalCharges"].isna().sum())
    if totalcharges_missing_after_conversion:
        unresolved = df.loc[df["TotalCharges"].isna(), "tenure"] != 0
        if unresolved.any():
            raise ValueError("Some missing TotalCharges rows have nonzero tenure; inspect them before imputation.")
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df = df.dropna(subset=["Churn"]).copy()
    df = df[df["Churn"].isin(["No", "Yes"])].copy()

    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    preprocessor = build_preprocessor(X_train)

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    baseline.fit(X_train, y_train)

    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_distributions = {
        "model__n_estimators": randint(150, 401),
        "model__max_depth": [None, 5, 8, 12, 16],
        "model__min_samples_split": randint(2, 15),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": ["sqrt", "log2"],
    }
    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_distributions,
        n_iter=args.iterations,
        scoring="average_precision",
        cv=args.cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_

    baseline_val_prob = baseline.predict_proba(X_val)[:, 1]
    rf_val_prob = best_rf.predict_proba(X_val)[:, 1]
    baseline_val_ap = float(average_precision_score(y_val, baseline_val_prob))
    rf_val_ap = float(average_precision_score(y_val, rf_val_prob))
    selected_name, selected_model = ("Random Forest", best_rf) if rf_val_ap >= baseline_val_ap else ("Logistic Regression baseline", baseline)
    val_prob = selected_model.predict_proba(X_val)[:, 1]
    chosen_threshold, threshold_selection, target_recall_met = choose_threshold(y_val, val_prob, args.target_recall)

    test_prob = selected_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= chosen_threshold).astype(int)
    test_metrics = metrics_at_threshold(y_test, test_prob, chosen_threshold)
    # Show the business trade-off at several candidate operating points on validation.
    operating_thresholds = [0.50, 0.60, float(chosen_threshold), 0.70]
    operating_rows = []
    for operating_threshold in operating_thresholds:
        operating_pred = (val_prob >= operating_threshold).astype(int)
        operating_rows.append({
            "threshold": operating_threshold,
            "precision": precision_score(y_val, operating_pred, zero_division=0),
            "recall": recall_score(y_val, operating_pred, zero_division=0),
            "f1": f1_score(y_val, operating_pred, zero_division=0),
            "flagged_customers": int(operating_pred.sum()),
        })
    threshold_tradeoff = pd.DataFrame(operating_rows).drop_duplicates(subset=["threshold"])
    threshold_tradeoff.to_csv(reports_dir / "threshold_tradeoff.csv", index=False)
    test_metrics["target_recall_goal"] = float(args.target_recall)
    test_metrics["target_recall_met_on_validation"] = bool(target_recall_met)
    test_metrics["target_recall_met_on_test"] = bool(test_metrics["recall"] >= args.target_recall)
    test_metrics["flagged_customers_on_test"] = int(test_pred.sum())
    test_metrics["missed_churners_on_test"] = int(((y_test.to_numpy() == 1) & (test_pred == 0)).sum())
    test_metrics["selected_model"] = selected_name
    test_metrics["validation_threshold_tradeoff"] = threshold_tradeoff.to_dict(orient="records")
    test_metrics["validation_threshold_metrics"] = {
        key: float(value) for key, value in threshold_selection.items()
    }
    test_metrics["baseline_validation_pr_auc"] = baseline_val_ap
    test_metrics["random_forest_validation_pr_auc"] = rf_val_ap
    test_metrics["random_forest_best_cv_pr_auc"] = float(search.best_score_)
    test_metrics["random_forest_best_params"] = search.best_params_
    test_metrics["data"] = {
        "rows_used": int(len(df)),
        "original_rows": int(original_rows),
        "columns_in_source": int(df.shape[1]),
        "totalcharges_missing_after_numeric_conversion": totalcharges_missing_after_conversion,
        "churn_count": {str(k): int(v) for k, v in df["Churn"].value_counts().items()},
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2, ensure_ascii=False)
    pd.DataFrame([
        {"model": "Logistic Regression baseline", "validation_pr_auc": baseline_val_ap},
        {"model": "Tuned Random Forest", "validation_pr_auc": rf_val_ap},
    ]).to_csv(reports_dir / "model_comparison.csv", index=False)
    report = classification_report(y_test, test_pred, target_names=["No Churn", "Churn"], zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(reports_dir / "classification_report.csv")

    sns.set_theme(style="whitegrid")
    save_eda_plots(df, figures_dir)
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted No", "Predicted Yes"], yticklabels=["Actual No", "Actual Yes"])
    plt.title(f"Test confusion matrix (threshold={chosen_threshold:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "test_confusion_matrix.png", dpi=160)
    plt.close()

    permutation = permutation_importance(
        selected_model, X_val, y_val, scoring="average_precision", n_repeats=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(reports_dir / "permutation_importance.csv", index=False)
    top = importance.head(15).sort_values("importance_mean")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#4472C4", alpha=0.9)
    plt.title("Permutation importance (validation set)")
    plt.xlabel("Decrease in average precision when shuffled")
    plt.tight_layout()
    plt.savefig(figures_dir / "permutation_importance.png", dpi=160)
    plt.close()

    joblib.dump({"model": selected_model, "threshold": chosen_threshold, "target_recall": args.target_recall, "model_name": selected_name}, models_dir / "churn_pipeline.joblib")

    rows = df["Churn"].value_counts()
    churn_rate = float((y == 1).mean())
    report_md = f"""# تقرير نموذج التنبؤ بتسرب العملاء\n\n## ملخص التنفيذ\n\n- عدد الصفوف: **{len(df):,}**، وعدد الأعمدة الأصلية: **{df.shape[1]}**.\n- نسبة التسرب: **{churn_rate:.1%}** ({int(rows.get('Yes', 0)):,} من {len(df):,}).\n- النموذج المختار اعتماداً على PR-AUC للتحقق: **{selected_name}**.\n- تقسيم Stratified: تدريب {len(X_train):,}، تحقق {len(X_val):,}، اختبار {len(X_test):,}.\n- العتبة المختارة من مجموعة التحقق: **{chosen_threshold:.4f}**؛ هدف Recall = **{args.target_recall:.0%}**؛ تحقق الهدف على التحقق: **{target_recall_met}**.\n\n## نتائج الاختبار النهائي\n\n| المقياس | النتيجة |\n|---|---:|\n| Precision | {test_metrics['precision']:.3f} |\n| Recall | {test_metrics['recall']:.3f} |\n| F1 | {test_metrics['f1']:.3f} |\n| PR-AUC / Average Precision | {test_metrics['pr_auc_average_precision']:.3f} |\n| ROC-AUC | {test_metrics['roc_auc']:.3f} |\n\n**مصفوفة الالتباس** (الصفوف: الواقع؛ الأعمدة: التنبؤ)\n\n```text\n{np.array(cm)}\n```\n\n## المقارنة وضبط النموذج\n\n- Logistic Regression validation PR-AUC: **{baseline_val_ap:.3f}**.\n- Random Forest validation PR-AUC: **{rf_val_ap:.3f}**.\n- أفضل Random Forest CV PR-AUC: **{search.best_score_:.3f}**.\n- تم استخدام `class_weight` للتعامل مع عدم التوازن بدلاً من إعادة أخذ العينات.\n\n## المعالجة والقيود\n\n- تم تحويل `TotalCharges` إلى رقم؛ القيم التي تعذر تحويلها إلى أرقام: **{totalcharges_missing_after_conversion}**؛ كانت جميعها لعملاء tenure=0 وعُوّضت بصفر وفق دلالة البيانات.\n- حُذف `customerID` من الميزات، واستخدم ترميز One-Hot للفئات وقياس للميزات الرقمية.\n- اختيرت العتبة على مجموعة التحقق، واستخدمت مجموعة الاختبار مرة واحدة للتقييم النهائي.\n- Permutation importance تشرح اعتماد أداء النموذج على الميزات ولا تثبت السببية.\n- هذه النتائج تعليمية وعلى بيانات IBM Telco عامة/مرفقة؛ لا تضمن الأداء على بيانات شركة فعلية، ولا تُعد توصية تشغيلية دون تحقق ومراجعة خصوصية وعدالة.\n\n## ملفات النتائج\n\n- `metrics.json` — المقاييس والبيانات المستخدمة.\n- `classification_report.csv` — تقرير التصنيف.\n- `model_comparison.csv` — مقارنة النموذجين على التحقق.\n- `permutation_importance.csv` — أهمية الميزات.\n- `figures/` — رسوم الاستكشاف ومصفوفة الالتباس والأهمية.\n"""
    top_features_md = "\n".join(
        f"- {row.feature}: {row.importance_mean:.3f}" for row in importance.head(5).itertuples()
    )
    report_md = report_md.replace(
        "\n## المعالجة والقيود\n",
        "\n## أعلى الميزات في Permutation Importance\n\n" + top_features_md + "\n\n## المعالجة والقيود\n"
    )
    (reports_dir / "Model_Report_AR.md").write_text(report_md, encoding="utf-8")

    comparison_note = (
        "The tuned Random Forest did not improve validation PR-AUC, so the Logistic Regression selected reference model was retained "
        "because it was slightly stronger and simpler."
        if rf_val_ap < baseline_val_ap else
        "The tuned Random Forest was retained because it achieved the stronger validation PR-AUC."
    )
    threshold_table = "\n".join(
        f"| {row.threshold:.4f} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | {int(row.flagged_customers)} |"
        for row in threshold_tradeoff.itertuples()
    )
    english_report = f"""# Customer Churn Prediction — Model Report

## Executive summary

This project predicts which Telco customers are at risk of churn so a retention team can prioritize outreach. The final model is the **Logistic Regression selected reference model**, selected as the simpler reference model because it achieved the stronger validation PR-AUC compared with the tuned Random Forest. The operating threshold was selected on the validation split to target recall of **{args.target_recall:.0%}**, then evaluated once on an unseen test split.

## Data and preprocessing

- Dataset: IBM Telco Customer Churn educational dataset
- Rows used: **{len(df):,}**
- Source columns: **{df.shape[1]}**
- Churned customers: **{int(rows.get('Yes', 0)):,} ({churn_rate:.1%})**
- Split: **60% train / 20% validation / 20% test**, stratified and seeded with 42
- `customerID`, when present in a source file, is excluded from modeling and is not included in row-level Power BI exports
- `TotalCharges` converted to numeric; **{totalcharges_missing_after_conversion}** blank values were replaced with 0 because those records have `tenure = 0`
- Numeric imputation/scaling and categorical one-hot encoding are fitted inside the pipeline

## Model comparison

| Model | Validation PR-AUC |
|---|---:|
| Logistic Regression (selected reference model) | {baseline_val_ap:.3f} |
| Tuned Random Forest comparison | {rf_val_ap:.3f} |

**Interpretation:** {comparison_note}

## Threshold and test results

The selected validation threshold is **{chosen_threshold:.4f}**. Validation recall was **{threshold_selection['recall']:.1%}**, meeting the 70% target on validation. On the unseen test set, recall was **{test_metrics['recall']:.1%}**, so the target did **not** generalize fully to test data.

| Metric | Test result |
|---|---:|
| Precision | {test_metrics['precision']:.3f} |
| Recall | {test_metrics['recall']:.3f} |
| F1 | {test_metrics['f1']:.3f} |
| PR-AUC / Average Precision | {test_metrics['pr_auc_average_precision']:.3f} |
| ROC-AUC | {test_metrics['roc_auc']:.3f} |

Confusion matrix, with rows as actual and columns as predicted:

```text
{np.array(cm)}
```

The threshold flags **{int(test_pred.sum()):,}** of {len(y_test):,} test customers, correctly identifies **{int(cm[1, 1]):,}** churners, and misses **{int(cm[1, 0]):,}** churners. A business should attach a retention action and cost to each flag before using this in production.

## Threshold trade-off on validation

The table below shows the precision/recall trade-off at several candidate thresholds. The selected threshold is chosen on validation to prioritize the stated recall goal; the test set remains reserved for final evaluation.

| Threshold | Precision | Recall | F1 | Flagged customers |
|---:|---:|---:|---:|---:|
{threshold_table}

## Explainability and EDA

Permutation importance on validation identifies the features most associated with predictive performance. The top features are:

{top_features_md}

Additional EDA charts cover churn rate by contract, internet service, payment method, and tenure band. Importance is predictive rather than causal.

## Limitations and responsible use

This is an educational model on a public dataset. The test recall is below the validation target, so the operating point requires further calibration on current company data. Monitor drift, subgroup performance, intervention cost, and fairness before production use. Do not use customer identifiers or predictions as automatic grounds for adverse decisions.
"""
    (reports_dir / "Model_Report_EN.md").write_text(english_report, encoding="utf-8")

    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print(f"\nArtifacts saved under: {project_dir}")


if __name__ == "__main__":
    main()
