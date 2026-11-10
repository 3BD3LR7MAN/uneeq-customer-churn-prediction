from pathlib import Path
import nbformat as nbf

out = Path(__file__).resolve().parents[1] / "notebooks" / "Customer_Churn_Pipeline_Colab.ipynb"
nb = nbf.v4.new_notebook()
nb.metadata = {
    "colab": {"name": "Customer_Churn_Pipeline_Colab.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
}
cells = []
cells.append(nbf.v4.new_markdown_cell("""# Customer Churn Prediction Pipeline

دفتر تدريبي متكامل لمجموعة Telco Customer Churn. يعمل في Google Colab ويُنتج تقريراً ورسومات ونموذجاً محفوظاً في Google Drive.

**طريقة التشغيل:** شغّل الخلايا بالترتيب. اربط Google Drive عند الطلب، وتأكد من وجود ملف CSV في مجلد المشروع. لا تستخدم بيانات الاختبار لاختيار النموذج أو العتبة.
"""))
cells.append(nbf.v4.new_code_cell("""# 1) ربط Google Drive وإنشاء مجلدات المشروع
from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = '/content/drive/MyDrive/customer-churn'
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
REPORT_DIR = os.path.join(PROJECT_DIR, 'reports')
FIGURES_DIR = os.path.join(REPORT_DIR, 'figures')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
for folder in [DATA_DIR, REPORT_DIR, FIGURES_DIR, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)
print('Project folder:', PROJECT_DIR)
"""))
cells.append(nbf.v4.new_code_cell("""# 2) تثبيت الاعتمادات
%pip -q install pandas numpy scikit-learn matplotlib seaborn scipy joblib
"""))
cells.append(nbf.v4.new_code_cell("""# 3) تحميل ملف CSV؛ إذا لم يكن في Drive، اختر الملف المرفق من جهازك
import os
DATA_PATH = os.path.join(DATA_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
if not os.path.exists(DATA_PATH):
    from google.colab import files
    uploaded = files.upload()
    uploaded_name = next(iter(uploaded))
    with open(DATA_PATH, 'wb') as f:
        f.write(uploaded[uploaded_name])
print('Using:', DATA_PATH)
"""))
cells.append(nbf.v4.new_code_cell("""# 4) قراءة البيانات وفحص الجودة
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
TARGET_RECALL = 0.70

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
print('Shape:', df.shape)
display(df.head())
print('Target counts:')
display(df['Churn'].value_counts())
print('Raw missing values:')
display(df.isna().sum().sort_values(ascending=False).head(10))

# TotalCharges قد يحتوي فراغات نصية لا تظهر كقيم مفقودة عند القراءة
original_rows = len(df)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
missing_totalcharges = int(df['TotalCharges'].isna().sum())
if missing_totalcharges:
    assert (df.loc[df['TotalCharges'].isna(), 'tenure'] == 0).all(), 'افحص القيم الفارغة ذات tenure غير الصفرية قبل التعويض'
    df.loc[df['TotalCharges'].isna(), 'TotalCharges'] = 0.0
print('TotalCharges blanks converted to numeric:', missing_totalcharges, '(filled with 0 only for tenure=0)')
"""))
cells.append(nbf.v4.new_code_cell("""# 5) استكشاف مرئي بسيط
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=df, x='Churn', order=['No', 'Yes'], color='#4472C4', ax=axes[0])
axes[0].set_title('Churn class counts')
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, multiple='stack', ax=axes[1])
axes[1].set_title('Tenure by churn')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'eda_overview.png'), dpi=160)
plt.show()
print('Churn rate:', round((df['Churn'] == 'Yes').mean(), 4))
"""))
cells.append(nbf.v4.new_code_cell("""# 6) Business-oriented EDA: churn rate by key segments
categorical_columns = ['Contract', 'InternetService', 'PaymentMethod']
fig, axes = plt.subplots(1, len(categorical_columns), figsize=(16, 4.5))
for ax, column in zip(axes, categorical_columns):
    rates = (df.groupby(column, observed=False)['Churn']
               .apply(lambda values: (values == 'Yes').mean())
               .sort_values(ascending=False).reset_index(name='churn_rate'))
    sns.barplot(data=rates, x='churn_rate', y=column, ax=ax, color='#ED7D31')
    ax.set_title(f'Churn rate by {column}')
    ax.set_xlabel('Churn rate')
    ax.set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'churn_rate_by_category.png'), dpi=160)
plt.show()

df['tenure_group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, np.inf], labels=['0-12', '13-24', '25-48', '49+'])
tenure_rates = (df.groupby('tenure_group', observed=False)['Churn']
                  .apply(lambda values: (values == 'Yes').mean())
                  .reset_index(name='churn_rate'))
sns.barplot(data=tenure_rates, x='tenure_group', y='churn_rate', color='#ED7D31')
plt.title('Churn rate by tenure band')
plt.xlabel('Tenure (months)')
plt.ylabel('Churn rate')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'churn_rate_by_tenure_band.png'), dpi=160)
plt.show()
"""))
cells.append(nbf.v4.new_code_cell("""# 7) فصل الهدف وتقسيم train / validation / test (60/20/20، مع stratify)
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn'].map({'No': 0, 'Yes': 1}).astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print('Train / validation / test:', len(X_train), len(X_val), len(X_test))
"""))
cells.append(nbf.v4.new_code_cell("""# 8) معالجة مسبقة داخل Pipeline لمنع تسرب المعلومات
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=['number']).columns.tolist()

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler(with_mean=False))
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_features),
    ('categorical', categorical_pipeline, categorical_features)
])
"""))
cells.append(nbf.v4.new_code_cell("""# 9) Baseline: Logistic Regression مع class_weight لمعالجة عدم التوازن
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
])
baseline.fit(X_train, y_train)
baseline_val_prob = baseline.predict_proba(X_val)[:, 1]
print('Baseline validation PR-AUC:', average_precision_score(y_val, baseline_val_prob))
"""))
cells.append(nbf.v4.new_code_cell("""# 10) نموذج مقارنة مضبوط: Random Forest مع بحث معاملات داخل Cross-Validation
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1))
])
param_distributions = {
    'model__n_estimators': randint(150, 401),
    'model__max_depth': [None, 5, 8, 12, 16],
    'model__min_samples_split': randint(2, 15),
    'model__min_samples_leaf': randint(1, 10),
    'model__max_features': ['sqrt', 'log2']
}
search = RandomizedSearchCV(
    rf_pipeline, param_distributions=param_distributions,
    n_iter=12, scoring='average_precision', cv=3,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)
best_rf = search.best_estimator_
rf_val_prob = best_rf.predict_proba(X_val)[:, 1]
print('Best CV PR-AUC:', search.best_score_)
print('Random Forest validation PR-AUC:', average_precision_score(y_val, rf_val_prob))
print('Best parameters:', search.best_params_)
"""))
cells.append(nbf.v4.new_code_cell("""# 11) مقارنة النموذجين واختيار الأفضل على validation فقط
from sklearn.metrics import precision_score, recall_score, f1_score

baseline_ap = average_precision_score(y_val, baseline_val_prob)
rf_ap = average_precision_score(y_val, rf_val_prob)
if rf_ap >= baseline_ap:
    selected_model, selected_name, val_prob = best_rf, 'Tuned Random Forest', rf_val_prob
else:
    selected_model, selected_name, val_prob = baseline, 'Logistic Regression baseline', baseline_val_prob
print('Selected model:', selected_name)
print('Validation PR-AUC:', max(baseline_ap, rf_ap))
"""))
cells.append(nbf.v4.new_code_cell("""# 12) اختيار العتبة على validation لتحقيق Recall >= 70% ثم تعظيم Precision
thresholds = np.unique(np.r_[val_prob, 0.5])
rows = []
for threshold in thresholds:
    pred = (val_prob >= threshold).astype(int)
    rows.append({
        'threshold': float(threshold),
        'precision': precision_score(y_val, pred, zero_division=0),
        'recall': recall_score(y_val, pred, zero_division=0),
        'f1': f1_score(y_val, pred, zero_division=0)
    })
threshold_table = pd.DataFrame(rows)
eligible = threshold_table[threshold_table['recall'] >= TARGET_RECALL]
target_met = not eligible.empty
if target_met:
    chosen = eligible.sort_values(['precision', 'f1', 'threshold'], ascending=[False, False, False]).iloc[0]
else:
    chosen = threshold_table.sort_values(['f1', 'precision'], ascending=[False, False]).iloc[0]
threshold = float(chosen['threshold'])
print(chosen)
if not target_met:
    print('تحذير: لم يتحقق هدف Recall على validation؛ تم اختيار أفضل F1.')
"""))
cells.append(nbf.v4.new_code_cell("""# 13) التقييم النهائي مرة واحدة على test
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

test_prob = selected_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= threshold).astype(int)
print('Decision threshold:', threshold)
print('PR-AUC:', average_precision_score(y_test, test_prob))
print('ROC-AUC:', roc_auc_score(y_test, test_prob))
print(classification_report(y_test, test_pred, target_names=['No Churn', 'Churn'], zero_division=0))
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.title(f'Test confusion matrix (threshold={threshold:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'test_confusion_matrix.png'), dpi=160)
plt.show()
"""))
cells.append(nbf.v4.new_code_cell("""# 14) Permutation importance على validation (تفسير تنبؤي لا سببي)
from sklearn.inspection import permutation_importance

perm = permutation_importance(
    selected_model, X_val, y_val, scoring='average_precision',
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
)
importance = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': perm.importances_mean,
    'importance_std': perm.importances_std
}).sort_values('importance_mean', ascending=False)
display(importance.head(15))
importance.head(15).sort_values('importance_mean').plot.barh(x='feature', y='importance_mean', xerr='importance_std', figsize=(8, 6), legend=False)
plt.title('Permutation importance (validation)')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'permutation_importance.png'), dpi=160)
plt.show()
"""))
cells.append(nbf.v4.new_code_cell("""# 15) حفظ النموذج والمقاييس والملفات في Google Drive
import json
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    'selected_model': selected_name,
    'threshold': threshold,
    'target_recall': TARGET_RECALL,
    'target_recall_met_on_validation': bool(target_met),
    'precision': float(precision_score(y_test, test_pred, zero_division=0)),
    'recall': float(recall_score(y_test, test_pred, zero_division=0)),
    'f1': float(f1_score(y_test, test_pred, zero_division=0)),
    'pr_auc_average_precision': float(average_precision_score(y_test, test_prob)),
    'roc_auc': float(roc_auc_score(y_test, test_prob)),
    'confusion_matrix': cm.tolist(),
    'baseline_validation_pr_auc': float(baseline_ap),
    'random_forest_validation_pr_auc': float(rf_ap),
    'random_forest_best_cv_pr_auc': float(search.best_score_),
    'random_forest_best_params': search.best_params_,
    'totalcharges_blanks_filled_zero': missing_totalcharges,
}
joblib.dump({'model': selected_model, 'threshold': threshold, 'model_name': selected_name}, os.path.join(MODEL_DIR, 'churn_pipeline.joblib'))
with open(os.path.join(REPORT_DIR, 'metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
importance.to_csv(os.path.join(REPORT_DIR, 'permutation_importance.csv'), index=False)
pd.DataFrame([
    {'model': 'Logistic Regression baseline', 'validation_pr_auc': baseline_ap},
    {'model': 'Tuned Random Forest', 'validation_pr_auc': rf_ap}
]).to_csv(os.path.join(REPORT_DIR, 'model_comparison.csv'), index=False)
pd.DataFrame(classification_report(y_test, test_pred, target_names=['No Churn', 'Churn'], zero_division=0, output_dict=True)).transpose().to_csv(os.path.join(REPORT_DIR, 'classification_report.csv'))
top_features_summary = '; '.join(f'{r.feature} ({r.importance_mean:.3f})' for r in importance.head(5).itertuples())
model_report = f'''# تقرير نموذج التنبؤ بتسرب العملاء

- عدد الصفوف: {len(df):,}؛ نسبة التسرب: {(y.mean()):.1%}.
- النموذج المختار: {selected_name}.
- العتبة المختارة على validation: {threshold:.4f}؛ هدف Recall {TARGET_RECALL:.0%} تحقق على validation: {target_met}.
- نتائج test: Precision={metrics['precision']:.3f}، Recall={metrics['recall']:.3f}، F1={metrics['f1']:.3f}، PR-AUC={metrics['pr_auc_average_precision']:.3f}، ROC-AUC={metrics['roc_auc']:.3f}.
- مصفوفة الالتباس (الصفوف واقع، الأعمدة تنبؤ): {cm.tolist()}.
- تمت معالجة {missing_totalcharges} قيمة TotalCharges فارغة بصفر لأن tenure=0.
- أهم الميزات حسب Permutation Importance: {top_features_summary}.
- أهمية الميزات تصف مساهمة تنبؤية ولا تثبت السببية. هذه النتائج تعليمية وليست ضماناً لأداء تشغيلي.
'''
with open(os.path.join(REPORT_DIR, 'Model_Report_AR.md'), 'w', encoding='utf-8') as f:
    f.write(model_report)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print('Saved to:', PROJECT_DIR)
"""))
cells.append(nbf.v4.new_markdown_cell("""## ملاحظات التسليم

- العتبة اختيرت باستخدام validation، والاختبار حُجز لتقدير الأداء النهائي.
- `class_weight` هو أسلوب الموازنة المستخدم؛ لم تُطبّق SMOTE.
- أهمية الميزات تصف اعتماد النموذج على البيانات ولا تثبت علاقة سببية.
- النتائج تخص هذه البيانات التعليمية ولا تضمن الأداء على بيانات شركة فعلية.
"""))
nb.cells = cells
nbf.write(nb, out)
print(out)
