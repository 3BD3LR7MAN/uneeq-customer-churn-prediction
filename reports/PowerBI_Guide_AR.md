# دليل بناء لوحة Power BI لمشروع تسرب العملاء

هذا الدليل يبني لوحة من بيانات العملاء، تنبؤات مجموعة الاختبار، ومقاييس النموذج. استخدم Power BI Desktop، ثم احفظ العمل بصيغة `.pbix`.

## 1) جهّز ملفات الإدخال

بعد فك ضغط المشروع، استخدم هذه الملفات:

- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` — جميع العملاء للاستكشاف.
- `reports/powerbi/PowerBI_test_predictions.csv` — الصفوف المحجوزة للاختبار، مع الاحتمال والتنبؤ والحقيقة.
- `reports/powerbi/PowerBI_model_metrics.csv` — مقاييس الاختبار الجاهزة كبطاقات.
- `reports/powerbi/PowerBI_confusion_matrix.csv` — جدول موجز لمصفوفة الالتباس.
- `reports/permutation_importance.csv` — أهمية الميزات.

إذا لم تظهر ملفات `reports/powerbi/` في الحزمة، أنشئها من مجلد المشروع:

```bash
python src/export_powerbi_assets.py \
  --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  --model models/churn_pipeline.joblib \
  --output reports/powerbi
```

ملف `PowerBI_test_predictions.csv` يعرض **مجموعة الاختبار فقط**؛ لا تستخدمه لتمثيل جميع العملاء. ملف البيانات الأصلي مناسب لتحليلات EDA، بينما تقييم النموذج يجب أن يعتمد على ملف الاختبار.

## 2) استورد الجداول

في Power BI Desktop: **Home → Get data → Text/CSV**، واستورد الملفات الأربعة أعلاه. أعد تسمية الاستعلامات إلى:

- `Customers` للبيانات الأصلية.
- `TestPredictions` لتنبؤات الاختبار.
- `ModelMetrics` لمقاييس النموذج.
- `FeatureImportance` لأهمية الميزات.

راجع الأنواع في Power Query:

- `tenure`, `MonthlyCharges`, `TotalCharges`, `Churn_Probability`, `Decision_Threshold`, `Value`, و`Count`: أرقام.
- `Churn`, `Actual_Churn`, `Predicted_Churn`, `Outcome`, `Contract`, `InternetService`: نصوص.
- لا يظهر `customerID` في تصدير `TestPredictions` المنشور؛ يُستبعد من الميزات ومن المخرجات المشتركة.

في الملف الأصلي، قد تُقرأ فراغات `TotalCharges` كنص. في هذه العينة تحديداً، القيم الإحدى عشرة التي تعذّر تحويلها تخص `tenure = 0` وقد عُوّضت بصفر في المعالجة. إذا كنت تنظف الملف داخل Power Query، حوّل العمود إلى Decimal Number، وافحص أخطاء التحويل، ثم عوّض بصفر **فقط** عندما تكون `tenure = 0`؛ لا تطبق ذلك تلقائياً على بيانات جديدة.

تصدير `TestPredictions` المنشور لا يحتوي على معرّفات، لذلك لا تربطه بجدول `Customers` عبر ترتيب الصفوف. أبقِ الجداول منفصلة. إذا احتاج سير عمل داخلي مرخّص إلى مفتاح ربط، أضف رمزاً مستعاراً داخل بيئة خاصة ومصرّح بها، ولا تنشره في المستودع العام.

## 3) أنشئ مقاييس DAX

من **Modeling → New measure**، أضف هذه المقاييس إلى جدول `TestPredictions`:

```DAX
TP =
CALCULATE(
    COUNTROWS(TestPredictions),
    TestPredictions[Actual_Churn] = "Yes",
    TestPredictions[Predicted_Churn] = "Yes"
)

FP =
CALCULATE(
    COUNTROWS(TestPredictions),
    TestPredictions[Actual_Churn] = "No",
    TestPredictions[Predicted_Churn] = "Yes"
)

FN =
CALCULATE(
    COUNTROWS(TestPredictions),
    TestPredictions[Actual_Churn] = "Yes",
    TestPredictions[Predicted_Churn] = "No"
)

TN =
CALCULATE(
    COUNTROWS(TestPredictions),
    TestPredictions[Actual_Churn] = "No",
    TestPredictions[Predicted_Churn] = "No"
)

Precision = DIVIDE([TP], [TP] + [FP], 0)
Recall = DIVIDE([TP], [TP] + [FN], 0)
F1 = DIVIDE(2 * [Precision] * [Recall], [Precision] + [Recall], 0)
Test Customers = COUNTROWS(TestPredictions)
```

أنشئ أيضاً مقاييس EDA ضمن `Customers`:

```DAX
Customers Count = COUNTROWS(Customers)

Churned Customers =
CALCULATE(
    [Customers Count],
    Customers[Churn] = "Yes"
)

Churn Rate = DIVIDE([Churned Customers], [Customers Count], 0)
```

نسّق `Churn Rate`, `Precision`, `Recall`, و`F1` كنسبة مئوية مع 1 منزلة عشرية. المقاييس TP/FP/FN/TN أعداد صحيحة.

## 4) صمّم صفحات التقرير

### الصفحة 1 — نظرة تنفيذية

ضع عنواناً واضحاً مثل **Customer Churn — Overview**، ثم صف بطاقات KPI:

- عدد العملاء — `Customers Count`.
- العملاء المتسربون — `Churned Customers`.
- معدل التسرب — `Churn Rate`.
- Recall على الاختبار — `[Recall]`.

أضف مخططاً عمودياً يوضح عدد العملاء حسب `Churn`، ومخططاً شريطياً أفقياً لمعدل التسرب حسب `Contract` (المقام هو عدد العملاء في الفئة، وليس إجمالي العملاء). أضف Slicers لـ`Contract`, `InternetService`, و`PaymentMethod`.

### الصفحة 2 — استكشاف البيانات EDA

- **100% Stacked column:** المحور `Contract`، وسلسلة البيانات `Churn`، والقيمة عدد العملاء. هذا يسهّل مقارنة نسبة التسرب بين أنواع العقود.
- **Column chart:** عدد العملاء حسب `tenure` بعد تجميعه في فئات مثل 0–12، 13–24، 25–48، 49+ شهراً؛ استخدم Legend لـ`Churn`.
- **Bar chart:** معدل التسرب حسب `InternetService` أو `PaymentMethod`.
- **Scatter chart:** `MonthlyCharges` مقابل `tenure`، وLegend حسب `Churn`، مع شفافية/حجم نقاط مناسب لتخفيف التداخل.

لإنشاء فئات مدة الاشتراك من **Modeling → New column**:

```DAX
Tenure Group =
SWITCH(
    TRUE(),
    Customers[tenure] <= 12, "01: 0–12 months",
    Customers[tenure] <= 24, "02: 13–24 months",
    Customers[tenure] <= 48, "03: 25–48 months",
    "04: 49+ months"
)
```

البادئات الرقمية تساعد على ترتيب الفئات. من **Column tools → Sort by column** يمكن ترتيبها حسب عمود ترتيب رقمي إذا أنشأت واحداً.

### الصفحة 3 — أداء النموذج

- أضف بطاقات `Precision`, `Recall`, `F1` من مقاييس DAX أعلاه.
- أضف بطاقة `PR-AUC` باستيراد `PowerBI_model_metrics.csv` واستخدام قيمة صف `Metric = PR-AUC / Average Precision`؛ أو اعرضها من `ModelMetrics[Value]` بعد إضافة فلتر `Metric` المناسب.
- أضف **Matrix visual**: `Rows = Actual_Churn`, `Columns = Predicted_Churn`, `Values = Outcome Count` من `TestPredictions`. يجب أن تكون أعداد الخلايا: TN=840، FP=195، FN=127، TP=247.
- لإظهار أسماء الحالات بدلاً من العدد، أضف مقياس `Outcome Count = COUNTROWS(TestPredictions)` واستخدمه في Values.
- أضف شريط `FeatureImportance[importance_mean]` حسب `FeatureImportance[feature]` ورتّب تنازلياً، ثم طبّق Top N = 10 إذا ازدحم الرسم.
- أضف بطاقة أو ملاحظة توضّح أن العتبة المستخدمة **0.6477**، وأن هدف Recall 70% تحقق في validation لكنه لم يتحقق على الاختبار (Recall الفعلي 66.0%).

إذا لم تعرض Matrix القيم الصحيحة، راجع اتجاه الصفوف والأعمدة؛ الصفوف هي الحقيقة `Actual_Churn`، والأعمدة هي التنبؤ `Predicted_Churn`.

## 5) تحسين المظهر والتفاعل

- استخدم خلفية فاتحة ومحايدة، ولوناً ثابتاً للبقاء مثل أزرق، ولوناً للتسرب مثل برتقالي/أحمر. لا تعتمد على اللون وحده؛ اكتب تسميات واضحة.
- ثبّت أحجام الخطوط، وحاذِ البطاقات والعناوين، واترك فراغاً بين المرئيات.
- من **View → Themes** اختر سمة بسيطة، ثم اضبط الألوان لتبقى متسقة عبر الصفحات.
- من **Format → Edit interactions** تأكد أن Slicers تؤثر فقط على الرسوم المناسبة. يمكن إبقاء مؤشرات الاختبار منفصلة عن تحليلات البيانات الكاملة حتى لا تختلط المقامات.
- أضف Tooltips فيها العدد والنسبة، وأوقف الحدود/الظلال غير اللازمة. تجنب الـ3D والمخططات الدائرية كثيرة الفئات.
- أضف تذييلاً: «النتائج تعليمية على بيانات Telco عامة؛ ليست ضماناً لأداء تشغيلي. أهمية الميزات لا تثبت السببية.»

## 6) التحقق قبل التسليم

1. تحقق أن صفحة EDA تعتمد على `Customers`، وصفحة الأداء تعتمد على `TestPredictions`.
2. راجع مصفوفة الالتباس مقابل أعداد الاختبار: TN=840، FP=195، FN=127، TP=247.
3. تأكد من أن Recall المحسوب من الجدول قريب من 66.0%، وPrecision قرابة 55.9%.
4. أظهر مصدر البيانات، حجم العينة، عتبة القرار، وتاريخ/إصدار النموذج في صفحة أو Tooltip.
5. احفظ الملف بصيغة `.pbix`. مشاركة التقرير على Power BI Service أو نشره للعامة إجراء منفصل، ولا ترفع بيانات إلا بعد مراجعة الترخيص والخصوصية.
