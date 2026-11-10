# Data source and citation

This project was evaluated on the **IBM Telco Customer Churn educational dataset**, commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

Suggested citation:

> IBM Watson Analytics sample data, Telco Customer Churn dataset. Used for educational and portfolio evaluation.

The source CSV is not included in this repository because its current redistribution and licensing terms have not been independently confirmed. Obtain the dataset from an authorized source and review its terms before local use. The project does not grant rights to the dataset.

The source may include `customerID`; the training code excludes that field from model features, and the Power BI exporter omits it from row-level output. Do not commit the raw source file, model artifact, or row-level predictions to this repository.
