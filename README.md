# Rossmann Store Sales Forecasting (Leakage-Safe Time Series ML)

## Project Overview

This project builds an **end-to-end, leakage-safe machine learning pipeline** to forecast **monthly sales per store** using the public **Rossmann Store Sales** dataset.

The goal is not just prediction accuracy, but to demonstrate **strong machine learning fundamentals**, including:

* Time-aware feature engineering
* Prevention of data leakage
* Proper preprocessing with pipelines
* Baseline comparison and multiple regression models

---

## Dataset

**Source:** Kaggle – *Rossmann Store Sales*

Files used:

* `train.csv` – daily sales data
* `store.csv` – store-level metadata

The dataset contains historical sales, promotions, holidays, and competition information for multiple stores over time.

---

## Problem Statement

Forecast **monthly gross sales per store** using historical sales and store-level features.

This is a **time-series regression problem** with:

* Multiple entities (stores)
* Strong seasonality
* Business-driven explanatory variables

---

## Key Machine Learning Concepts Demonstrated

### 1. Time-Series–Safe Data Splitting

* Data is split **by time**, not randomly
* Training data always precedes test data chronologically
* Prevents look-ahead bias

### 2. Feature Engineering

* Monthly aggregation of daily data
* Lag features (1, 2, 3 months)
* Rolling statistics (3-month mean & standard deviation)
* Seasonal encoding using sine/cosine transforms

### 3. Leakage Prevention (Very Important)

* Lag features use **only past values** (`shift(1)`)
* All preprocessing (imputation, scaling, encoding) is done **inside sklearn pipelines**
* Median imputation is learned **only from training data**

### 4. Baseline vs ML Models

Models trained and compared:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

Evaluation metrics:

* RMSE (Root Mean Squared Error)
* R² Score

---

## Pipeline Overview

1. Load and merge datasets
2. Clean and standardize columns
3. Create date-based features
4. Aggregate daily data to monthly store-level data
5. Engineer lag and rolling features
6. Perform time-based train/test split
7. Build preprocessing pipeline (imputation + scaling + encoding)
8. Train multiple regression models
9. Evaluate and compare performance

---

## Data Leakage Audit

This pipeline was explicitly designed to **avoid data leakage**:

✔ Time-based splitting prevents future data access
✔ Lag features use historical values only
✔ Rolling features exclude current month
✔ Imputation and scaling are fit on training data only
✔ One-hot encoding handles unseen stores safely

Minor aggregation choices were made carefully and do not introduce target leakage.

---

## Results (Example)

The final output is a comparison table showing RMSE and R² for each model, allowing clear evaluation against baselines.

Tree-based models (Random Forest, XGBoost) generally outperform linear models due to non-linear relationships and interactions.

---

## Skills Highlighted

* Python (pandas, numpy)
* Scikit-learn Pipelines
* Time-series feature engineering
* Regression modeling
* Model evaluation & comparison
* ML best practices (leakage prevention)

---

## Possible Extensions

* TimeSeriesSplit cross-validation
* Hyperparameter tuning with GridSearchCV
* Feature importance analysis
* Model explainability (SHAP)
* Deployment as a forecasting service

---

##  Author

This project was created as part of a **machine learning portfolio** to demonstrate applied ML skills on real-world, open data.

---

## License

This project uses publicly available data for educational and portfolio purposes only.

