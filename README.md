# European Flight Price Prediction

Supervised **regression** project that predicts intra-European economy-class airfares (EUR) from route, carrier, schedule, and booking-window signals. The workflow covers full exploratory analysis, preprocessing, feature engineering, seven scikit-learn models, and a comparative benchmark.

**Author:** Samvel Danielyan

---

## Problem statement

Airline ticket pricing is a continuous prediction task: given historical Google Flights scrape data, estimate **`price` (EUR)** for unseen route and booking scenarios.

**Business goal:** quantify how carrier choice, stops, advance purchase (`days_left`), trip duration, airports, and calendar effects relate to fare levels — useful for pricing awareness and revenue-style analytics.

**Task type:** Regression (not classification). Evaluation uses **MAE**, **MSE**, **RMSE**, and **R²** — not confusion matrices.

---

## Dataset

| Item | Detail |
|------|--------|
| **File** | `europe_flights_google_prices.csv` |
| **Source** | [Oleg Zay — European Flight Prices (Google Flights)](https://www.kaggle.com/datasets) *(Google Flights scrape)* |
| **Size** | ~200,000 rows × 20 columns |
| **Target** | `price` (EUR) |
| **Scope** | Intra-European economy flights, May–June 2026 booking window |

**Key columns:** `airline`, `stops`, `days_left`, `duration`, `source_airport`, `destination_airport`, `departure_daypart`, `departure_day_of_week`, and related route/time fields.

---

## Repository structure

```
european-flight-price-prediction/
├── europe_flights_google_prices.csv   # Raw dataset
├── flight_pricing_analysis.ipynb      # Full analysis & modeling notebook
└── README.md
```

---

## Methodology

Analysis is organized in **`flight_pricing_analysis.ipynb`** in four sections:

### Section 1 — Task selection & problem definition
Problem framing, success criteria, and regression rationale.

### Section 2 — Data exploration & preparation
| Step | Action |
|------|--------|
| EDA | Shape, dtypes, `describe()`, correlation heatmap & scatter plots |
| Missing values | Detection, imputation (median / mode) |
| Outliers | Price boxplot + IQR-based removal on `price` |
| Feature engineering | `duration_hours`, `Is_Last_Minute`, `Is_Weekend`, `estimated_distance_km`, `Price_per_km` |
| Encoding | Top-category grouping for airline/airports + one-hot encoding on **low-cardinality** columns only (`df_encoded`) |

High-cardinality fields (`flight`, timestamps, `scraped_at`) are **excluded** from one-hot encoding to avoid sparse high-dimensional matrices.

### Section 3 — Model training & evaluation
Train/test split (80/20), `StandardScaler` on feature matrix, then:

| # | Model |
|---|--------|
| 1 | Linear Regression |
| 2 | Lasso Regression |
| 3 | K-Nearest Neighbors (k tuned on subsample) |
| 4 | Weighted KNN (distance-weighted) |
| 5 | Decision Tree Regressor |
| 6 | Random Forest Regressor |
| 7 | Gradient Boosting Regressor |
| 8 | **Hyperparameter tuning** — `GridSearchCV` on Random Forest (5-fold CV) |

Each model includes test metrics plus **actual vs. predicted** and **residual** plots.

**Hyperparameter tuning (Section 3.8):** `GridSearchCV` searches `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` for Random Forest. The tuned model replaces the default RF in the final benchmark, with a before/after comparison chart.

### Section 4 — Comprehensive benchmark
Side-by-side comparison table and bar charts for **MAE**, **MSE**, **RMSE**, and **R²** across all models, with a short analytical interpretation.

---

## Getting started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the project

1. Clone or download this repository.
2. Place `europe_flights_google_prices.csv` in the project root (already included if you cloned with data).
3. Start Jupyter from the project folder:

```bash
jupyter notebook flight_pricing_analysis.ipynb
```

4. **Kernel → Restart & Run All**, or run cells top-to-bottom in order:
   - Section 2 through encoding must finish before Section 3.
   - After encoding, expect `df_encoded` with ~**200 columns** (not 100k+).
   - Section 3 scaling should print train/test shapes with ~**178** features.

---

## Engineering notes

- **`df`** — cleaned dataframe after imputation, outliers, and feature engineering (for EDA).
- **`df_model` / `df_encoded`** — modeling tables after grouping and one-hot encoding.
- **`Price_per_km`** is kept for correlation EDA but dropped from model features (target leakage).
- Tree-based models (Decision Tree, Random Forest, Gradient Boosting) typically perform strongly on mixed numeric + dummy features; linear models provide interpretable coefficients.

---

## Expected deliverables

- Reproducible Jupyter notebook with EDA, preprocessing, and modeling
- Seven trained regression models with consistent evaluation
- Comparative benchmark and interpretation of best performers
- Visual diagnostics: correlation plots, price distributions, actual vs. predicted, residuals

---

## Limitations

- Prices reflect a **fixed scrape window** (Google Flights, May 2026); models do not forecast future market shifts.
- Duration is used as a **distance proxy**, not true great-circle route length.
- Carrier/airport grouping (`top-N` + `Other`) trades granularity for stable encoding.
- Results depend on running the full pipeline in order after a **kernel restart** if cells were re-run out of sequence.

---

## Acknowledgments

- Dataset: **Oleg Zay — European Flight Prices**
- Course project: supervised learning / data mining — European intra-continental airfare analysis

---

## License

This repository is submitted as an academic coursework project. Check your institution's policy before reusing the dataset or notebook externally.
