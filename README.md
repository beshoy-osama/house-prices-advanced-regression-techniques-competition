# 🏠 House Prices — Advanced Regression Techniques

> **Kaggle Competition** · Faculty of Computers and Artificial Intelligence · Machine Learning Project 2026  
> **Author:** Eng. Beshoy Osama

A complete machine learning pipeline to predict residential property sale prices in Ames, Iowa, using 79 explanatory variables. The project covers exploratory data analysis, preprocessing, feature engineering, and multi-model evaluation.

---

## 📊 Results at a Glance

| Model | Train RMSLE | CV RMSLE | CV Std | Kaggle Public |
|---|---|---|---|---|
| SVR (Baseline) | 0.0769 | 0.1872 | 0.0175 | — |
| SVR (Tuned) | 0.0907 | 0.1154 | 0.0067 | 0.14023 |
| Linear Regression | 0.0943 | ❌ Unstable | — | — |
| Lasso | 0.0958 | 0.1157 | 0.0071 | 0.13981 |
| **XGBoost** | **0.0806** | **0.1138** | **0.0079** | **0.13115** ✅ |

**Best model:** XGBoost · CV RMSLE `0.1138` · Kaggle Public `0.13115` (est. Top 30%)

---

## 📁 Project Structure

```
house-prices/
├── house_prices_notebook-b.ipynb   # Full pipeline: EDA → preprocessing → modeling
├── house_prices_report.pdf         # Detailed project report with figures
├── train.csv                       # Training data (1,460 rows, 81 features)
├── test.csv                        # Test data (1,459 rows, 80 features)
├── submission_svr_tuned.csv        # Kaggle submission — SVR Tuned
├── submission_lasso.csv            # Kaggle submission — Lasso
└── submission_xgboost.csv          # Kaggle submission — XGBoost (best)
```

---

## 📦 Requirements

**Python 3.8+**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading and manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | SVR, Linear Regression, Lasso, preprocessing, cross-validation |
| `xgboost` | Gradient boosted trees |

---

## 🚀 Quick Start

1. **Clone the repository and install dependencies**

   ```bash
   git clone <repo-url>
   cd house-prices
   pip install -r requirements.txt
   ```

2. **Download the dataset** from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place `train.csv` and `test.csv` in the project root.

3. **Run the notebook**

   ```bash
   jupyter notebook house_prices_notebook-b.ipynb
   ```

4. **Submit to Kaggle** — uncomment the submission cells at the end of the notebook and upload the generated `.csv` file.

---

## 🔍 Dataset

| Split | Rows | Columns | Target |
|---|---|---|---|
| Train | 1,460 | 81 | `SalePrice` |
| Test | 1,459 | 80 | — (held out by Kaggle) |

**Source:** [kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

Features span three types:

- **Categorical (43):** Neighborhood, MSZoning, GarageType, RoofStyle, …
- **Integer (35):** OverallQual, YearBuilt, GrLivArea, BedroomAbvGr, …
- **Float (3):** LotFrontage, MasVnrArea, GarageYrBlt

---

## 🧪 Exploratory Data Analysis

### Target Variable
The raw `SalePrice` distribution is right-skewed (skewness ≈ 1.88). A `log1p` transform reduces skewness to ≈ 0.12, producing a near-normal distribution required by several regression models.

### Missing Values
The training set has **7,829 missing values** across 19 columns. Columns with more than 40% missing data were dropped:

| Column | Missing % | Action |
|---|---|---|
| PoolQC | 99.5% | Dropped |
| MiscFeature | 96.3% | Dropped |
| Alley | 93.8% | Dropped |
| Fence | 80.8% | Dropped |
| MasVnrType | ~59% | Dropped |
| FireplaceQu | ~47% | Dropped |

### Top Correlated Features

| Feature | Correlation with SalePrice |
|---|---|
| OverallQual | 0.79 |
| GrLivArea | 0.71 |
| GarageCars | ~0.64 |
| GarageArea | ~0.62 |
| TotalBsmtSF | ~0.61 |

> ⚠️ GarageArea and GarageCars are highly collinear (r = 0.88), motivating the use of regularized models.

---

## ⚙️ Preprocessing Pipeline

All preprocessing was applied to the **combined** train + test set to ensure consistent encoding.

| Step | Action |
|---|---|
| 1 | Remove 2 outlier rows (GrLivArea > 4,000 sqft with SalePrice < $200k) |
| 2 | Log-transform target: `y = log1p(SalePrice)` |
| 3 | Drop columns with > 40% missing values |
| 4 | Impute `LotFrontage` using per-Neighborhood median |
| 5 | Impute remaining categorical NAs with column mode |
| 6 | Impute remaining numerical NAs with column median |
| 7 | Engineer 6 new features (see table below) |
| 8 | Drop 14 source columns used for feature engineering |
| 9 | Ordinal encode quality/condition columns (None=0 → Ex=5) |
| 10 | One-hot encode all remaining categoricals → 228 total features |
| 11 | `StandardScaler` — fit on train, transform on test |

### Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `TotalSF` | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | Total living area across all floors |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtFullBath + … | Weighted bathroom count |
| `TotalPorchSF` | Sum of all porch columns | Aggregate outdoor living space |
| `HouseAge` | YrSold − YearBuilt | Age of house at time of sale |
| `RemodAge` | YrSold − YearRemodAdd | Time since last renovation |
| `IsRemodeled` | 1 if YearBuilt ≠ YearRemodAdd else 0 | Binary flag for remodelled homes |

---

## 🤖 Models

All models predict `log(SalePrice)` and are evaluated via **5-fold cross-validated RMSLE**. Predictions are exponentiated back to dollar values for submission.

### SVR (Support Vector Regression)
- **Baseline** (sklearn defaults, C=1): CV RMSLE = 0.1872
- **Tuned** (GridSearchCV — best: C=10, gamma=0.0001, epsilon=0.01): CV RMSLE = **0.1154**
- A 38% improvement from tuning alone. `StandardScaler` is critical for SVM performance.

### Linear Regression
- Train RMSLE = 0.0943, but CV RMSLE collapses (~1.37 trillion) due to near-singular design matrices in high-dimensional data (228 features, ~1,458 samples).
- **Fix:** Replace OLS with Ridge Regression (L2 penalty).

### Lasso (L1 Regularisation)
- GridSearchCV selected `alpha=0.001`.
- Performs automatic feature selection by shrinking less-important coefficients to zero.
- CV RMSLE = **0.1157** — comparable to Tuned SVR.

### XGBoost
- Does not require feature scaling; handles sparse one-hot data natively.
- Best config: `n_estimators=1500`, `learning_rate=0.05`, `max_depth=3`, `colsample_bytree=0.5`, `gamma=0.05`, `reg_alpha=0.01`, `min_child_weight=3`.
- CV RMSLE = **0.1138** — best across all models.

---

## 📈 Evaluation Metric

**RMSLE — Root Mean Squared Logarithmic Error**

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\log(1 + \hat{y}_i) - \log(1 + y_i)\right)^2}$$

RMSLE penalises under-predictions more than over-predictions and is well-suited to price data spanning orders of magnitude.

---

## 🗝️ Key Findings

- **Hyperparameter tuning** was the single largest improvement: tuning SVR delivered a 38% RMSLE reduction without changing any preprocessing.
- **XGBoost is the best model** on both CV and Kaggle public scores.
- A **consistent gap of ~0.015–0.025** exists between CV RMSLE and Kaggle public scores across all models, likely due to minor distributional differences between the train and test sets.
- **OLS Linear Regression fails** on high-dimensional data; Ridge Regression is the appropriate replacement.
- **SVR Tuned and Lasso generalise well**, with tight overfit gaps of ~0.02–0.025.

---

## 📜 License

This project is submitted as an academic assignment for the Faculty of Computers and Artificial Intelligence, 2026. The dataset is provided by Kaggle under their [competition rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules).
