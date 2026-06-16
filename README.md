# Regression Suite — Ames Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)
![Status](https://img.shields.io/badge/Status-Phase%201%20Complete-brightgreen)

End-to-end machine learning regression pipeline built on the Ames Housing dataset 
(Kaggle: House Prices - Advanced Regression Techniques). Covers the full workflow 
from exploratory data analysis to model evaluation and interpretation.

---

## Project Structure

```text
regression-suite/
├── housing_analysis.ipynb       # Main notebook — EDA to model evaluation
├── best_model_pipeline.joblib   # Saved best model pipeline
├── train.csv                    # Dataset (Kaggle)
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── screenshots/                 # Visual outputs
    ├── target_distribution.png
    ├── missing_data.png
    ├── model_comparison.png
    ├── residuals_analysis.png
    └── feature_coefficients.png
```

---

## Dataset

- **Source**: [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size**: 1,460 rows, 81 columns
- **Target**: SalePrice (residential home sale price in Ames, Iowa)
- **Features**: Numeric and categorical — covering lot size, neighborhood, construction quality, basement, garage, and more.

---

## Quick Start

```bash
git clone https://github.com/Rehanku/regression-suite.git
cd regression-suite
pip install -r requirements.txt
jupyter notebook housing_analysis.ipynb
```
*Note: The dataset (`train.csv`) is not included in the repo. Download it from Kaggle and place it in the project root.*

---

## Key Visuals

### Target Variable Distribution
![Target Distribution](screenshots/target_distribution.png)

### Missing Data Pattern
![Missing Data](screenshots/missing_data.png)

### Model Comparison (Test R² & RMSE)
![Model Comparison](screenshots/model_comparison.png)

### Residual Analysis
![Residual Analysis](screenshots/residuals_analysis.png)

### Feature Coefficients (Top 20)
![Feature Coefficients](screenshots/feature_coefficients.png)

---

## Workflow

### 1. Exploratory Data Analysis
* Shape, dtypes, and statistical summary across all 81 columns.
* Missing data analysis — 19 columns with missing values identified and visualized.
* Target variable distribution — SalePrice confirmed right-skewed (skewness = 1.88).
* Log transformation applied to normalize target (post-transform skewness = 0.12).
* Correlation analysis across all numeric features.
* Multicollinearity check — redundant features removed based on inter-feature correlation threshold of 0.7.
* Outlier detection using IQR — 31 outliers removed from GrLivArea.

### 2. Feature Engineering
Created 7 domain-driven features from existing columns:

| Feature | Formula | Rationale |
|---|---|---|
| **TotalSF** | TotalBsmtSF + 1stFlrSF + 2ndFlrSF | Total living area across all floors |
| **HouseAge** | YrSold - YearBuilt | Age of house at time of sale |
| **RemodAge** | YrSold - YearRemodAdd | Years since last remodel |
| **HasPool** | PoolArea > 0 | Binary presence flag |
| **HasGarage** | GarageArea > 0 | Binary presence flag |
| **Has2ndFloor** | 2ndFlrSF > 0 | Binary presence flag |
| **HasBsmt** | TotalBsmtSF > 0 | Binary presence flag |

### 3. Preprocessing Pipeline
Built using scikit-learn `ColumnTransformer` + `Pipeline`:
* **Numeric features (12):** Median imputation → StandardScaler
* **Categorical features (6):** Mode imputation → OneHotEncoder
* *No data leakage — all transformations learned on training set only.*

---

## Models Compared

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Lasso | 0.0911 | 0.1263 | 0.8872 |
| Linear Regression | 0.0920 | 0.1273 | 0.8854 |
| ElasticNet | 0.0909 | 0.1275 | 0.8851 |
| Ridge | 0.0912 | 0.1280 | 0.8841 |
| Gradient Boosting | 0.0934 | 0.1335 | 0.8740 |
| Polynomial (deg 2) | 0.0987 | 0.1411 | 0.8591 |

*\*Metrics reported on log(SalePrice). All models trained on 80/20 train-test split.*

### Cross-Validation Results (5-Fold)

| Model | CV R² Mean | CV R² Std |
|---|---|---|
| **Ridge (Best)** | **0.8757** | **0.0122** |
| ElasticNet | 0.8756 | 0.0125 |
| Lasso | 0.8751 | 0.0147 |
| Linear Regression | 0.8728 | 0.0149 |
| Gradient Boosting | 0.8637 | 0.0121 |
| Polynomial (deg 2) | 0.8454 | 0.0229 |

*Selected based on highest mean CV R² and lowest standard deviation among top performers, indicating both strong performance and stability across folds.*

---

## Residual Analysis
* Residuals show random scatter around zero — no systematic bias detected.
* Residual distribution approximately normal.
* Minor heteroscedasticity at higher predicted values — expected given the 18-feature model does not capture all price drivers for luxury properties.

---

## Key Insights
> "A house’s overall quality rating alone can explain ~67% of its price variation. After controlling for size and neighbourhood, a pool adds almost no value in Ames, Iowa. Regularized linear models outperform more complex ones — the relationships are largely linear after proper feature engineering."

* Overall quality rating is the single strongest predictor of sale price (r = 0.82).
* Total square footage across all floors outperforms individual floor area features.
* Neighborhood and zoning contribute meaningful signal beyond numeric features.
* Houses with no basement or garage sell at a significant discount even after controlling for size.
* Regularized linear models (Ridge, Lasso, ElasticNet) outperform Polynomial and Gradient Boosting on this dataset.

---

## Tech Stack
* Python 3.x
* pandas, numpy, scipy
* scikit-learn
* matplotlib, seaborn
* joblib

---

## Roadmap (Phase 2)
- [ ] FastAPI prediction endpoint (`POST /predict`)
- [ ] Docker containerization
- [ ] MLflow experiment tracking
- [ ] GitHub Actions CI/CD pipeline

---

## Author
**Rehan Khan** [GitHub Profile](https://github.com/Rehanku)