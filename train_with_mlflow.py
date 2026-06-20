import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import time
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── MLflow setup — SQLite backend ──────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("regression-suite-ames-housing")

# ── Data loading and preprocessing ─────────────────────────────────────────
df = pd.read_csv("train.csv")

# Log transform target
df["LogSalePrice"] = np.log(df["SalePrice"])

# Feature engineering
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
df["HasPool"] = (df["PoolArea"] > 0).astype(int)
df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)
df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)

# Outlier removal
Q1 = df["GrLivArea"].quantile(0.25)
Q3 = df["GrLivArea"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["GrLivArea"] >= Q1 - 1.5*IQR) & 
        (df["GrLivArea"] <= Q3 + 1.5*IQR)].copy()

# Features
numeric_features = ["OverallQual", "TotalSF", "GrLivArea", "GarageCars",
                    "TotalBsmtSF", "FullBath", "YearBuilt", "HouseAge",
                    "RemodAge", "HasGarage", "HasBsmt", "Has2ndFloor"]

categorical_features = ["Neighborhood", "MSZoning", "SaleCondition",
                        "BldgType", "HouseStyle", "RoofStyle"]

X = df[numeric_features + categorical_features]
y = df["LogSalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Preprocessor ────────────────────────────────────────────────────────────
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ── Model definitions ────────────────────────────────────────────────────────
models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },
    "Ridge": {
        "model": Ridge(alpha=10.0),
        "params": {"alpha": 10.0}
    },
    "Lasso": {
        "model": Lasso(alpha=0.001, max_iter=10000),
        "params": {"alpha": 0.001}
    },
    "ElasticNet": {
        "model": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        "params": {"alpha": 0.001, "l1_ratio": 0.5}
    },
    "Polynomial_deg2": {
        "model": Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", Ridge(alpha=10.0))
        ]),
        "params": {"degree": 2, "alpha": 10.0}
    },
    "Gradient Boosting": {
        "model": GradientBoostingRegressor(n_estimators=200, random_state=42),
        "params": {"n_estimators": 200}
    }
}

# ── Training loop with MLflow logging ───────────────────────────────────────
best_run_id = None
best_cv_r2 = -np.inf

for model_name, config in models.items():
    with mlflow.start_run(run_name=model_name):

        # Build full pipeline
        if model_name == "Polynomial_deg2":
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", Ridge(alpha=10.0))
            ])
        else:
            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", config["model"])
            ])

        # Log parameters
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_features", len(numeric_features) + len(categorical_features))
        for k, v in config["params"].items():
            mlflow.log_param(k, v)

        # Train
        start = time.time()
        full_pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        # Evaluate on test set
        y_pred = full_pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Cross validation
        cv_scores = cross_val_score(full_pipeline, X, y, cv=5, scoring="r2")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_test", r2)
        mlflow.log_metric("cv_r2_mean", cv_mean)
        mlflow.log_metric("cv_r2_std", cv_std)
        mlflow.log_metric("train_time_seconds", train_time)

        # Log model
        mlflow.sklearn.log_model(
            full_pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype"]
        )

        print(f"{model_name}: MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f} | CV={cv_mean:.4f}±{cv_std:.4f}")

        # Track best model
        if cv_mean > best_cv_r2:
            best_cv_r2 = cv_mean
            best_run_id = mlflow.active_run().info.run_id
            best_model_name = model_name
            best_pipeline = full_pipeline

print(f"\nBest model: {best_model_name} — CV R²: {best_cv_r2:.4f}")

# Save best model pipeline
joblib.dump(best_pipeline, "best_model_pipeline.joblib")
print("Saved: best_model_pipeline.joblib")

# Tag best run in MLflow
with mlflow.start_run(run_id=best_run_id):
    mlflow.set_tag("best_model", "true")
    mlflow.set_tag("deployed", "false")