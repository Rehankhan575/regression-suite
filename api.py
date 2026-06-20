from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Regression Suite — Ames Housing Price Prediction",
    description="Predicts house sale prices using a Ridge Regression model trained on the Ames Housing dataset.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Load pipeline ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_pipeline.joblib")
pipeline = joblib.load(MODEL_PATH)

# ── Input schema ─────────────────────────────────────────────────────────────
class HouseFeatures(BaseModel):
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality (1-10)")
    TotalSF: float = Field(..., ge=0, le=15000, description="Total square footage across all floors")
    GrLivArea: float = Field(..., ge=0, le=6000, description="Above grade living area in sq ft")
    GarageCars: int = Field(..., ge=0, le=5, description="Garage capacity in car units")
    TotalBsmtSF: float = Field(..., ge=0, le=6000, description="Total basement area in sq ft")
    FullBath: int = Field(..., ge=0, le=5, description="Number of full bathrooms")
    YearBuilt: int = Field(..., ge=1872, le=2010, description="Year house was built")
    HouseAge: int = Field(..., ge=0, le=150, description="Age of house at time of sale")
    RemodAge: int = Field(..., ge=0, le=100, description="Years since last remodel")
    HasGarage: int = Field(..., ge=0, le=1, description="Has garage (1=yes, 0=no)")
    HasBsmt: int = Field(..., ge=0, le=1, description="Has basement (1=yes, 0=no)")
    Has2ndFloor: int = Field(..., ge=0, le=1, description="Has second floor (1=yes, 0=no)")
    Neighborhood: str = Field(..., description="Neighborhood name")
    MSZoning: str = Field(..., description="General zoning classification")
    SaleCondition: str = Field(..., description="Condition of sale")
    BldgType: str = Field(..., description="Type of dwelling")
    HouseStyle: str = Field(..., description="Style of dwelling")
    RoofStyle: str = Field(..., description="Type of roof")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "OverallQual": 7,
                    "TotalSF": 2800,
                    "GrLivArea": 1800,
                    "GarageCars": 2,
                    "TotalBsmtSF": 1000,
                    "FullBath": 2,
                    "YearBuilt": 2003,
                    "HouseAge": 7,
                    "RemodAge": 7,
                    "HasGarage": 1,
                    "HasBsmt": 1,
                    "Has2ndFloor": 1,
                    "Neighborhood": "CollgCr",
                    "MSZoning": "RL",
                    "SaleCondition": "Normal",
                    "BldgType": "1Fam",
                    "HouseStyle": "2Story",
                    "RoofStyle": "Gable"
                }
            ]
        }
    }

# ── Output schema ─────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    price_formatted: str
    confidence_note: str
    model_info: dict

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "Ridge Regression", "dataset": "Ames Housing"}

# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    try:
        input_df = pd.DataFrame([features.model_dump()])

        log_prediction = pipeline.predict(input_df)[0]
        predicted_price = float(np.exp(log_prediction))

        # 95% confidence interval — model RMSE is 0.128 in log space
        rmse_log = 0.128
        price_low = float(np.exp(log_prediction - 1.96 * rmse_log))
        price_high = float(np.exp(log_prediction + 1.96 * rmse_log))

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            price_range_low=round(price_low, 2),
            price_range_high=round(price_high, 2),
            price_formatted=f"${predicted_price:,.0f}",
            confidence_note=f"95% confidence interval: ${price_low:,.0f} — ${price_high:,.0f}",
            model_info={
                "model": "Ridge Regression",
                "cv_r2": 0.8757,
                "test_r2": 0.8841,
                "dataset": "Ames Housing — 1,429 samples after cleaning"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Ames Housing Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict"
    }