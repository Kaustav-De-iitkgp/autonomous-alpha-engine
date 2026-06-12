from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import os

app = FastAPI(title="Autonomous Alpha Trading API", version="1.0")

model = xgb.XGBClassifier()
# Path assumes the API is run from the root folder
MODEL_PATH = "models/xgboost_alpha_model.json"

if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)

class MarketFeatures(BaseModel):
    vader_mean: float
    finbert_mean: float
    headline_count: int
    daily_return: float
    volatility_7d: float
    volatility_30d: float
    price_to_sma14: float
    rsi_14: float
    macd_hist: float

@app.get("/")
def health_check():
    return {"status": "Active", "system": "Alpha Engine Online"}

@app.post("/predict")
def generate_signal(features: MarketFeatures):
    try:
        X_input = np.array([[
            features.vader_mean, features.finbert_mean, features.headline_count,
            features.daily_return, features.volatility_7d, features.volatility_30d,
            features.price_to_sma14, features.rsi_14, features.macd_hist
        ]])
        
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0].max()
        
        return {
            "signal": "UP" if prediction == 1 else "DOWN",
            "action": "BUY" if prediction == 1 else "HOLD/SHORT",
            "confidence": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
