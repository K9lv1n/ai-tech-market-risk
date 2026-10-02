import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


API_DIR = Path(__file__).resolve().parents[1]
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app import app
from model_service import FEATURE_COLUMNS, MODEL_COLUMNS, MODEL_FILE


client = TestClient(app)


def make_sample(ticker: str = "NVDA") -> dict:
    sample = {feature: 0.0 for feature in FEATURE_COLUMNS}
    sample["Relative_Volume_20D"] = 1.0
    sample["Ticker"] = ticker
    return sample


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_matches_direct_model() -> None:
    sample = make_sample()
    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    model = joblib.load(MODEL_FILE)
    direct_input = pd.DataFrame([sample], columns=MODEL_COLUMNS)
    direct_probability = float(model.predict_proba(direct_input)[0, 1])
    api_probability = float(response.json()["large_move_probability"])

    assert np.isclose(direct_probability, api_probability, rtol=1e-12, atol=1e-12)


def test_invalid_ticker_is_rejected() -> None:
    response = client.post("/predict", json=make_sample(ticker="AAPL"))

    assert response.status_code == 422
