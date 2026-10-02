import json
from pathlib import Path

from fastapi import FastAPI

from model_service import (
    FEATURE_COLUMNS,
    HORIZON_TRADING_DAYS,
    MODEL_NAME,
    MOVE_THRESHOLD,
    model,
    predict_large_move,
)
from schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)


MODEL_METADATA_FILE = Path(__file__).resolve().parent / "artifacts" / "model_metadata.json"

if not MODEL_METADATA_FILE.exists():
    raise FileNotFoundError(f"Model metadata not found: {MODEL_METADATA_FILE}")

with MODEL_METADATA_FILE.open(encoding="utf-8") as file:
    MODEL_METADATA = json.load(file)


app = FastAPI(
    title="AI Tech Market Risk API",
    version="0.1.0",
    description=(
        "Predicts the probability that a tracked AI/technology stock experiences "
        "an absolute move greater than 3% over the next five trading days."
    ),
)


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> dict:
    return {
        "model_name": MODEL_NAME,
        "model_family": MODEL_METADATA["model_family"],
        "feature_count": len(FEATURE_COLUMNS),
        "ticker_universe": MODEL_METADATA["ticker_universe"],
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "move_threshold": MOVE_THRESHOLD,
        "validation_roc_auc": MODEL_METADATA["validation_roc_auc"],
        "validation_pr_auc": MODEL_METADATA["validation_pr_auc"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> dict:
    return predict_large_move(request.model_dump())
