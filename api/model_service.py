from pathlib import Path

import joblib
import pandas as pd


FEATURE_COLUMNS = [
    "Return_1D",
    "Return_5D",
    "Return_10D",
    "Return_20D",
    "Volatility_5D",
    "Volatility_10D",
    "Volatility_20D",
    "Volume_Change_1D",
    "Relative_Volume_20D",
    "Price_vs_MA_5D",
    "Price_vs_MA_20D",
    "SPY_Return_1D",
    "QQQ_Return_1D",
    "SMH_Return_1D",
    "SPY_Return_5D",
    "QQQ_Return_5D",
    "SMH_Return_5D",
    "Excess_vs_QQQ_1D",
    "Excess_vs_SMH_1D",
]
MODEL_COLUMNS = FEATURE_COLUMNS + ["Ticker"]

CLASSIFICATION_THRESHOLD = 0.5
HORIZON_TRADING_DAYS = 5
MOVE_THRESHOLD = 0.03
MODEL_NAME = "LogisticRegression"

MODEL_FILE = Path(__file__).resolve().parent / "artifacts" / "logistic_regression_candidate.joblib"

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Champion model not found: {MODEL_FILE}")

model = joblib.load(MODEL_FILE)


def predict_large_move(payload: dict) -> dict:
    row = pd.DataFrame([payload], columns=MODEL_COLUMNS)
    probability = float(model.predict_proba(row)[0, 1])

    return {
        "ticker": payload["Ticker"],
        "large_move_probability": probability,
        "predicted_large_move": probability >= CLASSIFICATION_THRESHOLD,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "model_name": MODEL_NAME,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "move_threshold": MOVE_THRESHOLD,
    }
