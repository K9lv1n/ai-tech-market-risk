from typing import Literal

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    Return_1D: float
    Return_5D: float
    Return_10D: float
    Return_20D: float
    Volatility_5D: float
    Volatility_10D: float
    Volatility_20D: float
    Volume_Change_1D: float
    Relative_Volume_20D: float
    Price_vs_MA_5D: float
    Price_vs_MA_20D: float
    SPY_Return_1D: float
    QQQ_Return_1D: float
    SMH_Return_1D: float
    SPY_Return_5D: float
    QQQ_Return_5D: float
    SMH_Return_5D: float
    Excess_vs_QQQ_1D: float
    Excess_vs_SMH_1D: float
    Ticker: Literal["NVDA", "AMD", "MSFT", "GOOGL", "META"]


class PredictionResponse(BaseModel):
    ticker: str
    large_move_probability: float
    predicted_large_move: bool
    classification_threshold: float
    model_name: str
    horizon_trading_days: int
    move_threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_family: str
    feature_count: int
    ticker_universe: list[str]
    horizon_trading_days: int
    move_threshold: float
    validation_roc_auc: float
    validation_pr_auc: float
