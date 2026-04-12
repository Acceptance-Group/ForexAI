import torch

DATA_CONFIG = {
    "start_date": "2005-01-01",
    "parquet_path": "data/eurusd_features.parquet",
    "lookback": 21,
    "barrier_tp": 0.008,
    "barrier_sl": 0.008,
    "barrier_max_days": 10,
    "temperature": 5.0,
    "trade_threshold": 0.505,
    "atr_period": 14,
    "atr_filter_quantile": 0.25,
}

RAW_FEATURES = ["EUR_USD", "DXY", "VIX", "T10Y2Y", "FEDFUNDS"]

FEATURE_COLUMNS = [
    "eur_log_return",
    "eur_ret_3",
    "eur_ret_5",
    "eur_ret_10",
    "dxy_log_return",
    "dxy_ret_3",
    "dxy_ret_5",
    "dxy_ret_10",
    "dxy_ret_20",
    "vix_log_return",
    "t10y2y_diff",
    "fedfunds_diff",
    "relative_strength",
    "dxy_momentum",
    "atr_14",
]

INPUT_DIM = len(FEATURE_COLUMNS)

FEATURE_WEIGHTS = torch.tensor([
    2.0,
    1.5,
    1.2,
    1.0,
    2.0,
    1.5,
    1.2,
    1.0,
    1.3,
    1.0,
    1.0,
    1.0,
    1.5,
    1.5,
    1.0,
])

YFINANCE_TICKERS = {
    "EUR_USD": "EURUSD=X",
    "DXY": "DX-Y.NYB",
    "VIX": "^VIX",
}

FRED_SERIES = {
    "T10Y2Y": "T10Y2Y",
    "FEDFUNDS": "FEDFUNDS",
}

MODEL_CONFIG = {
    "input_dim": INPUT_DIM,
    "hidden_dim": 64,
    "n_layers": 2,
    "dropout": 0.15,
    "n_heads": 4,
    "temperature": 5.0,
}

TRAIN_CONFIG = {
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 3e-3,
    "weight_decay": 1e-4,
    "patience": 35,
    "n_splits": 5,
    "purge_gap": 5,
    "focal_alpha": 0.55,
    "focal_gamma": 2.5,
    "label_sharpening": 0.15,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKTEST_CONFIG = {
    "start_date": "2024-01-01",
    "end_date": "2026-01-01",
}

MODEL_SAVE_PATH = "models/forex_predictor.pth"
SCALER_SAVE_PATH = "models/scaler.pkl"