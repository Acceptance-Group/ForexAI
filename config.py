import torch

DATA_CONFIG = {
    "start_date": "2010-01-01",
    "parquet_path": "data/eurusd_features.parquet",
    "lookback": 21,
    "barrier_tp": 0.008,
    "barrier_sl": 0.008,
    "barrier_max_days": 10,
    "temperature": 1.0,
    "no_trade_low": 0.47,
    "no_trade_high": 0.53,
    "atr_period": 14,
    "atr_filter_quantile": 0.05,
    "tp_sl_ratio": 3.5,
    "trend_filter_ma": 0,
}

RAW_FEATURES = ["EUR_USD", "EUR_USD_High", "EUR_USD_Low", "EUR_USD_Volume", "DXY", "VIX", "T10Y2Y", "FEDFUNDS"]

FEATURE_COLUMNS = [
    "eur_log_return",
    "eur_ret_3",
    "eur_ret_5",
    "dxy_log_return",
    "dxy_ret_5",
    "dxy_ret_20",
    "fedfunds_diff",
    "relative_strength",
    "atr_norm",
    "macd_hist",
    "bb_pctb",
    "adx",
    "bb_width",
    "ma_50_ratio",
    "ma_200_ratio",
    "dxy_momentum",
]

INPUT_DIM = len(FEATURE_COLUMNS)

FEATURE_WEIGHTS = torch.tensor([
    2.0,  # eur_log_return
    1.5,  # eur_ret_3
    1.2,  # eur_ret_5
    2.0,  # dxy_log_return
    1.2,  # dxy_ret_5
    1.3,  # dxy_ret_20
    1.0,  # fedfunds_diff
    1.5,  # relative_strength
    1.0,  # atr_norm
    1.5,  # macd_hist
    1.5,  # bb_pctb
    1.5,  # adx
    1.5,  # bb_width
    1.5,  # ma_50_ratio
    1.5,  # ma_200_ratio
    1.3,  # dxy_momentum
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
    "dropout": 0.20,
    "n_heads": 4,
    "temperature": 1.0,
}

TRAIN_CONFIG = {
    "epochs": 400,
    "batch_size": 256,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "patience": 50,
    "n_splits": 5,
    "purge_gap": 5,
    "focal_alpha": 0.50,
    "focal_gamma": 2.0,
    "label_sharpening": 0.10,
}

RISK_CONFIG = {
    "risk_per_trade": 0.02,
    "max_position_fraction": 2.0,
    "initial_equity": 10000.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKTEST_CONFIG = {
    "start_date": "2023-01-01",
    "end_date": "2026-01-01",
}

MODEL_SAVE_PATH = "models/forex_predictor.pth"
SCALER_SAVE_PATH = "models/scaler.pkl"