import torch

DATA_CONFIG = {
    "start_date": "2010-01-01",
    "end_date": "2026-04-14",
    "parquet_path": "data/eurusd_d1_features.parquet",
    "raw_parquet_path": "data/eurusd_d1_raw.parquet",
    "lookback": 1,
    "barrier_tp": 0.010,
    "barrier_sl": 0.005,
    "barrier_max_bars": 20,
    "no_trade_buy_above": 0.60,
    "no_trade_sell_below": 0.45,
    "atr_period": 14,
    "tp_sl_ratio": 3.5,
    "min_sl_pips": 40.0,
    "min_adx": 0.30,
    "timeframe": "1d",
    "session_filter": False,
    "session_start_utc": 7,
    "session_end_utc": 20,
    "trend_filter": False,
    "entry_on_open": False,
    "trailing_stop": True,
    "breakeven_pips": 15,
    "trail_pips": 10,
    "max_hold_bars": 10,
}

FEATURE_COLUMNS = [
    "d1_log_return", "d1_ret_3", "d1_ret_5", "d1_ret_10", "d1_ret_20",
    "dxy_momentum",
    "atr_norm", "adx", "rsi",
    "macd_hist",
    "body_ratio", "lower_shadow_pct",
    "vol_sma_ratio",
    "trend_ma20", "trend_ma50",
    "vwap_ratio",
    "gbpusd_ret_1", "gbpusd_ret_5", "gbpusd_corr_20",
    "usdjpy_ret_1", "usdjpy_ret_5", "usdjpy_corr_20",
    "h4_ret_last", "h4_ret_3d",
    "h4_range_pct", "h4_body_pct",
]

INPUT_DIM = len(FEATURE_COLUMNS)

FEATURE_WEIGHTS = torch.tensor([1.0] * INPUT_DIM)

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
    "hidden_dim": 32,
    "n_layers": 2,
    "dropout": 0.30,
    "n_heads": 4,
    "temperature": 0.35,
}

TRAIN_CONFIG = {
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 1.5e-3,
    "weight_decay": 5e-4,
    "patience": 30,
    "n_splits": 5,
    "purge_gap": 2,
    "focal_alpha": 0.50,
    "focal_gamma": 2.5,
    "label_sharpening": 0.15,
    "walk_forward_months": 12,
    "val_months": 3,
    "step_months": 3,
}

RISK_CONFIG = {
    "risk_per_trade": 0.02,
    "max_leverage": 50.0,
    "initial_equity": 10000.0,
    "pip_value_per_lot": 10.0,
    "lot_step": 0.01,
    "min_lots": 0.01,
    "contract_size": 100000,
    "spread_pips": 1.0,
    "commission_per_lot": 3.5,
    "compound": False,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKTEST_CONFIG = {
    "start_date": "2024-01-01",
    "end_date": "2026-04-14",
}

MODEL_SAVE_PATH = "models/forex_predictor.pth"
SCALER_SAVE_PATH = "models/scaler.pkl"

CTRADER_CONFIG = {
    "host": "demo.ctraderapi.com",
    "port": 5035,                           
    "client_id": "25719_p6KKm1idyxlgQmGIna3j0Vx3E3yaSkmUd2IE5HfVjogt59hkDY",
    "client_secret": "VGcTNNc0SlPQ3bAVktCWQnsQcVW5RNlEVit14EsjZtolEx4RTz",
    "access_token": "Wf0y4aaLLsbvaMRvlQnyp_uWixAIC2lQEG4iIIwICKI",
    "refresh_token": "UczvFq7BsK99J9PNxghS-18kGTyBeV1B1dkEXBplrAc",
    "account_id": 46995819,                  
    "symbol": "EURUSD",
}