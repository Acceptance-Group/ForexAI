import io
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    YFINANCE_TICKERS,
    FRED_SERIES,
    DATA_CONFIG,
    RAW_FEATURES,
    FEATURE_COLUMNS,
)


def fetch_yfinance_data() -> pd.DataFrame:
    frames = {}
    for name, ticker in YFINANCE_TICKERS.items():
        df = yf.download(ticker, start=DATA_CONFIG["start_date"], progress=False)
        close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        frames[name] = close
        if name == "EUR_USD":
            frames["EUR_USD_High"] = df["High"].iloc[:, 0] if isinstance(df["High"], pd.DataFrame) else df["High"]
            frames["EUR_USD_Low"] = df["Low"].iloc[:, 0] if isinstance(df["Low"], pd.DataFrame) else df["Low"]
            frames["EUR_USD_Volume"] = df["Volume"].iloc[:, 0] if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]
    return pd.DataFrame(frames)


def fetch_fred_data() -> pd.DataFrame:
    frames = {}
    for name, series_id in FRED_SERIES.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.rename(columns={df.columns[0]: "date", df.columns[1]: name}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")
        df = df.set_index("date")
        df = df[df.index >= DATA_CONFIG["start_date"]]
        frames[name] = df[name]
    return pd.DataFrame(frames)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bb_pctb(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pctb = (prices - lower) / (upper - lower + 1e-10)
    return pctb


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)
    mask_plus = (high - prev_close) < (prev_close - low)
    mask_minus = (prev_close - low) < (high - prev_close)
    plus_dm[~mask_plus] = 0
    minus_dm[~mask_minus] = 0

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period, min_periods=1).mean()
    return adx / 100.0


def compute_obv(prices: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(prices.diff())
    direction.iloc[0] = 0
    obv = (volume * direction).cumsum()
    obv_ma = obv.rolling(20, min_periods=1).mean()
    obv_std = obv.rolling(20, min_periods=1).std()
    obv_norm = (obv - obv_ma) / (obv_std + 1e-10)
    return obv_norm


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    close = df["EUR_USD"]
    high = df["EUR_USD_High"]
    low = df["EUR_USD_Low"]
    volume = df["EUR_USD_Volume"].replace(0, np.nan).ffill()

    # Returns
    feats["eur_log_return"] = np.log(close / close.shift(1))
    feats["eur_ret_3"] = close.pct_change(3)
    feats["eur_ret_5"] = close.pct_change(5)

    # DXY features
    feats["dxy_log_return"] = np.log(df["DXY"] / df["DXY"].shift(1))
    feats["dxy_ret_5"] = df["DXY"].pct_change(5)
    feats["dxy_ret_20"] = df["DXY"].pct_change(20)

    # Macro
    feats["fedfunds_diff"] = df["FEDFUNDS"].diff()

    # Relative strength: EUR vs DXY
    eur_ret = close.pct_change()
    dxy_ret = df["DXY"].pct_change()
    feats["relative_strength"] = eur_ret - dxy_ret

    # ATR (normalized by 60-day mean)
    atr_raw = compute_atr(high, low, close, DATA_CONFIG["atr_period"])
    atr_ma = atr_raw.rolling(60, min_periods=1).mean()
    feats["atr_norm"] = atr_raw / (atr_ma + 1e-10)

    # MACD histogram (normalized)
    _, _, macd_hist = compute_macd(close)
    macd_hist_scale = macd_hist.rolling(60, min_periods=1).std() + 1e-10
    feats["macd_hist"] = macd_hist / macd_hist_scale

    # Bollinger Band %B
    bb_pctb = compute_bb_pctb(close)
    feats["bb_pctb"] = bb_pctb

    # Bollinger Band width (volatility proxy)
    bb_sma = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std()
    bb_width = (bb_std * 4) / (bb_sma + 1e-10)
    bb_width_ma = bb_width.rolling(60, min_periods=1).mean()
    feats["bb_width"] = bb_width / (bb_width_ma + 1e-10)

    # ADX (trend strength)
    feats["adx"] = compute_adx(high, low, close)

    # MA ratios
    feats["ma_50_ratio"] = close / close.rolling(50, min_periods=1).mean() - 1.0
    feats["ma_200_ratio"] = close / close.rolling(200, min_periods=1).mean() - 1.0

    # DXY momentum (5d vs 20d)
    dxy_ret_5 = df["DXY"].pct_change(5)
    dxy_ret_20 = df["DXY"].pct_change(20)
    feats["dxy_momentum"] = dxy_ret_5 - dxy_ret_20

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.ffill().bfill()

    return feats


def apply_triple_barrier(prices: pd.Series, tp: float, sl: float, max_days: int) -> pd.DataFrame:
    results = []
    prices_arr = prices.values
    for i in range(len(prices_arr) - max_days):
        entry_price = prices_arr[i]
        label = 0
        exit_day = max_days
        exit_price = prices_arr[i + max_days]

        for j in range(1, max_days + 1):
            if i + j >= len(prices_arr):
                break
            current_price = prices_arr[i + j]
            ret = (current_price - entry_price) / entry_price

            if ret >= tp:
                label = 1
                exit_day = j
                exit_price = current_price
                break
            elif ret <= -sl:
                label = -1
                exit_day = j
                exit_price = current_price
                break
        else:
            ret_final = (prices_arr[i + max_days] - entry_price) / entry_price
            label = 1 if ret_final > 0 else -1 if ret_final < 0 else 0
            exit_day = max_days
            exit_price = prices_arr[i + max_days]

        results.append({
            "date": prices.index[i],
            "label": label,
            "exit_day": exit_day,
            "exit_return": (exit_price - entry_price) / entry_price,
        })

    return pd.DataFrame(results).set_index("date")


def verify_no_leakage(feats: pd.DataFrame, prices: pd.DataFrame):
    issues = []
    next_return = np.log(prices["EUR_USD"] / prices["EUR_USD"].shift(-1))
    for col in feats.columns:
        corr = feats[col].corr(next_return)
        if abs(corr) > 0.5:
            msg = f"LEAKAGE WARNING: {col} corr with next-day return = {corr:.3f}"
            issues.append(msg)
            print(msg)

    if not issues:
        print("Leakage check PASSED.")
    return len(issues) == 0


def build_dataset(force_download: bool = False) -> pd.DataFrame:
    raw_path = DATA_CONFIG["parquet_path"].replace(".parquet", "_raw.parquet")

    if not force_download and os.path.exists(raw_path):
        print("Loading cached raw data...")
        df = pd.read_parquet(raw_path)
        if set(RAW_FEATURES).issubset(set(df.columns)):
            df = df[RAW_FEATURES]
            df = df.dropna()
            print(f"Loaded {len(df)} rows from cache")
        else:
            print("Cached raw data missing required columns, re-downloading...")
            force_download = True

    if force_download or not os.path.exists(raw_path):
        print("Fetching yfinance data...")
        df_yf = fetch_yfinance_data()
        print("Fetching FRED data...")
        df_fred = fetch_fred_data()

        df = df_yf.join(df_fred, how="outer")
        df = df.sort_index()
        df = df.ffill().bfill()
        df = df[RAW_FEATURES]
        df = df.dropna()

        os.makedirs(os.path.dirname(DATA_CONFIG["parquet_path"]), exist_ok=True)
        df.to_parquet(raw_path)

    print("Engineering features...")
    feats = engineer_features(df)

    print("Applying Triple Barrier labeling...")
    barriers = apply_triple_barrier(
        df["EUR_USD"],
        tp=DATA_CONFIG["barrier_tp"],
        sl=DATA_CONFIG["barrier_sl"],
        max_days=DATA_CONFIG["barrier_max_days"],
    )

    feats = feats.loc[feats.index.isin(barriers.index)]
    barriers = barriers.loc[barriers.index.isin(feats.index)]
    feats = feats.loc[barriers.index]

    print("Checking for look-ahead bias...")
    verify_no_leakage(feats, df.loc[df.index.isin(feats.index)])

    feats.to_parquet(DATA_CONFIG["parquet_path"])
    barriers.to_parquet(DATA_CONFIG["parquet_path"].replace(".parquet", "_labels.parquet"))
    print(f"Features shape={feats.shape}, Labels shape={barriers.shape}")
    return feats


def load_dataset() -> pd.DataFrame:
    path = DATA_CONFIG["parquet_path"]
    if not os.path.exists(path):
        return build_dataset()
    df = pd.read_parquet(path)
    actual_columns = list(df.columns)
    missing = [c for c in FEATURE_COLUMNS if c not in actual_columns]
    if missing:
        print(f"Missing columns {missing}, rebuilding...")
        return build_dataset()
    return df[FEATURE_COLUMNS]


def load_labels() -> pd.DataFrame:
    path = DATA_CONFIG["parquet_path"].replace(".parquet", "_labels.parquet")
    if not os.path.exists(path):
        build_dataset()
    return pd.read_parquet(path)


def load_raw_prices() -> pd.DataFrame:
    path = DATA_CONFIG["parquet_path"].replace(".parquet", "_raw.parquet")
    if not os.path.exists(path):
        build_dataset()
    return pd.read_parquet(path)[RAW_FEATURES]