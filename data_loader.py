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


def compute_atr(prices: pd.Series, period: int = 14) -> pd.Series:
    high_low = prices.rolling(period).max() - prices.rolling(period).min()
    return high_low


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)

    feats["eur_log_return"] = np.log(df["EUR_USD"] / df["EUR_USD"].shift(1))
    feats["eur_ret_3"] = df["EUR_USD"].pct_change(3)
    feats["eur_ret_5"] = df["EUR_USD"].pct_change(5)
    feats["eur_ret_10"] = df["EUR_USD"].pct_change(10)

    feats["dxy_log_return"] = np.log(df["DXY"] / df["DXY"].shift(1))
    feats["dxy_ret_3"] = df["DXY"].pct_change(3)
    feats["dxy_ret_5"] = df["DXY"].pct_change(5)
    feats["dxy_ret_10"] = df["DXY"].pct_change(10)
    feats["dxy_ret_20"] = df["DXY"].pct_change(20)

    feats["vix_log_return"] = np.log(df["VIX"] / df["VIX"].shift(1))
    feats["t10y2y_diff"] = df["T10Y2Y"].diff()
    feats["fedfunds_diff"] = df["FEDFUNDS"].diff()

    eur_ret = df["EUR_USD"].pct_change()
    dxy_ret = df["DXY"].pct_change()
    feats["relative_strength"] = eur_ret - dxy_ret

    dxy_ret_5 = df["DXY"].pct_change(5)
    dxy_ret_20 = df["DXY"].pct_change(20)
    feats["dxy_momentum"] = dxy_ret_5 - dxy_ret_20

    feats["atr_14"] = compute_atr(df["EUR_USD"], DATA_CONFIG["atr_period"])
    atr_mean = feats["atr_14"].rolling(60).mean()
    feats["atr_14"] = feats["atr_14"] / atr_mean

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.ffill().bfill()

    return feats


def apply_triple_barrier(prices: pd.Series, tp: float, sl: float, max_days: int) -> pd.DataFrame:
    results = []
    for i in range(len(prices) - max_days):
        entry_price = prices.iloc[i]
        label = 0
        exit_day = max_days
        exit_price = prices.iloc[i + max_days]

        for j in range(1, max_days + 1):
            if i + j >= len(prices):
                break
            current_price = prices.iloc[i + j]
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
            ret_final = (prices.iloc[i + max_days] - entry_price) / entry_price
            label = 1 if ret_final > 0 else -1 if ret_final < 0 else 0
            exit_day = max_days
            exit_price = prices.iloc[i + max_days]

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


def build_dataset() -> pd.DataFrame:
    print("Fetching yfinance data...")
    df_yf = fetch_yfinance_data()
    print("Fetching FRED data...")
    df_fred = fetch_fred_data()

    df = df_yf.join(df_fred, how="outer")
    df = df.sort_index()
    df = df.ffill().bfill()

    df = df[RAW_FEATURES]
    df = df.dropna()

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

    os.makedirs(os.path.dirname(DATA_CONFIG["parquet_path"]), exist_ok=True)
    df.to_parquet(DATA_CONFIG["parquet_path"].replace(".parquet", "_raw.parquet"))
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
    if actual_columns != FEATURE_COLUMNS:
        print(f"Column mismatch! Rebuilding...")
        print(f"  Expected: {FEATURE_COLUMNS}")
        print(f"  Got:      {actual_columns}")
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