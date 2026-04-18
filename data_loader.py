import io
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    YFINANCE_TICKERS, FRED_SERIES, DATA_CONFIG,
    FEATURE_COLUMNS, CTRADER_CONFIG,
)

BARS_LIMIT = 20000

_broker_instance = None


def _get_broker():
    global _broker_instance
    if _broker_instance is not None:
        from broker import get_broker
        b = get_broker()
        if b and b.is_connected:
            return b
    from broker import init_broker, get_broker
    if not init_broker():
        raise RuntimeError("Cannot connect to broker")
    _broker_instance = True
    return get_broker()


def _get_broker_symbol(symbol=None):
    if symbol is None:
        symbol = CTRADER_CONFIG.get("symbol", "EURUSD")
    clean = symbol.replace("/", "").replace(" ", "").upper()
    symbol_map = {
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD",
        "USDJPY": "USDJPY",
    }
    return symbol_map.get(clean, clean)


def fetch_d1_data(symbol=None) -> pd.DataFrame:
    try:
        broker = _get_broker()
        sym = _get_broker_symbol(symbol)

        start_date = DATA_CONFIG["start_date"]
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        end_date = DATA_CONFIG.get("end_date")
        if end_date is None:
            end_date = pd.Timestamp.now()
        elif isinstance(end_date, str):
            end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)

        all_dfs = []
        offset = 0
        chunk_count = 0
        tf_d1 = broker.TIMEFRAME_D1

        while True:
            rates = broker.copy_rates_from_pos(sym, tf_d1, offset, BARS_LIMIT)
            if rates is None or len(rates) == 0:
                break

            df_chunk = pd.DataFrame(rates)
            df_chunk["time"] = pd.to_datetime(df_chunk["time"], unit="s")
            df_chunk = df_chunk.set_index("time")
            df_chunk = df_chunk[["open", "high", "low", "close", "tick_volume"]]
            df_chunk = df_chunk.dropna()
            df_chunk = df_chunk[df_chunk["close"] > 0]
            if df_chunk.index.tz:
                df_chunk.index = df_chunk.index.tz_localize(None)

            chunk_start = df_chunk.index[0]
            print(f"  D1 {sym} chunk {chunk_count+1}: {chunk_start.strftime('%Y-%m-%d')} ({len(df_chunk)} bars)")

            if chunk_start < start_date:
                df_chunk = df_chunk[df_chunk.index >= start_date]

            all_dfs.append(df_chunk)
            chunk_count += 1

            if chunk_start < start_date:
                break

            offset += BARS_LIMIT - 500
            if chunk_count >= 20:
                break

        if not all_dfs:
            print(f"No D1 data from broker for {sym}")
            return None

        df = pd.concat(all_dfs)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        print(f"Loaded {len(df)} D1 bars for {sym}: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"Broker D1 error for {symbol}: {e}")
        return None


def fetch_cross_symbols() -> dict:
    symbols = {"GBPUSD": "GBPUSD", "USDJPY": "USDJPY"}
    results = {}
    for name, sym in symbols.items():
        print(f"Fetching {name} D1 data from broker...")
        for alias in [sym, sym.replace("USD", "USD.")]:
            df = fetch_d1_data(alias)
            if df is not None and len(df) > 100:
                results[name] = df
                break
        if name not in results:
            print(f"  Could not fetch {name}")
    return results


def fetch_daily_macro() -> pd.DataFrame:
    print("Fetching daily macro data (DXY, VIX, FRED)...")
    frames = {}
    end_date = DATA_CONFIG.get("end_date")

    for name, ticker in YFINANCE_TICKERS.items():
        if name == "EUR_USD":
            continue
        try:
            df = yf.download(ticker, start=DATA_CONFIG["start_date"],
                             end=end_date if end_date else None, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                close_col = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
            else:
                close_col = df["Close"]
            frames[name] = close_col
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    for name, series_id in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
            df.rename(columns={df.columns[0]: "date", df.columns[1]: name}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df[name] = pd.to_numeric(df[name], errors="coerce")
            df = df.set_index("date")
            frames[name] = df[name]
        except Exception as e:
            print(f"  Warning: Could not fetch {name}: {e}")

    macro = pd.DataFrame(frames)
    macro = macro.sort_index()
    macro = macro.ffill().bfill()
    if macro.index.tz:
        macro.index = macro.index.tz_localize(None)
    print(f"  Macro data: {len(macro)} rows")
    return macro


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_val = compute_atr(high, low, close, period)
    plus_di = 100 * (plus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period, min_periods=1).mean() / (atr_val + 1e-10))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period, min_periods=1).mean()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    for i in range(period, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd - macd_signal


def build_economic_calendar(start_date, end_date=None):
    calendars = {
        "nfp": ["2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03",
                "2024-06-07", "2024-07-05", "2024-08-02", "2024-09-06", "2024-10-04",
                "2024-11-01", "2024-12-06", "2025-01-10", "2025-02-07", "2025-03-07",
                "2025-04-04", "2025-05-02", "2025-06-06", "2025-07-04", "2025-08-01",
                "2025-09-05", "2025-10-03", "2025-11-07", "2025-12-05",
                "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03"],
        "fomc": ["2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
                 "2024-09-18", "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19",
                 "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-11-05",
                 "2025-12-17", "2026-01-28", "2026-03-18"],
    }
    start = pd.Timestamp(start_date)
    end_d = pd.Timestamp(end_date) if end_date else pd.Timestamp.now()
    all_dates = pd.DatetimeIndex([])
    for name, dates in calendars.items():
        for d in dates:
            dt = pd.Timestamp(d)
            for offset_days in [-1, 0, 1]:
                event_day = dt + pd.Timedelta(days=offset_days)
                if start <= event_day <= end_d:
                    all_dates = all_dates.append(pd.DatetimeIndex([event_day]))
    return all_dates.unique().sort_values() if len(all_dates) > 0 else pd.DatetimeIndex([])


def fetch_intraday_features(symbol=None) -> dict:
    results = {}

    broker = _get_broker()
    tf_map = {"H4": broker.TIMEFRAME_H4, "H1": broker.TIMEFRAME_H1}

    for tf_name, tf_const in tf_map.items():
        try:
            if symbol is None:
                sym = CTRADER_CONFIG.get("symbol", "EURUSD")
            else:
                sym = _get_broker_symbol(symbol)

            start_date = DATA_CONFIG["start_date"]
            if isinstance(start_date, str):
                start_date = pd.Timestamp(start_date)
            end_date = DATA_CONFIG.get("end_date", "2026-04-15")
            if isinstance(end_date, str):
                end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            bars_limit = 50000
            rates = broker.copy_rates_from_pos(sym, tf_const, 0, bars_limit)

            if rates is None or len(rates) == 0:
                print(f"  No {tf_name} intraday data")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            df = df[["open", "high", "low", "close", "tick_volume"]]
            df = df.dropna()
            df = df[df["close"] > 0]
            if df.index.tz:
                df.index = df.index.tz_localize(None)
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            print(f"  {tf_name} intraday: {len(df)} bars")
            results[tf_name] = df
        except Exception as e:
            print(f"  {tf_name} intraday error: {e}")

    return results


def engineer_d1_features(d1, macro, cross_data, intraday_data, calendar):
    df = d1.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["tick_volume"].astype(float)

    df["d1_log_return"] = np.log(close / close.shift(1))
    df["d1_ret_3"] = df["d1_log_return"].rolling(3).sum()
    df["d1_ret_5"] = df["d1_log_return"].rolling(5).sum()
    df["d1_ret_10"] = df["d1_log_return"].rolling(10).sum()
    df["d1_ret_20"] = df["d1_log_return"].rolling(20).sum()

    if "DXY" in macro.columns:
        dxy = macro["DXY"].reindex(df.index, method="ffill")
        df["dxy_momentum"] = dxy.pct_change(5).reindex(df.index)
    else:
        df["dxy_momentum"] = 0.0

    atr = compute_atr(high, low, close, DATA_CONFIG["atr_period"])
    adx = compute_adx(high, low, close, DATA_CONFIG["atr_period"])
    rsi = compute_rsi(close, 14)
    df["atr_norm"] = atr / close
    df["adx"] = adx / 100.0
    df["rsi"] = rsi / 100.0
    df["macd_hist"] = compute_macd_hist(close)

    body = close - df["open"] if "open" in df.columns else close - close.shift(1)
    rng = high - low
    df["body_ratio"] = (body.abs() / (rng + 1e-10)).fillna(0)
    lower_shadow = df["open"].clip(upper=close) - low if "open" in df.columns else low * 0
    df["lower_shadow_pct"] = (lower_shadow / (rng + 1e-10)).fillna(0)

    vol_sma = vol.rolling(20, min_periods=1).mean()
    df["vol_sma_ratio"] = (vol / (vol_sma + 1e-10)).fillna(1.0)

    ma20 = close.rolling(20, min_periods=1).mean()
    ma50 = close.rolling(50, min_periods=1).mean()
    df["trend_ma20"] = (close / ma20 - 1.0).fillna(0)
    df["trend_ma50"] = (close / ma50 - 1.0).fillna(0)

    typical_price = (high + low + close) / 3
    vol_recursive = vol.replace(0, vol.median()).ffill().bfill()
    tp_vol = typical_price * vol_recursive
    vwap = tp_vol.rolling(20, min_periods=1).sum() / (vol_recursive.rolling(20, min_periods=1).sum() + 1e-10)
    df["vwap_ratio"] = (close / vwap - 1.0).fillna(0.0)

    for cross_name, cross_df in cross_data.items():
        prefix = cross_name.lower()
        cross_close = cross_df["close"].reindex(df.index, method="ffill")
        df[f"{prefix}_ret_1"] = cross_close.pct_change(1)
        df[f"{prefix}_ret_5"] = cross_close.pct_change(5)
        rolling_corr = close.rolling(20, min_periods=10).corr(cross_close)
        df[f"{prefix}_corr_20"] = rolling_corr.fillna(0)

    if "H4" in intraday_data:
        h4 = intraday_data["H4"]
        h4_aligned = h4.reindex(df.index, method="ffill")
        df["h4_ret_last"] = h4_aligned["close"].pct_change(1).fillna(0)
        h4_ret_3d = h4_aligned["close"].pct_change(3).fillna(0)
        df["h4_ret_3d"] = h4_ret_3d
        h4_range = (h4_aligned["high"] - h4_aligned["low"]) / (h4_aligned["close"] + 1e-10)
        df["h4_range_pct"] = h4_range.fillna(0)
        h4_body = (h4_aligned["close"] - h4_aligned["open"]).abs() / (h4_aligned["high"] - h4_aligned["low"] + 1e-10)
        df["h4_body_pct"] = h4_body.fillna(0)

    df = df.shift(1)

    if len(calendar) > 0:
        calendar_set = set(calendar.normalize())
        df["is_news_day"] = df.index.normalize().isin(calendar_set).astype(int)
    else:
        df["is_news_day"] = 0

    return df


def verify_no_leakage(features, raw_prices):
    shifted_cols = [c for c in features.columns if c in [
        "d1_log_return", "d1_ret_3", "d1_ret_5", "d1_ret_10", "d1_ret_20",
        "atr_norm", "adx", "rsi", "macd_hist", "vwap_ratio",
    ]]
    for col in shifted_cols:
        if col in features.columns and col in raw_prices.columns:
            future_vals = raw_prices[col].shift(-1)
            corr = features[col].corr(future_vals)
            if abs(corr) > 0.1:
                print(f"  WARNING: {col} may leak (shifted corr={corr:.4f})")


def apply_triple_barrier_d1(close, tp=0.010, sl=0.005, max_bars=20):
    n = len(close)
    labels = pd.DataFrame(index=close.index, columns=["label", "pips", "pnl"])
    for i in range(n - 1):
        entry_price = close.iloc[i + 1]
        if np.isnan(entry_price) or entry_price <= 0:
            continue
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        label = 0
        exit_price = entry_price
        for j in range(1, min(max_bars + 1, n - i - 1)):
            high_j = close.iloc[i + 1 + j - 1] if (i + 1 + j - 1) < n else close.iloc[i + 1]
            low_j = close.iloc[i + 1 + j - 1] if (i + 1 + j - 1) < n else close.iloc[i + 1]
            if j < n - i - 2:
                h = close.iloc[i + 1:i + 1 + j + 2].max()
                l = close.iloc[i + 1:i + 1 + j + 2].min()
            else:
                h = close.iloc[i + 1 + j] if i + 1 + j < n else close.iloc[i + 1]
                l = close.iloc[i + 1 + j] if i + 1 + j < n else close.iloc[i + 1]
            if h >= tp_price:
                label = 1
                exit_price = tp_price
                break
            if l <= sl_price:
                label = -1
                exit_price = sl_price
                break
        else:
            label = 0
            exit_price = close.iloc[min(i + 1 + max_bars, n - 1)]
        pips = (exit_price - entry_price) * 10000
        labels.iloc[i] = [label, pips, pips]
    labels = labels.shift(1)
    labels.columns = ["label", "pips", "pnl"]
    return labels.dropna()


def build_dataset(force_download: bool = False) -> pd.DataFrame:
    parquet_path = DATA_CONFIG["parquet_path"]
    raw_path = DATA_CONFIG["raw_parquet_path"]

    if not force_download and os.path.exists(parquet_path):
        feats = pd.read_parquet(parquet_path)
        if set(FEATURE_COLUMNS).issubset(set(feats.columns)):
            print(f"Loaded {len(feats)} D1 features from cache")
            return feats[FEATURE_COLUMNS]

    d1 = fetch_d1_data()
    if d1 is None or len(d1) < 500:
        raise RuntimeError("Cannot get D1 data. Check broker connection and try again.")

    cross_data = fetch_cross_symbols()
    macro = fetch_daily_macro()

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    d1.to_parquet(raw_path)
    print(f"Saved D1 raw data: {len(d1)} bars")

    print("Engineering D1 features...")
    intraday_data = fetch_intraday_features()
    calendar = build_economic_calendar(DATA_CONFIG["start_date"])
    feats = engineer_d1_features(d1, macro, cross_data, intraday_data, calendar)

    print("Applying Triple Barrier labeling...")
    barriers = apply_triple_barrier_d1(
        d1["close"],
        tp=DATA_CONFIG["barrier_tp"],
        sl=DATA_CONFIG["barrier_sl"],
        max_bars=DATA_CONFIG["barrier_max_bars"],
    )

    common_idx = feats.index.intersection(barriers.index)
    feats = feats.loc[common_idx]
    barriers = barriers.loc[common_idx]

    print("Checking for look-ahead bias...")
    d1_aligned = d1.loc[d1.index.isin(feats.index)]
    verify_no_leakage(feats, d1_aligned)

    feats = feats[FEATURE_COLUMNS]

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    feats.to_parquet(parquet_path)
    barriers.to_parquet(parquet_path.replace(".parquet", "_labels.parquet"))
    print(f"Features shape={feats.shape}, Labels shape={barriers.shape}")
    return feats


def load_dataset() -> pd.DataFrame:
    parquet_path = DATA_CONFIG["parquet_path"]
    if not os.path.exists(parquet_path):
        return build_dataset()
    df = pd.read_parquet(parquet_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Missing columns {missing}, rebuilding...")
        return build_dataset(force_download=True)
    return df[FEATURE_COLUMNS]


def load_labels() -> pd.DataFrame:
    path = DATA_CONFIG["parquet_path"].replace(".parquet", "_labels.parquet")
    if not os.path.exists(path):
        build_dataset()
    return pd.read_parquet(path)


def load_raw_prices() -> pd.DataFrame:
    raw_path = DATA_CONFIG["raw_parquet_path"]
    if not os.path.exists(raw_path):
        build_dataset()
    return pd.read_parquet(raw_path)