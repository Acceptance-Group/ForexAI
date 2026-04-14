import io
import os

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    YFINANCE_TICKERS, FRED_SERIES, DATA_CONFIG,
    FEATURE_COLUMNS, MT5_CONFIG,
)

MT5_BARS_LIMIT = 20000


def fetch_d1_data(symbol=None) -> pd.DataFrame:
    try:
        import MetaTrader5 as mt5
        if symbol is None:
            symbol = MT5_CONFIG["symbol"]
        mt5_path = MT5_CONFIG.get("path", r"C:\Program Files\MetaTrader 5\terminal64.exe")
        if not mt5.initialize(path=mt5_path):
            print(f"MT5 init failed: {mt5.last_error()}")
            return None

        if not mt5.symbol_info(symbol):
            for s in mt5.symbols_get():
                if s.name == symbol:
                    break

        mt5.symbol_select(symbol, True)

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

        while True:
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, offset, MT5_BARS_LIMIT)
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
            print(f"  D1 {symbol} chunk {chunk_count+1}: {chunk_start.strftime('%Y-%m-%d')} ({len(df_chunk)} bars)")

            if chunk_start < start_date:
                df_chunk = df_chunk[df_chunk.index >= start_date]

            all_dfs.append(df_chunk)
            chunk_count += 1

            if chunk_start < start_date:
                break

            offset += MT5_BARS_LIMIT - 500
            if chunk_count >= 20:
                break

        mt5.shutdown()

        if not all_dfs:
            print(f"No D1 data from MT5 for {symbol}")
            return None

        df = pd.concat(all_dfs)
        df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        print(f"Loaded {len(df)} D1 bars for {symbol}: {df.index[0]} to {df.index[-1]}")
        return df
    except ImportError:
        print("MetaTrader5 not available")
        return None
    except Exception as e:
        print(f"MT5 D1 error for {symbol}: {e}")
        try:
            mt5.shutdown()
        except:
            pass
        return None


def fetch_cross_symbols() -> dict:
    symbols = {"GBPUSD": "GBPUSD", "USDJPY": "USDJPY"}
    results = {}
    for name, sym in symbols.items():
        print(f"Fetching {name} D1 data from MT5...")
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


def compute_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0


def compute_candlestick_features(df):
    body = (df["close"] - df["open"]).abs()
    full_range = df["high"] - df["low"]
    full_range_safe = full_range.replace(0, np.nan).fillna(1e-10)
    body_ratio = body / full_range_safe
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow_pct = upper_shadow / full_range_safe
    lower_shadow_pct = lower_shadow / full_range_safe
    is_hammer = ((lower_shadow >= 2.0 * body) & (upper_shadow <= body * 0.3) & (body > 0)).astype(float)
    is_doji = (body / full_range_safe < 0.1).astype(float)
    return body_ratio, upper_shadow_pct, lower_shadow_pct, is_hammer, is_doji


def compute_vwap_ratio(df):
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].astype(float)
    vol = vol.replace(0, vol.median()).ffill().bfill()
    tp_vol = typical_price * vol
    vwap = tp_vol.rolling(20, min_periods=1).sum() / (vol.rolling(20, min_periods=1).sum() + 1e-10)
    return (df["close"] / vwap - 1.0).fillna(0.0)


def fetch_intraday_features(symbol=None) -> dict:
    """Fetch H4 and H1 last-bar features for each D1 date from MT5."""
    results = {}
    for tf_name, tf_const_val in [("H4", 16388), ("H1", 16385)]:
        try:
            import MetaTrader5 as mt5
            mt5_path = MT5_CONFIG.get("path", r"C:\Program Files\MetaTrader 5\terminal64.exe")
            if not mt5.initialize(path=mt5_path):
                continue
            if symbol is None:
                symbol = MT5_CONFIG["symbol"]
            mt5.symbol_select(symbol, True)

            start_date = DATA_CONFIG["start_date"]
            if isinstance(start_date, str):
                start_date = pd.Timestamp(start_date)
            end_date = DATA_CONFIG.get("end_date", "2026-04-15")
            if isinstance(end_date, str):
                end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)

            bars_limit = 50000
            rates = mt5.copy_rates_from_pos(symbol, tf_const_val, 0, bars_limit)
            mt5.shutdown()

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


def build_economic_calendar(start="2010-01-01") -> pd.DataFrame:
    """Build economic event calendar with date flags for major releases."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=PAYEMS"
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            nfp_dates = set(df.index.normalize())
        else:
            nfp_dates = set()
    except:
        nfp_dates = set()

    date_range = pd.date_range(start=start, end="2026-12-31", freq="D")
    cal = pd.DataFrame(index=date_range)
    cal["is_nfp_day"] = 0.0
    cal["is_fomc_day"] = 0.0
    cal["is_cpi_day"] = 0.0
    cal["news_impact"] = 0.0

    for d in nfp_dates:
        if d in cal.index:
            cal.loc[d, "is_nfp_day"] = 1.0
            cal.loc[d, "news_impact"] = 1.0

    fomc_dates_approx = [
        "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10","2010-09-21","2010-11-03","2010-12-14",
        "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09","2011-09-21","2011-11-02","2011-12-13",
        "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-08-01","2012-09-13","2012-10-24","2012-12-12",
        "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31","2013-09-18","2013-10-30","2013-12-18",
        "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30","2014-09-17","2014-10-29","2014-12-17",
        "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29","2015-09-17","2015-10-28","2015-12-16",
        "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27","2016-09-21","2016-11-02","2016-12-14",
        "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26","2017-09-20","2017-11-01","2017-12-13",
        "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01","2018-09-26","2018-11-08","2018-12-19",
        "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31","2019-09-18","2019-10-30","2019-12-11",
        "2020-01-29","2020-03-15","2020-04-29","2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
        "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15",
        "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02","2022-12-14",
        "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
        "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18",
        "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30","2025-09-17","2025-11-05","2025-12-17",
    ]
    for d_str in fomc_dates_approx:
        d = pd.Timestamp(d_str)
        if d in cal.index:
            cal.loc[d, "is_fomc_day"] = 1.0
            cal.loc[d, "news_impact"] = np.maximum(cal.loc[d, "news_impact"], 1.0)

    cpi_dates_approx = [
        "2010-01-15","2010-02-19","2010-03-17","2010-04-15","2010-05-14","2010-06-17","2010-07-16","2010-08-13","2010-09-15","2010-10-15","2010-11-17","2010-12-15",
        "2011-01-14","2011-02-18","2011-03-17","2011-04-15","2011-05-13","2011-06-15","2011-07-15","2011-08-16","2011-09-15","2011-10-14","2011-11-16","2011-12-16",
    ]
    for y in range(2012, 2027):
        for m in range(1, 13):
            cpi_dates_approx.append(f"{y}-{m:02d}-13")
    for d_str in cpi_dates_approx:
        d = pd.Timestamp(d_str)
        if d in cal.index:
            cal.loc[d, "is_cpi_day"] = 1.0

    return cal


def engineer_d1_features(d1: pd.DataFrame, macro: pd.DataFrame,
                         cross_data: dict = None,
                         intraday_data: dict = None,
                         calendar: pd.DataFrame = None) -> pd.DataFrame:
    feats = pd.DataFrame(index=d1.index)
    close = d1["close"]
    high = d1["high"]
    low = d1["low"]

    feats["d1_log_return"] = np.log(close / close.shift(1))
    feats["d1_ret_3"] = close.pct_change(3)
    feats["d1_ret_5"] = close.pct_change(5)
    feats["d1_ret_10"] = close.pct_change(10)
    feats["d1_ret_20"] = close.pct_change(20)

    # Shifted to avoid leakage
    atr_raw = compute_atr(high, low, close, DATA_CONFIG["atr_period"])
    atr_ma = atr_raw.rolling(60, min_periods=1).mean()
    feats["atr_norm"] = (atr_raw / (atr_ma + 1e-10)).shift(1)

    feats["adx"] = compute_adx(high, low, close).shift(1)

    _, _, macd_hist = compute_macd(close)
    macd_hist_scale = macd_hist.rolling(60, min_periods=1).std() + 1e-10
    feats["macd_hist"] = (macd_hist / macd_hist_scale).shift(1)

    feats["rsi"] = compute_rsi(close, 14).shift(1)

    (feats["body_ratio"], _, feats["lower_shadow_pct"],
     feats["hammer"], feats["doji"]) = compute_candlestick_features(d1)

    vol = d1["tick_volume"].astype(float)
    vol = vol.replace(0, vol.median()).ffill().bfill()
    vol_ma = vol.rolling(20, min_periods=1).mean()
    feats["vol_sma_ratio"] = (vol / (vol_ma + 1e-10)).fillna(1.0)

    sma20 = close.rolling(20, min_periods=1).mean()
    sma50 = close.rolling(50, min_periods=1).mean()
    feats["trend_ma20"] = (close / sma20 - 1.0).shift(1)
    feats["trend_ma50"] = (close / sma50 - 1.0).shift(1)

    feats["vwap_ratio"] = compute_vwap_ratio(d1).shift(1)

    if cross_data is not None:
        for sym_name, sym_df in cross_data.items():
            if sym_df is not None and len(sym_df) > 100:
                sym_close = sym_df["close"]
                sym_reindexed = sym_close.reindex(d1.index).ffill().bfill()
                feats[f"{sym_name.lower()}_ret_1"] = sym_reindexed.pct_change(1).shift(1).values[:len(feats)]
                feats[f"{sym_name.lower()}_ret_5"] = sym_reindexed.pct_change(5).shift(1).values[:len(feats)]
                eur_ret = close.pct_change(1).values[:len(feats)]
                sym_ret = sym_reindexed.pct_change(1).values[:len(feats)]
                rolling_corr = pd.Series(eur_ret).rolling(20, min_periods=5).corr(pd.Series(sym_ret))
                feats[f"{sym_name.lower()}_corr_20"] = rolling_corr.shift(1).values[:len(feats)]
            else:
                feats[f"{sym_name.lower()}_ret_1"] = 0.0
                feats[f"{sym_name.lower()}_ret_5"] = 0.0
                feats[f"{sym_name.lower()}_corr_20"] = 0.0
    else:
        feats["gbpusd_ret_1"] = 0.0
        feats["gbpusd_ret_5"] = 0.0
        feats["gbpusd_corr_20"] = 0.0
        feats["usdjpy_ret_1"] = 0.0
        feats["usdjpy_ret_5"] = 0.0
        feats["usdjpy_corr_20"] = 0.0

    if macro is not None and "DXY" in macro.columns:
        macro_daily = macro.resample("D").last().ffill()
        macro_daily = macro_daily.shift(1)
        dates = d1.index.normalize()
        macro_aligned = macro_daily.reindex(dates).ffill()

        dxy_close = macro_aligned["DXY"]
        feats["dxy_log_return"] = np.log(dxy_close / dxy_close.shift(1)).values[:len(feats)]
        dxy_ret_5 = dxy_close.pct_change(5)
        dxy_ret_20 = dxy_close.pct_change(20)
        feats["dxy_momentum"] = (dxy_ret_5 - dxy_ret_20).values[:len(feats)]
    else:
        feats["dxy_log_return"] = 0.0
        feats["dxy_momentum"] = 0.0

    # Multi-timeframe intraday features (shifted by 1 day to avoid leakage)
    if intraday_data is not None:
        for tf_name, tf_cols in [("H4", ["h4_ret_last", "h4_ret_3d", "h4_vol_ratio", "h4_range_pct", "h4_body_pct"]),
                                   ("H1", ["h1_ret_last", "h1_vol_ratio", "h1_range_pct"])]:
            if tf_name in intraday_data and intraday_data[tf_name] is not None:
                tf_df = intraday_data[tf_name]
                tf_dates = tf_df.index.normalize()
                tf_close = tf_df["close"]
                tf_vol = tf_df["tick_volume"].astype(float).replace(0, tf_df["tick_volume"].median())
                tf_high = tf_df["high"]
                tf_low = tf_df["low"]
                tf_open = tf_df["open"]

                daily_groups = tf_df.groupby(tf_dates)
                daily_last_close = daily_groups["close"].last()
                daily_first_close = daily_groups["close"].first()
                daily_open_close_3d = daily_groups["close"].last().pct_change(3)

                last_close_reindexed = daily_last_close.shift(1).reindex(d1.index).ffill()
                first_close_reindexed = daily_first_close.shift(1).reindex(d1.index).ffill()
                ret_3d_reindexed = daily_open_close_3d.shift(1).reindex(d1.index).ffill()

                if tf_name == "H4":
                    daily_vol_mean = daily_groups["tick_volume"].mean()
                    vol_mean_reindexed = daily_vol_mean.shift(1).reindex(d1.index).ffill()
                    daily_vol_last = daily_groups["tick_volume"].last()
                    vol_last_reindexed = daily_vol_last.shift(1).reindex(d1.index).ffill()
                    feats["h4_ret_last"] = ((last_close_reindexed / first_close_reindexed) - 1).values[:len(feats)]
                    feats["h4_ret_3d"] = ret_3d_reindexed.values[:len(feats)]
                    vol_ratio = (vol_last_reindexed / (vol_mean_reindexed + 1e-10)).values[:len(feats)]
                    feats["h4_vol_ratio"] = np.nan_to_num(vol_ratio, nan=1.0)

                    daily_range_pct = daily_groups.apply(lambda x: (x["high"].max() - x["low"].min()) / (x["close"].iloc[0] + 1e-10) if len(x) > 0 else 0)
                    feats["h4_range_pct"] = daily_range_pct.shift(1).reindex(d1.index).ffill().values[:len(feats)]

                    daily_body_pct = daily_groups.apply(lambda x: abs(x["close"].iloc[-1] - x["open"].iloc[0]) / (x["high"].max() - x["low"].min() + 1e-10) if len(x) > 0 else 0)
                    feats["h4_body_pct"] = daily_body_pct.shift(1).reindex(d1.index).ffill().values[:len(feats)]

                elif tf_name == "H1":
                    daily_vol_mean = daily_groups["tick_volume"].mean()
                    vol_mean_reindexed = daily_vol_mean.shift(1).reindex(d1.index).ffill()
                    daily_vol_last = daily_groups["tick_volume"].last()
                    vol_last_reindexed = daily_vol_last.shift(1).reindex(d1.index).ffill()
                    feats["h1_ret_last"] = ((last_close_reindexed / first_close_reindexed) - 1).values[:len(feats)]
                    feats["h1_vol_ratio"] = (vol_last_reindexed / (vol_mean_reindexed + 1e-10)).values[:len(feats)]

                    daily_range_pct = daily_groups.apply(lambda x: (x["high"].max() - x["low"].min()) / (x["close"].iloc[0] + 1e-10) if len(x) > 0 else 0)
                    feats["h1_range_pct"] = daily_range_pct.shift(1).reindex(d1.index).ffill().values[:len(feats)]
            else:
                if tf_name == "H4":
                    for col in ["h4_ret_last", "h4_ret_3d", "h4_vol_ratio", "h4_range_pct", "h4_body_pct"]:
                        feats[col] = 0.0
                else:
                    for col in ["h1_ret_last", "h1_vol_ratio", "h1_range_pct"]:
                        feats[col] = 0.0
    else:
        for col in ["h4_ret_last", "h4_ret_3d", "h4_vol_ratio", "h4_range_pct", "h4_body_pct",
                     "h1_ret_last", "h1_vol_ratio", "h1_range_pct"]:
            feats[col] = 0.0

    # Economic calendar features (shifted by 1 day)
    if calendar is not None:
        cal_shifted = calendar.shift(1)
        cal_reindexed = cal_shifted.reindex(d1.index).ffill().fillna(0)
        for col in ["news_impact", "is_nfp_day", "is_fomc_day", "is_cpi_day"]:
            if col in cal_reindexed.columns:
                feats[col] = cal_reindexed[col].values[:len(feats)]
            else:
                feats[col] = 0.0
    else:
        feats["news_impact"] = 0.0
        feats["is_nfp_day"] = 0.0
        feats["is_fomc_day"] = 0.0
        feats["is_cpi_day"] = 0.0

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.ffill().bfill()

    for col in FEATURE_COLUMNS:
        if col not in feats.columns:
            feats[col] = 0.0

    return feats[FEATURE_COLUMNS]


def apply_triple_barrier_d1(prices: pd.Series, tp: float, sl: float, max_bars: int) -> pd.DataFrame:
    results = []
    prices_arr = prices.values

    for i in range(len(prices_arr) - max_bars):
        entry_price = prices_arr[i]
        label = 0
        exit_bar = max_bars

        for j in range(1, max_bars + 1):
            if i + j >= len(prices_arr):
                break
            ret = (prices_arr[i + j] - entry_price) / entry_price

            if ret >= tp:
                label = 1
                exit_bar = j
                break
            elif ret <= -sl:
                label = -1
                exit_bar = j
                break
        else:
            ret_final = (prices_arr[i + max_bars] - entry_price) / entry_price
            label = 1 if ret_final > 0 else -1 if ret_final < 0 else 0
            exit_bar = max_bars

        results.append({
            "date": prices.index[i],
            "label": label,
            "exit_bar": exit_bar,
        })

    return pd.DataFrame(results).set_index("date")


def verify_no_leakage(feats: pd.DataFrame, prices: pd.DataFrame):
    issues = []
    h4_close = prices["close"] if isinstance(prices, pd.DataFrame) else prices
    next_return = np.log(h4_close / h4_close.shift(-1))
    for col in feats.columns:
        corr = feats[col].corr(next_return.reindex(feats.index))
        if abs(corr) > 0.5:
            msg = f"LEAKAGE WARNING: {col} corr with next-bar return = {corr:.3f}"
            issues.append(msg)
            print(msg)
    if not issues:
        print("Leakage check PASSED.")
    return len(issues) == 0


def build_regime_labels(prices: pd.Series, atr: pd.Series, lookback: int = 20) -> pd.DataFrame:
    next_ret = prices.pct_change().shift(-1)
    next_abs_ret = next_ret.abs()
    atr_val = atr
    rolling_atr_mean = atr_val.rolling(lookback, min_periods=1).mean()
    threshold = 0.5 * atr_val / prices
    trending = (next_abs_ret > threshold).astype(int)
    results = pd.DataFrame({
        "regime_label": trending,
        "next_abs_ret": next_abs_ret,
        "threshold": threshold,
    }, index=prices.index)
    results = results.replace([np.inf, -np.inf], np.nan).dropna()
    return results


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
        raise RuntimeError("Cannot get D1 data. Please start MetaTrader 5 and try again.")

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