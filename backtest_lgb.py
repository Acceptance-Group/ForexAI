import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, RISK_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS,
)
from data_loader import load_dataset, load_raw_prices, load_labels


def build_tabular_features(features_np, features_df, labels_df, lookback):
    rows = []
    valid_indices = []
    label_indices = labels_df.index.intersection(features_df.index)

    for idx_str in label_indices:
        idx = features_df.index.get_loc(idx_str)
        if idx < lookback:
            continue
        seq = features_np[idx - lookback:idx]

        row = {}
        for feat_idx, feat_name in enumerate(FEATURE_COLUMNS):
            vals = seq[:, feat_idx]
            row[f"{feat_name}_mean"] = np.mean(vals)
            row[f"{feat_name}_std"] = np.std(vals)
            row[f"{feat_name}_last"] = vals[-1]
            row[f"{feat_name}_trend"] = (vals[-1] - vals[0]) / (np.std(vals) + 1e-10)
            row[f"{feat_name}_min"] = np.min(vals)
            row[f"{feat_name}_max"] = np.max(vals)

        rows.append(row)
        valid_indices.append(idx_str)

    df_tabular = pd.DataFrame(rows, index=valid_indices)
    return df_tabular, valid_indices


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_drawdown(returns):
    curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max
    return dd.min()


def run_backtest_lgb():
    feats_df = load_dataset()
    prices_df = load_raw_prices()
    labels_df = load_labels()

    common_idx = feats_df.index.intersection(labels_df.index)
    feats_df = feats_df.loc[common_idx]
    labels_df = labels_df.loc[common_idx]

    price_common = feats_df.index.intersection(prices_df.index)
    feats_df = feats_df.loc[price_common]
    labels_df = labels_df.loc[price_common]
    prices_df = prices_df.loc[price_common]

    with open(SCALER_SAVE_PATH.replace(".pkl", "_lgb.pkl"), "rb") as f:
        saved = pickle.load(f)
    scaler = saved["scaler"]
    tabular_info = saved["tabular_info"]

    model_path = MODEL_SAVE_PATH.replace(".pth", "_lgb.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    lookback = DATA_CONFIG["lookback"]
    scaled = scaler.transform(feats_df.values)

    X_tabular, valid_indices = build_tabular_features(
        scaled, feats_df, labels_df, lookback
    )

    X = X_tabular.values.astype(np.float32)
    eurusd = prices_df["EUR_USD"].values

    no_trade_low = DATA_CONFIG["no_trade_low"]
    no_trade_high = DATA_CONFIG["no_trade_high"]
    tp_sl_ratio = DATA_CONFIG["tp_sl_ratio"]
    risk_pct = RISK_CONFIG["risk_per_trade"]
    max_pos_frac = RISK_CONFIG["max_position_fraction"]
    initial_equity = RISK_CONFIG["initial_equity"]

    probs = model.predict(X)

    valid_idx_ints = []
    for vi in valid_indices:
        valid_idx_ints.append(feats_df.index.get_loc(vi))
    valid_idx_ints = np.array(valid_idx_ints)

    mask = (feats_df.index[valid_idx_ints] >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index[valid_idx_ints] < BACKTEST_CONFIG["end_date"])
    bt_mask = np.where(mask)[0]
    bt_mask = bt_mask[bt_mask < len(eurusd) - 1]

    if len(bt_mask) == 0:
        print("No backtest data available.")
        return

    atr_series = pd.Series(eurusd).diff().abs().rolling(DATA_CONFIG["atr_period"]).mean()
    atr_mean = atr_series.rolling(60).mean()

    ma_period = DATA_CONFIG.get("trend_filter_ma", 200)
    if ma_period <= 0:
        ma_period = 200
    ma_200 = pd.Series(eurusd).rolling(ma_period).mean().values

    dates_list = []
    actual_returns = []
    dir_probs = []
    signals_basic = []
    signals_risk = []
    pos_fractions = []
    returns_capped = []

    for bi in bt_mask:
        idx = valid_idx_ints[bi]
        prob = probs[bi]
        current_price = eurusd[idx - 1]
        next_price = eurusd[idx]
        actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0

        idx_date = feats_df.index[idx]
        atr_val = atr_series.iloc[idx] if idx < len(atr_series) else np.nan
        atr_mean_val = atr_mean.iloc[idx] if idx < len(atr_mean) else np.nan
        ma_val = ma_200[idx] if idx < len(ma_200) else np.nan

        atr_ok = not np.isnan(atr_val) and atr_val > 0
        above_ma200 = not np.isnan(ma_val) and current_price > ma_val
        below_ma200 = not np.isnan(ma_val) and current_price < ma_val

        if not np.isnan(atr_val) and not np.isnan(atr_mean_val) and atr_mean_val > 0:
            vol_ratio = atr_val / atr_mean_val
            dynamic_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
            dynamic_tp_sl = min(dynamic_tp_sl, 3.0)
        else:
            dynamic_tp_sl = tp_sl_ratio

        signal_b = 1.0 if prob > 0.5 else -1.0
        signal_r = 0.0
        pos_frac = 0.0
        capped_ret = 0.0

        if atr_ok:
            if prob > no_trade_high:
                if above_ma200:
                    signal_r = 1.0
                elif below_ma200:
                    signal_r = 0.5
                else:
                    signal_r = 0.75
            elif prob < no_trade_low:
                if below_ma200:
                    signal_r = -1.0
                elif above_ma200:
                    signal_r = -0.5
                else:
                    signal_r = -0.75

            if signal_r != 0.0:
                sl_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.01
                tp_pct = sl_pct * dynamic_tp_sl
                pos_frac = min(risk_pct / sl_pct, max_pos_frac) if sl_pct > 0 else 0.0

                if signal_r > 0:
                    if actual_ret >= tp_pct:
                        capped_ret = tp_pct
                    elif actual_ret <= -sl_pct:
                        capped_ret = -sl_pct
                    else:
                        capped_ret = actual_ret
                elif signal_r < 0:
                    if actual_ret <= -tp_pct:
                        capped_ret = tp_pct
                    elif actual_ret >= sl_pct:
                        capped_ret = -sl_pct
                    else:
                        capped_ret = -actual_ret

        dates_list.append(idx_date)
        actual_returns.append(actual_ret)
        dir_probs.append(prob)
        signals_basic.append(signal_b)
        signals_risk.append(signal_r)
        pos_fractions.append(pos_frac)
        returns_capped.append(capped_ret)

    actual_returns = np.array(actual_returns)
    dir_probs = np.array(dir_probs)
    signals_basic = np.array(signals_basic)
    signals_risk = np.array(signals_risk)
    pos_fractions = np.array(pos_fractions)
    returns_capped = np.array(returns_capped)
    dates = pd.to_datetime(dates_list)

    actual_direction = (actual_returns > 0).astype(float)
    pred_direction = (dir_probs > 0.5).astype(float)

    da_basic = np.mean(pred_direction == actual_direction) * 100
    traded_mask = signals_risk != 0
    da_risk = np.mean(pred_direction[traded_mask] == actual_direction[traded_mask]) * 100 if traded_mask.sum() > 0 else 0.0

    up_trades = signals_risk > 0
    down_trades = signals_risk < 0
    n_up = up_trades.sum()
    n_down = down_trades.sum()
    precision_up = np.mean(actual_direction[up_trades] == 1) * 100 if n_up > 0 else 0.0
    precision_down = np.mean(actual_direction[down_trades] == 0) * 100 if n_down > 0 else 0.0

    trade_frequency = traded_mask.sum() / len(traded_mask) * 100

    strategy_basic = signals_basic * actual_returns
    strategy_risk_fixed = signals_risk * actual_returns
    strategy_risk_sized = signals_risk * pos_fractions * actual_returns
    strategy_tpsl = np.where(traded_mask, signals_risk * pos_fractions * returns_capped, 0.0)

    cum_basic = np.cumprod(1 + strategy_basic) - 1
    cum_risk_fixed = np.cumprod(1 + strategy_risk_fixed) - 1
    cum_risk_sized = np.cumprod(1 + strategy_risk_sized) - 1
    cum_tpsl = np.cumprod(1 + strategy_tpsl) - 1
    cum_hold = np.cumprod(1 + actual_returns) - 1

    md_basic = max_drawdown(strategy_basic)
    md_risk_fixed = max_drawdown(strategy_risk_fixed) if traded_mask.any() else 0.0
    md_risk_sized = max_drawdown(strategy_risk_sized) if traded_mask.any() else 0.0
    md_tpsl = max_drawdown(strategy_tpsl) if traded_mask.any() else 0.0

    pf_basic = profit_factor(strategy_basic)
    pf_risk_fixed = profit_factor(strategy_risk_fixed) if traded_mask.any() else 0.0
    pf_risk_sized = profit_factor(strategy_risk_sized) if traded_mask.any() else 0.0
    pf_tpsl = profit_factor(strategy_tpsl) if traded_mask.any() else 0.0

    def avg_wl(returns, mask=None):
        r = returns[mask > 0] if mask is not None else returns
        w = r[r > 0]
        l = r[r < 0]
        aw = np.mean(w) * 100 if len(w) > 0 else 0.0
        al = abs(np.mean(l)) * 100 if len(l) > 0 else 0.0
        ratio = aw / al if al > 0 else float("inf")
        win_rate = len(w) / (len(w) + len(l)) * 100 if (len(w) + len(l)) > 0 else 0.0
        return aw, al, ratio, len(w) + len(l), win_rate

    aw_b, al_b, wl_b, nt_b, wr_b = avg_wl(strategy_basic)
    aw_f, al_f, wl_f, nt_f, wr_f = avg_wl(strategy_risk_fixed, traded_mask)
    aw_s, al_s, wl_s, nt_s, wr_s = avg_wl(strategy_risk_sized, traded_mask)
    aw_t, al_t, wl_t, nt_t, wr_t = avg_wl(strategy_tpsl, traded_mask)

    sharpe_basic = np.mean(strategy_basic) / (np.std(strategy_basic) + 1e-8) * np.sqrt(252)
    sharpe_risk_fixed = np.mean(strategy_risk_fixed) / (np.std(strategy_risk_fixed) + 1e-8) * np.sqrt(252) if traded_mask.any() else 0.0
    sharpe_risk_sized = np.mean(strategy_risk_sized) / (np.std(strategy_risk_sized) + 1e-8) * np.sqrt(252) if traded_mask.any() else 0.0
    sharpe_tpsl = np.mean(strategy_tpsl) / (np.std(strategy_tpsl) + 1e-8) * np.sqrt(252) if traded_mask.any() else 0.0

    final_risk_fixed = initial_equity * (1 + cum_risk_fixed[-1])
    final_risk_sized = initial_equity * (1 + cum_risk_sized[-1])
    final_tpsl = initial_equity * (1 + cum_tpsl[-1])
    final_hold = initial_equity * (1 + cum_hold[-1])

    print(f"\n{'='*70}")
    print(f"LIGHTGBM BACKTEST ({BACKTEST_CONFIG['start_date']} - {BACKTEST_CONFIG['end_date']})")
    print(f"{'='*70}")
    print(f"Config: No-Trade Zone [{no_trade_low}, {no_trade_high}], TP/SL={tp_sl_ratio}")
    print(f"        Risk/trade={risk_pct*100:.1f}%, Max Pos={max_pos_frac*100:.0f}%")
    print(f"Samples                    : {len(actual_returns)}")
    print(f"")
    print(f"--- Directional Accuracy ---")
    print(f"DA Raw (>0.5)             : {da_basic:.2f}%")
    print(f"DA Risk-Managed            : {da_risk:.2f}%")
    print(f"")
    print(f"--- Per-Direction ---")
    print(f"UP Precision               : {precision_up:.2f}%  ({n_up} trades)")
    print(f"DOWN Precision             : {precision_down:.2f}%  ({n_down} trades)")
    print(f"Trade Frequency            : {trade_frequency:.1f}% ({traded_mask.sum()} trades)")
    print(f"")
    print(f"--- Avg Win vs Avg Loss ---")
    print(f"                  AvgWin%  AvgLoss%  W/L   WR%    N")
    print(f"Basic             : {aw_b:.3f}   {al_b:.3f}   {wl_b:.2f}x  {wr_b:.1f}%  {nt_b}")
    print(f"Risk Fixed        : {aw_f:.3f}   {al_f:.3f}   {wl_f:.2f}x  {wr_f:.1f}%  {nt_f}")
    print(f"Risk Sized        : {aw_s:.3f}   {al_s:.3f}   {wl_s:.2f}x  {wr_s:.1f}%  {nt_s}")
    print(f"TP/SL             : {aw_t:.3f}   {al_t:.3f}   {wl_t:.2f}x  {wr_t:.1f}%  {nt_t}")
    print(f"")
    print(f"--- Dollar Results (${initial_equity:,.0f}) ---")
    print(f"Risk Fixed      : ${final_risk_fixed:,.0f}  (Net: ${final_risk_fixed - initial_equity:,.0f})")
    print(f"Risk Sized      : ${final_risk_sized:,.0f}  (Net: ${final_risk_sized - initial_equity:,.0f})")
    print(f"TP/SL           : ${final_tpsl:,.0f}  (Net: ${final_tpsl - initial_equity:,.0f})")
    print(f"Buy & Hold      : ${final_hold:,.0f}  (Net: ${final_hold - initial_equity:,.0f})")
    print(f"")
    print(f"--- Returns ---")
    print(f"Basic            : {cum_basic[-1]*100:.2f}%")
    print(f"Risk Fixed       : {cum_risk_fixed[-1]*100:.2f}%")
    print(f"Risk Sized       : {cum_risk_sized[-1]*100:.2f}%")
    print(f"TP/SL            : {cum_tpsl[-1]*100:.2f}%")
    print(f"Buy & Hold       : {cum_hold[-1]*100:.2f}%")
    print(f"")
    print(f"--- Profit Factor ---")
    print(f"PF Basic         : {pf_basic:.2f}")
    print(f"PF Risk Fixed    : {pf_risk_fixed:.2f}")
    print(f"PF Risk Sized    : {pf_risk_sized:.2f}")
    print(f"PF TP/SL         : {pf_tpsl:.2f}")
    print(f"")
    print(f"--- Max Drawdown ---")
    print(f"MD Basic         : {md_basic*100:.2f}%")
    print(f"MD Risk Fixed    : {md_risk_fixed*100:.2f}%")
    print(f"MD Risk Sized    : {md_risk_sized*100:.2f}%")
    print(f"MD TP/SL         : {md_tpsl*100:.2f}%")
    print(f"")
    print(f"--- Sharpe ---")
    print(f"Sharpe Basic     : {sharpe_basic:.2f}")
    print(f"Sharpe Risk Fixed: {sharpe_risk_fixed:.2f}")
    print(f"Sharpe Risk Sized: {sharpe_risk_sized:.2f}")
    print(f"Sharpe TP/SL     : {sharpe_tpsl:.2f}")

    return {
        "da_basic": da_basic,
        "da_risk": da_risk,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "pf_tpsl": pf_tpsl,
        "final_tpsl": final_tpsl,
        "cum_tpsl": cum_tpsl[-1],
        "md_tpsl": md_tpsl,
        "sharpe_tpsl": sharpe_tpsl,
    }


if __name__ == "__main__":
    run_backtest_lgb()