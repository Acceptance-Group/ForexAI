import os
import pickle
import itertools

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_CONFIG, BACKTEST_CONFIG, DEVICE, RISK_CONFIG,
    MODEL_SAVE_PATH, SCALER_SAVE_PATH,
    FEATURE_COLUMNS, FEATURE_WEIGHTS,
)
from data_loader import load_dataset, load_raw_prices, load_labels
from model import ForexClassifier


def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf")
    return gross_profit / gross_loss


def max_drawdown(returns):
    if len(returns) == 0:
        return 0.0
    curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(curve)
    dd = (curve - running_max) / running_max
    return dd.min()


def run_backtest_with_params(no_trade_low, no_trade_high, tp_sl_ratio,
                              use_trend_ma, atr_quantile, risk_pct):
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

    eurusd = prices_df["EUR_USD"].values

    with open(SCALER_SAVE_PATH, "rb") as f:
        scaler = pickle.load(f)

    model = ForexClassifier(feature_weights=FEATURE_WEIGHTS).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    scaled = scaler.transform(feats_df.values)
    lookback = DATA_CONFIG["lookback"]
    max_pos_frac = RISK_CONFIG["max_position_fraction"]
    initial_equity = RISK_CONFIG["initial_equity"]

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices >= lookback]
    bt_indices = bt_indices[bt_indices < len(eurusd) - 1]

    if len(bt_indices) == 0:
        return None

    atr_series_raw = pd.Series(eurusd).diff().abs().rolling(DATA_CONFIG["atr_period"]).mean()
    atr_threshold = atr_series_raw.rolling(60).quantile(atr_quantile)

    ma_period = 200 if use_trend_ma else 0
    ma_vals = pd.Series(eurusd).rolling(max(ma_period, 2)).mean().values if ma_period > 0 else np.full(len(eurusd), np.nan)

    strategy_returns = []

    with torch.no_grad():
        for idx in bt_indices:
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prob = model.predict_proba(X).detach().cpu().item()

            current_price = eurusd[idx - 1]
            next_price = eurusd[idx]
            actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0
            idx_date = feats_df.index[idx]

            atr_val = atr_series_raw.iloc[idx] if idx < len(atr_series_raw) else np.nan
            atr_thr = atr_threshold.iloc[idx] if idx < len(atr_threshold) else np.nan

            atr_ok = not np.isnan(atr_val) and not np.isnan(atr_thr) and atr_val > atr_thr
            ma_val = ma_vals[idx] if idx < len(ma_vals) else np.nan

            if use_trend_ma and ma_period > 0:
                above_ma = not np.isnan(ma_val) and current_price > ma_val
                below_ma = not np.isnan(ma_val) and current_price < ma_val
            else:
                above_ma = True
                below_ma = True

            if not np.isnan(atr_val):
                atr_mean = atr_series_raw.rolling(60).mean().iloc[idx] if idx < len(atr_series_raw) else np.nan
                if not np.isnan(atr_mean) and atr_mean > 0:
                    vol_ratio = atr_val / atr_mean
                    dyn_tp_sl = tp_sl_ratio / max(vol_ratio, 0.5)
                    dyn_tp_sl = min(dyn_tp_sl, 3.0)
                else:
                    dyn_tp_sl = tp_sl_ratio
            else:
                dyn_tp_sl = tp_sl_ratio

            signal = 0.0
            pos_frac = 0.0
            capped_ret = 0.0

            if atr_ok:
                if prob > no_trade_high:
                    if above_ma:
                        signal = 1.0
                    elif below_ma:
                        signal = 0.5
                    else:
                        signal = 0.75
                elif prob < no_trade_low:
                    if below_ma:
                        signal = -1.0
                    elif above_ma:
                        signal = -0.5
                    else:
                        signal = -0.75

                if signal != 0.0:
                    sl_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.01
                    tp_pct = sl_pct * dyn_tp_sl
                    pos_frac = min(risk_pct / sl_pct, max_pos_frac) if sl_pct > 0 else 0.0

                    if signal > 0:
                        if actual_ret >= tp_pct:
                            capped_ret = tp_pct
                        elif actual_ret <= -sl_pct:
                            capped_ret = -sl_pct
                        else:
                            capped_ret = actual_ret
                    elif signal < 0:
                        if actual_ret <= -tp_pct:
                            capped_ret = tp_pct
                        elif actual_ret >= sl_pct:
                            capped_ret = -sl_pct
                        else:
                            capped_ret = -actual_ret

            strategy_returns.append({
                "ret": actual_ret,
                "prob": prob,
                "signal": signal,
                "pos_frac": pos_frac,
                "capped_ret": capped_ret,
            })

    df = pd.DataFrame(strategy_returns)
    rets = df["ret"].values
    probs = df["prob"].values
    sigs = df["signal"].values
    pos_fracs = df["pos_frac"].values
    cap_rets = df["capped_ret"].values

    traded = sigs != 0
    if traded.sum() == 0:
        return None

    dir_actual = (rets > 0).astype(float)
    dir_pred = (probs > 0.5).astype(float)
    da = np.mean(dir_pred == dir_actual) * 100
    da_traded = np.mean(dir_pred[traded] == dir_actual[traded]) * 100 if traded.sum() > 0 else 0

    up_mask = sigs > 0
    dn_mask = sigs < 0
    prec_up = np.mean(dir_actual[up_mask] == 1) * 100 if up_mask.sum() > 0 else 0
    prec_dn = np.mean(dir_actual[dn_mask] == 0) * 100 if dn_mask.sum() > 0 else 0

    strat_tpsl = np.where(traded, sigs * pos_fracs * cap_rets, 0.0)
    cum_tpsl = np.cumprod(1 + strat_tpsl) - 1
    pf = profit_factor(strat_tpsl[traded]) if traded.sum() > 0 else 0
    md = max_drawdown(strat_tpsl)
    net_profit = initial_equity * cum_tpsl[-1]
    n_trades = int(traded.sum())
    freq = traded.sum() / len(traded) * 100

    wins = strat_tpsl[traded][strat_tpsl[traded] > 0]
    losses = strat_tpsl[traded][strat_tpsl[traded] < 0]
    win_rate = len(wins) / (len(wins) + len(losses)) * 100 if (len(wins) + len(losses)) > 0 else 0

    return {
        "da_raw": da,
        "da_traded": da_traded,
        "prec_up": prec_up,
        "prec_dn": prec_dn,
        "net_profit": net_profit,
        "pf": pf,
        "md": abs(md),
        "n_trades": n_trades,
        "freq": freq,
        "win_rate": win_rate,
        "cum_ret": cum_tpsl[-1],
    }


def optimize_parameters():
    no_trade_ranges = [
        (0.44, 0.56),
        (0.45, 0.55),
        (0.47, 0.53),
        (0.40, 0.60),
        (0.42, 0.58),
    ]
    tp_sl_ratios = [1.2, 1.5, 2.0, 2.5]
    trend_mas = [0, 200]
    atr_quantiles = [0.15, 0.20, 0.25, 0.30]
    risk_pcts = [0.005, 0.01, 0.02]

    best_result = None
    best_params = None
    results = []

    total = len(no_trade_ranges) * len(tp_sl_ratios) * len(trend_mas) * len(atr_quantiles) * len(risk_pcts)
    count = 0

    for (ntl, nth) in no_trade_ranges:
        for tp_sl in tp_sl_ratios:
            for use_ma in trend_mas:
                for atr_q in atr_quantiles:
                    for risk in risk_pcts:
                        count += 1
                        result = run_backtest_with_params(
                            ntl, nth, tp_sl, use_ma == 200, atr_q, risk
                        )
                        if result is None:
                            continue

                        result["ntl"] = ntl
                        result["nth"] = nth
                        result["tp_sl"] = tp_sl
                        result["use_ma"] = use_ma == 200
                        result["atr_q"] = atr_q
                        result["risk"] = risk
                        results.append(result)

                        score = result["net_profit"]
                        if best_result is None or score > best_result["net_profit"]:
                            best_result = result
                            best_params = {
                                "no_trade": (ntl, nth),
                                "tp_sl": tp_sl,
                                "use_ma": use_ma == 200,
                                "atr_q": atr_q,
                                "risk": risk,
                            }

                        if count % 20 == 0:
                            print(f"  [{count}/{total}] Best so far: ${best_result['net_profit']:,.0f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("net_profit", ascending=False)

    print(f"\n{'='*80}")
    print("PARAMETER OPTIMIZATION RESULTS (Top 20 by Net Profit)")
    print(f"{'='*80}")
    cols = ["ntl", "nth", "tp_sl", "use_ma", "atr_q", "risk",
            "net_profit", "pf", "md", "da_traded", "prec_up", "prec_dn",
            "n_trades", "freq", "win_rate"]
    for i, row in results_df.head(20).iterrows():
        print(f"  NT=[{row['ntl']},{row['nth']}] TP/SL={row['tp_sl']:.1f} "
              f"MA={'Y' if row['use_ma'] else 'N'} ATRq={row['atr_q']:.2f} "
              f"Risk={row['risk']:.3f} | "
              f"Net=${row['net_profit']:,.0f} PF={row['pf']:.2f} "
              f"MD={row['md']*100:.1f}% DA={row['da_traded']:.1f}% "
              f"UP={row['prec_up']:.1f}% DN={row['prec_dn']:.1f}% "
              f"Trades={row['n_trades']} WR={row['win_rate']:.1f}%")

    print(f"\nBEST PARAMETERS:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Net Profit: ${best_result['net_profit']:,.0f}")
    print(f"  Profit Factor: {best_result['pf']:.2f}")
    print(f"  Max Drawdown: {best_result['md']*100:.1f}%")
    print(f"  DA (traded): {best_result['da_traded']:.1f}%")
    print(f"  UP Precision: {best_result['prec_up']:.1f}%")
    print(f"  DOWN Precision: {best_result['prec_dn']:.1f}%")
    print(f"  Trades: {best_result['n_trades']}")

    results_df.to_csv("backtest/param_sweep_results.csv", index=False)
    print(f"\nAll results saved to backtest/param_sweep_results.csv")
    return results_df


if __name__ == "__main__":
    optimize_parameters()