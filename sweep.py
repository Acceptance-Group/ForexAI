import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
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


def run_sweep():
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

    mask = (feats_df.index >= BACKTEST_CONFIG["start_date"]) & \
           (feats_df.index < BACKTEST_CONFIG["end_date"])
    bt_indices = np.where(mask)[0]
    bt_indices = bt_indices[bt_indices >= lookback]
    bt_indices = bt_indices[bt_indices < len(eurusd) - 1]

    atr_raw = pd.Series(eurusd).diff().abs().rolling(DATA_CONFIG["atr_period"]).mean()

    all_probs = []
    with torch.no_grad():
        for idx in bt_indices:
            seq = scaled[idx - lookback:idx]
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prob = model.predict_proba(X).detach().cpu().item()
            all_probs.append(prob)
    all_probs = np.array(all_probs)

    results = []

    for ntl, nth in [(0.45, 0.55), (0.46, 0.54), (0.47, 0.53), (0.44, 0.56), (0.42, 0.58)]:
        for tp_sl in [2.0, 2.5, 3.0, 3.5]:
            for risk in [0.01, 0.015, 0.02, 0.03]:
                for max_pos in [1.0, 1.5, 2.0]:
                    strat_returns = []
                    traded_count = 0
                    up_count = 0
                    dn_count = 0
                    wins = 0
                    losses = 0

                    for i, idx in enumerate(bt_indices):
                        prob = all_probs[i]
                        current_price = eurusd[idx - 1]
                        next_price = eurusd[idx]
                        actual_ret = np.log(next_price / current_price) if current_price > 0 else 0.0

                        atr_val = atr_raw.iloc[idx] if idx < len(atr_raw) else np.nan
                        if np.isnan(atr_val):
                            strat_returns.append(0.0)
                            continue

                        signal = 0.0
                        if prob > nth:
                            signal = 1.0
                        elif prob < ntl:
                            signal = -1.0

                        if signal == 0.0:
                            strat_returns.append(0.0)
                            continue

                        traded_count += 1
                        if signal > 0:
                            up_count += 1
                        else:
                            dn_count += 1

                        atr_ma = atr_raw.rolling(60, min_periods=1).mean().iloc[idx]
                        vol_ratio = atr_val / (atr_ma + 1e-10) if not np.isnan(atr_ma) else 1.0
                        dyn_tp_sl = tp_sl / max(vol_ratio, 0.5)
                        dyn_tp_sl = min(dyn_tp_sl, 4.0)

                        sl_pct = atr_val / current_price if atr_val > 0 and current_price > 0 else 0.01
                        pos_frac = min(risk / sl_pct, max_pos) if sl_pct > 0 else 0.0

                        if signal > 0:
                            tp_pct = sl_pct * dyn_tp_sl
                            if actual_ret >= tp_pct:
                                capped = tp_pct
                            elif actual_ret <= -sl_pct:
                                capped = -sl_pct
                            else:
                                capped = actual_ret
                        else:
                            tp_pct = sl_pct * dyn_tp_sl
                            if actual_ret <= -tp_pct:
                                capped = tp_pct
                            elif actual_ret >= sl_pct:
                                capped = -sl_pct
                            else:
                                capped = -actual_ret

                        ret = signal * pos_frac * capped
                        if ret > 0:
                            wins += 1
                        else:
                            losses += 1
                        strat_returns.append(ret)

                    strat_returns = np.array(strat_returns)
                    if len(strat_returns) == 0 or traded_count == 0:
                        continue

                    cum_ret = np.cumprod(1 + strat_returns) - 1
                    pf = profit_factor(strat_returns[strat_returns != 0]) if np.any(strat_returns != 0) else 0
                    md = max_drawdown(strat_returns)
                    net_profit = 10000 * cum_ret[-1]
                    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

                    results.append({
                        "ntl": ntl, "nth": nth, "tp_sl": tp_sl, "risk": risk,
                        "max_pos": max_pos, "net": net_profit, "pf": pf,
                        "md": abs(md) * 100, "trades": traded_count,
                        "wr": wr, "up": up_count, "dn": dn_count,
                        "cum_ret": cum_ret[-1] * 100,
                    })

    df = pd.DataFrame(results)
    df = df.sort_values("net", ascending=False)

    print(f"\n{'='*100}")
    print("PARAMETER SWEEP RESULTS (Top 30 by Net Profit)")
    print(f"{'='*100}")
    print(f"{'NT_L':>5} {'NT_H':>5} {'TP/SL':>5} {'Risk':>5} {'MaxP':>5} | {'Net$':>8} {'PF':>5} {'MD%':>5} {'Tr':>5} {'WR%':>5} {'UP':>4} {'DN':>4} {'Ret%':>7}")
    print("-" * 100)
    for _, r in df.head(30).iterrows():
        print(f"{r['ntl']:.2f}  {r['nth']:.2f}  {r['tp_sl']:.1f}   {r['risk']:.3f}  {r['max_pos']:.1f}  | "
              f"${r['net']:>7,.0f} {r['pf']:>5.2f} {r['md']:>5.1f} {int(r['trades']):>4d} {r['wr']:>5.1f} "
              f"{int(r['up']):>4d} {int(r['dn']):>4d} {r['cum_ret']:>6.1f}%")

    df.to_csv("backtest/sweep_results.csv", index=False)
    print(f"\nFull results saved to backtest/sweep_results.csv")


if __name__ == "__main__":
    run_sweep()