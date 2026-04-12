from param_sweep import run_backtest_with_params

configs = [
    (0.44, 0.56, 2.0, True, 0.15, 0.01),
    (0.45, 0.55, 2.0, True, 0.15, 0.01),
    (0.46, 0.54, 2.0, True, 0.15, 0.01),
    (0.44, 0.56, 2.5, True, 0.15, 0.01),
    (0.44, 0.56, 3.0, True, 0.15, 0.01),
    (0.44, 0.56, 1.5, True, 0.15, 0.01),
    (0.44, 0.56, 2.0, True, 0.15, 0.005),
    (0.44, 0.56, 2.0, True, 0.15, 0.02),
    (0.44, 0.56, 2.0, False, 0.15, 0.01),
    (0.45, 0.55, 2.5, False, 0.15, 0.01),
]

for i, c in enumerate(configs):
    ntl, nth, tp_sl, use_ma, atr_q, risk = c
    r = run_backtest_with_params(ntl, nth, tp_sl, use_ma, atr_q, risk)
    if r:
        print(f"{i+1:2d}. NT=[{ntl},{nth}] TP/SL={tp_sl} MA={'Y' if use_ma else 'N'} R={risk:.3f} | Net=${r['net_profit']:,.0f} PF={r['pf']:.2f} DA={r['da_traded']:.1f}% UP={r['prec_up']:.1f}% DN={r['prec_dn']:.1f}% T={r['n_trades']} MD={r['md']*100:.1f}% WR={r['win_rate']:.1f}%")