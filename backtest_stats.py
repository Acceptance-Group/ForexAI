import pandas as pd

df = pd.read_csv(r'C:\Users\moonway\Desktop\fin_model\backtest\backtest_results_vol.csv')
trades = df[df['signal'] != 0]
wins = trades[trades['pips'] > 0]
losses = trades[trades['pips'] <= 0]

aw = wins['pips'].mean() if len(wins) > 0 else 0
al = abs(losses['pips'].mean()) if len(losses) > 0 else 0
rr = aw / al if al > 0 else 0

print(f'Trades: {len(trades)}')
print(f'Win rate: {len(wins)/len(trades)*100:.1f}%')
print(f'Avg win: {aw:.1f} pips')
print(f'Avg loss: {al:.1f} pips')
print(f'RR (R:R): {rr:.2f}:1')
print(f'Net P&L: ${trades["pnl"].sum():.0f}')

longs = trades[trades['signal'] > 0]
shorts = trades[trades['signal'] < 0]
lw = longs[longs['pips'] > 0]
sw = shorts[shorts['pips'] > 0]
print(f'Long: {len(longs)} trades, {len(lw)} wins ({len(lw)/len(longs)*100:.1f}%)')
print(f'Short: {len(shorts)} trades, {len(sw)} wins ({len(sw)/len(shorts)*100:.1f}%)')
print(f'Best: {trades["pips"].max():.1f} pips')
print(f'Worst: {trades["pips"].min():.1f} pips')


lw_pips = lw['pips'].mean() if len(lw) > 0 else 0
ll_pips = abs(longs[longs['pips']<=0]['pips'].mean()) if len(longs[longs['pips']<=0]) > 0 else 0
sw_pips = sw['pips'].mean() if len(sw) > 0 else 0
sl_pips = abs(shorts[shorts['pips']<=0]['pips'].mean()) if len(shorts[shorts['pips']<=0]) > 0 else 0
print(f'Long RR: {lw_pips/al if al>0 else 0:.2f}:1 (avg win {lw_pips:.1f}p / avg loss {ll_pips:.1f}p)')
print(f'Short RR: {sw_pips/al if al>0 else 0:.2f}:1 (avg win {sw_pips:.1f}p / avg loss {sl_pips:.1f}p)')