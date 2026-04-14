@echo off
cd /d C:\Users\moonway\Desktop\fin_model
set USE_ALL_DATA=1
python trainer_xgb.py
python trainer_multi.py
python backtest_chart.py
echo.
echo ========================================
echo   Full retrain complete (USE_ALL_DATA=1)
echo ========================================
pause