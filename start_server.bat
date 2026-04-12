@echo off
echo Starting EUR/USD Prediction Server...
echo.
echo Server will be available at: http://localhost:5000
echo.
echo Endpoints:
echo   - Main page:     http://localhost:5000/
echo   - Forecast chart: http://localhost:5000/forecast
echo   - API status:    http://localhost:5000/api/status
echo   - API predict:   http://localhost:5000/api/predict
echo.
echo Press Ctrl+C to stop the server
echo.
python server.py