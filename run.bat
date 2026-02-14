@echo off
echo ==================================================
echo       STARTING AI FACE SCANNER SYSTEM
echo ==================================================
echo.
echo 1. Killing any old background instances...
taskkill /F /IM python.exe /T >nul 2>&1
echo.

echo 2. Launching Application...
echo    - Open your browser to: http://127.0.0.1:5000
echo    - Press CTRL+C here to stop.
echo.
python app.py
pause
