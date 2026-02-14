@echo off
echo Stopping all running Python processes...
taskkill /F /IM python.exe
echo.
echo All Python processes have been stopped. You can now run run.bat again.
pause
