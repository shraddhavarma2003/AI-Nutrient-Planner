@echo off
echo ===================================================
echo AI Nutrition Safe Launcher (Port 8081)
echo ===================================================
echo.
echo Attempting to start server...
echo.

set PYTHONPATH=%~dp0
echo PYTHONPATH set to: %PYTHONPATH%

echo Trying 'venv python'...
venv\Scripts\python src/main.py
if %ERRORLEVEL% EQU 0 goto success

echo.
echo 'python' command failed. Trying 'py'...
py src/main.py
if %ERRORLEVEL% EQU 0 goto success

echo.
echo 'py' command failed. Trying 'python3'...
python3 src/main.py
if %ERRORLEVEL% EQU 0 goto success

echo.
echo CRITICAL ERROR: Could not start Python. 
echo Please ensure Python is installed and added to PATH.
pause
exit /b 1

:success
echo.
echo Server stopped.
pause
