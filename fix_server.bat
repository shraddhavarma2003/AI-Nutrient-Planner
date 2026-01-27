@echo off
echo ===================================================
echo AI Nutrition Server Repair Tool
echo ===================================================
echo.
echo [1/3] Stopping any running server instances...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM uvicorn.exe >nul 2>&1
echo Done.
echo.

echo [2/3] Setting up environment...
set PYTHONPATH=%~dp0
echo PYTHONPATH set to: %PYTHONPATH%
echo.

echo [3/3] Starting the Server...
echo.
echo Please keep this window OPEN. 
echo You can access the app at: http://localhost:8081/static/app.html
echo.
python src/main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Server failed to start.
    echo Please ensure Python is installed and added to PATH.
    pause
)
