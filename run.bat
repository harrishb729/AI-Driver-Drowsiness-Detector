@echo off
setlocal
pushd "%~dp0"

if exist "AI_Driver_Drowsiness_Detector.exe" (
    "AI_Driver_Drowsiness_Detector.exe"
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u main.py
) else (
    python -u main.py
)
set "APP_EXIT_CODE=%ERRORLEVEL%"
if not "%APP_EXIT_CODE%"=="0" (
    echo.
    echo Application stopped with exit code %APP_EXIT_CODE%.
    echo Review logs\application.log if it was created.
    popd
    pause
    exit /b %APP_EXIT_CODE%
)

popd
exit /b 0
