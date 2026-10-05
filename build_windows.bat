@echo off
setlocal
pushd "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python was not found.
        goto failed
    )
    set "PYTHON=python"
)

if not exist "face_detection_yunet_2023mar.onnx" (
    echo ERROR: Missing face_detection_yunet_2023mar.onnx.
    goto failed
)
if not exist "shape_predictor_68_face_landmarks.dat" (
    echo ERROR: Missing shape_predictor_68_face_landmarks.dat.
    goto failed
)

echo Installing pinned build requirements...
%PYTHON% -m pip install -r requirements-build.txt
if errorlevel 1 goto failed

echo Building one-folder Windows release...
%PYTHON% -m PyInstaller --noconfirm --clean AI_Driver_Drowsiness_Detector.spec
if errorlevel 1 goto failed

set "RELEASE=dist\AI_Driver_Drowsiness_Detector"
if not exist "%RELEASE%\logs" mkdir "%RELEASE%\logs"
if errorlevel 1 goto failed
if not exist "%RELEASE%\session_logs" mkdir "%RELEASE%\session_logs"
if errorlevel 1 goto failed
copy /Y "run.bat" "%RELEASE%\run.bat" >nul
if errorlevel 1 goto failed
copy /Y "README.md" "%RELEASE%\README.md" >nul
if errorlevel 1 goto failed
copy /Y "CHANGELOG.md" "%RELEASE%\CHANGELOG.md" >nul
if errorlevel 1 goto failed

if not exist "%RELEASE%\_internal\models\face_detection_yunet_2023mar.onnx" (
    echo ERROR: YuNet model is missing from the packaged output.
    goto failed
)
if not exist "%RELEASE%\_internal\models\shape_predictor_68_face_landmarks.dat" (
    echo ERROR: dlib model is missing from the packaged output.
    goto failed
)

echo.
echo Build completed successfully:
echo %CD%\%RELEASE%
popd
pause
exit /b 0

:failed
echo.
echo Build failed. Review the error above.
popd
pause
exit /b 1
