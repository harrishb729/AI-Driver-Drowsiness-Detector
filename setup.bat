@echo off
setlocal
pushd "%~dp0"

echo AI Driver Drowsiness Detector - Setup
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found. Install Python 3.14 or newer and enable Add Python to PATH.
    goto failed
)

python -c "import sys; raise SystemExit(0 if (3, 14) <= sys.version_info[:2] < (3, 15) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.14.x is required for the pinned runtime wheels.
    goto failed
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 goto failed
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 goto failed

echo Installing pinned runtime requirements...
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

if not exist "face_detection_yunet_2023mar.onnx" (
    echo ERROR: Missing face_detection_yunet_2023mar.onnx in the project folder.
    goto failed
)
if not exist "shape_predictor_68_face_landmarks.dat" (
    echo ERROR: Missing shape_predictor_68_face_landmarks.dat in the project folder.
    goto failed
)

python -c "import cv2, dlib, numpy; assert hasattr(cv2, 'FaceDetectorYN_create')"
if errorlevel 1 (
    echo ERROR: Runtime dependency verification failed.
    goto failed
)

if not exist "logs" mkdir "logs"
if errorlevel 1 goto failed
if not exist "session_logs" mkdir "session_logs"
if errorlevel 1 goto failed

echo.
echo Setup completed successfully.
echo Launch the application with run.bat.
popd
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above and correct the reported issue.
popd
pause
exit /b 1
