# AI Driver Drowsiness Detector

**Version:** 1.0.0  
**Platform:** Windows desktop  
**Processing:** Local webcam, OpenCV, YuNet, and dlib 68-point facial landmarks

## 1. Product overview

AI Driver Drowsiness Detector is a local, real-time driver-monitoring prototype. It analyzes webcam frames for facial landmarks, eye closure and blinks, sustained mouth opening/yawns, and head pose. Its temporal engine combines valid signals into a bounded drowsiness score and drives the existing warning/alarm behavior.

This release packages the already-approved detector. It does not add or retune detection features.

## 2. Features

- OpenCV webcam capture and live camera panel
- YuNet face detection with probable-driver selection
- dlib 68-point facial landmarks
- Per-driver eye and neutral-mouth calibration
- Bilateral blink event counting and sustained-yawn counting
- Head-pose monitoring
- Valid-frame PERCLOS-style eye-closure history
- Bounded drowsiness score with recovery and status bands
- Existing audible alarm with hysteresis
- Timestamped CSV session event logs
- Optional development debug view

## 3. System requirements

- 64-bit Windows 10 or Windows 11
- For source installation: 64-bit Python 3.14.x (the pinned dlib wheel is for the tested Python 3.14 runtime)
- A working desktop session capable of displaying an OpenCV window
- Internet access only during dependency installation or build setup; runtime inference is local

## 4. Hardware requirements

- Webcam supported by OpenCV
- CPU capable of real-time image processing
- Enough free storage for the Python environment, build tools, and the dlib landmark model (approximately 100 MB)

No cloud service or dedicated GPU is required.

## 5. Software requirements

Runtime packages are pinned in [`requirements.txt`](./requirements.txt):

- OpenCV (`opencv-python`)
- NumPy
- `dlib-bin`

The Windows release is built with PyInstaller, listed separately in [`requirements-build.txt`](./requirements-build.txt). Windows `winsound` is provided by the Python standard library.

## 6. Installation

Obtain the complete project, including both required model files. Do not substitute models from unverified sources.

Required model files:

- `face_detection_yunet_2023mar.onnx`
- `shape_predictor_68_face_landmarks.dat`

The model files are not downloaded by the setup scripts. If you already have a configured environment, install the pinned runtime dependencies with:

```text
python -m pip install -r requirements.txt
```

## 7. Setup

Double-click `setup.bat`, or run it from Command Prompt. It checks Python, creates `.venv` if needed, installs the pinned requirements, verifies the runtime imports and model files, and creates `logs` and `session_logs`.

Setup requires a writable project directory. The script pauses at the end so errors remain visible.

## 8. Running the application

Double-click `run.bat`, or launch from the project folder:

```text
python -u main.py
```

`run.bat` locates the project relative to its own location and uses `.venv` when present. Press **Q** in the application window to quit. Press **R** to recalibrate and **D** to toggle the development debug view. Closing the dashboard window or pressing Ctrl+C also requests shutdown.

## 9. Dashboard explanation

The approved layout keeps the live camera on the left and the Driver Monitoring Dashboard on the right. It shows the driver status, drowsiness score, eye state, blink and yawn counts, head state, alarm state, session duration, and logged event count. The dashboard is not redesigned for this release.

## 10. Calibration

At startup the application loads both models, initializes the camera, and calibrates when valid facial landmarks are available. Keep the driver’s face visible, eyes open, and mouth relaxed during calibration. The application displays `CALIBRATION COMPLETE` when its existing calibration procedure finishes. Press **R** to request recalibration.

If a face is unavailable or calibration cannot complete, the application remains in its existing waiting/calibration behavior; missing measurements are not treated as drowsiness.

## 11. Blink detection

Blink events use the existing eye analyzer and its bilateral temporal event logic, with calibrated eye baselines. The event counter is separate from sustained eye-closure scoring. This documentation does not imply that every camera/driver combination has been validated.

## 12. Yawn detection

Yawn events use the existing personalized mouth calibration, opening/release hysteresis, sustained timing, recovery, and re-arm behavior. Brief mouth motion is not equivalent to a confirmed yawn. Smiling or talking may still affect image measurements; validate behavior on the intended camera and driver.

## 13. Drowsiness score

The existing temporal engine combines its valid eye-closure/PERCLOS, yawn, head-pose, and blink-pattern inputs into a bounded score and status. It recovers over time when risk signals subside. Invalid face/landmark measurements are not interpreted as eye closure. The score is an informational software output, not a medical assessment.

## 14. Alarm

The existing warning/alarm thresholds and hysteresis are unchanged. The alarm uses the Windows audio alert implementation and is stopped during application cleanup.

## 15. Session logging

Each launch attempts to create a unique timestamped CSV in `session_logs/`. Existing CSV columns and value meanings are retained. Session logging failures are reported in `logs/application.log` where possible and do not by themselves stop monitoring. The application log is written to `logs/application.log` and rotated when it grows.

## 16. Troubleshooting

- **Python not found:** install 64-bit Python 3.14.x and enable the installer’s PATH option, then reopen the terminal.
- **Missing model:** place both exact model filenames in the project root for source runs. Packaged models are included beneath the release’s `_internal/models` directory.
- **Model load error:** verify that each file is complete and corresponds to the expected YuNet ONNX or dlib 68-point predictor format.
- **Camera unavailable/already in use:** close other camera applications and check Windows camera permissions.
- **Camera frame acquisition failure:** check the webcam connection and driver; restart after resolving the issue.
- **Calibration does not complete:** face the camera with eyes open and mouth relaxed, with adequate light and an unobstructed face.
- **No CSV session file:** ensure the application directory is writable. The monitoring application log will record a session-logging failure when possible.
- **Application log unavailable:** ensure `logs` is writable; startup will report the file-logging problem in the console and continue where possible.
- **OpenCV window issues:** use a local Windows desktop session; remote/headless sessions may not provide a display.
- **dlib import or native-library error:** rerun `setup.bat` and verify that Python, package architecture, and environment are 64-bit and consistent.

## 17. Development and testing

From the project folder:

```text
python -m unittest discover -v
python -m compileall -q .
```

The unit tests cover deterministic temporal behavior. They do not prove live webcam accuracy. The optional `manual_behavior_test.py` is a developer diagnostic, not part of normal application startup or the packaged release.

## 18. Windows deployment

Run `build_windows.bat` from the project folder, or install the build dependency and invoke PyInstaller directly:

```text
python -m pip install -r requirements-build.txt
python -m PyInstaller --noconfirm --clean AI_Driver_Drowsiness_Detector.spec
```

The supported output is a one-folder build at:

```text
dist/AI_Driver_Drowsiness_Detector/
```

It contains `AI_Driver_Drowsiness_Detector.exe`, bundled dependencies, and both models under `_internal/models`. The application creates `logs/` and `session_logs/` next to the executable when needed. Keep the whole folder together when copying or distributing it. The console window is intentional so startup errors remain visible.

## 19. Known limitations

- Accuracy depends on camera placement, lighting, face angle, glasses/occlusion, and individual facial geometry.
- Face selection is a webcam-based probable-driver heuristic, not verified driver identity.
- A successful unit test or startup does not establish live blink/yawn accuracy for every user.
- The application depends on a usable local camera and Windows display/audio facilities.
- It is a prototype monitoring aid, not a replacement for driver attention, rest, or professional safety systems.

## 20. Safety notice

This software is not medically certified, automotive certified, or validated as a safety-critical vehicle control. It does not guarantee accident prevention and must not be relied upon as the sole means of preventing drowsy driving. Drivers must stop safely and rest when tired, regardless of the displayed score or alarm.

## Privacy

Webcam frames and inference are processed locally. This application does not upload video, use remote AI APIs, or add telemetry or remote storage. After dependencies and models are installed, normal operation does not require network access.
