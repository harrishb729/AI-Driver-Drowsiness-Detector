from pathlib import Path

project_root = Path(SPECPATH)
model_files = (
    project_root / "face_detection_yunet_2023mar.onnx",
    project_root / "shape_predictor_68_face_landmarks.dat",
)
missing_models = [str(path) for path in model_files if not path.is_file()]
if missing_models:
    raise SystemExit(
        "Cannot build: required model file(s) missing:\n  "
        + "\n  ".join(missing_models)
    )

datas = [(str(path), "models") for path in model_files]

a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI_Driver_Drowsiness_Detector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="AI_Driver_Drowsiness_Detector",
)
