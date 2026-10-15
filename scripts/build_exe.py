"""Reproducible PyInstaller build script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
NAME = "SpeechEnhancementNoiseRemoval"
ENTRY = ROOT / "src" / "app.py"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", NAME,
    "--distpath", str(DIST),
    "--workpath", str(BUILD),
    "--paths", str(ROOT),
    "--add-data", f"{ROOT / 'models'};models",
    "--collect-all", "onnxruntime",
    "--collect-all", "sounddevice",
    "--collect-all", "scipy",
    "--hidden-import", "scipy.signal",
    "--hidden-import", "scipy.signal.windows",
    "--hidden-import", "PySide6.QtCore",
    "--hidden-import", "PySide6.QtGui",
    "--hidden-import", "PySide6.QtWidgets",
    "--exclude-module", "torch",
    "--exclude-module", "pandas",
    "--exclude-module", "pytest",
    str(ENTRY),
]

print(" ".join(cmd))
subprocess.run(cmd, check=True)
print("Build complete:", DIST / NAME / (NAME + ".exe"))
