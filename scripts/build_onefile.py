import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist_onefile"
BUILD = ROOT / "build_onefile"
ENTRY = ROOT / "src" / "app.py"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm", "--clean", "--windowed", "--onefile",
    "--name", "SpeechEnhancementNoiseRemoval",
    "--distpath", str(DIST),
    "--workpath", str(BUILD),
    "--paths", str(ROOT),
    "--add-data", f"{ROOT / 'models'};models",
    "--collect-binaries", "onnxruntime",
    "--collect-data", "onnxruntime",
    "--hidden-import", "onnxruntime.capi",
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
subprocess.run(cmd, check=False)