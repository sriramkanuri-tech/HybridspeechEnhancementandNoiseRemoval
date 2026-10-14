"""Resource path helpers for source and frozen PyInstaller builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def bundle_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[2]


def resource_path(*parts) -> Path:
    return bundle_dir().joinpath(*parts)


def models_dir() -> Path:
    # Prefer alongside executable for user-supplied models.
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent / "models"
        if exe_dir.exists():
            return exe_dir
    bundled = bundle_dir() / "models"
    if bundled.exists():
        return bundled
    return Path("models")
