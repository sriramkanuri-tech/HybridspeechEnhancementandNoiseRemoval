"""Application entry point."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.paths import bundle_dir
from src.ui.main_window import run_app


def main():
    logdir = None
    try:
        from pathlib import Path
        logdir = Path.home() / ".speech-enhancement-noise-removal"
        logdir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(logdir / "app.log"),
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
        )
    except Exception:
        pass
    from src.ml.enhancer import MLEnhancer
    ml = MLEnhancer()
    try:
        logdir = Path.home() / ".speech-enhancement-noise-removal"
        logdir.mkdir(parents=True, exist_ok=True)
        try:
            from src.config.paths import models_dir
            md = models_dir()
        except Exception as e:
            md = str(e)
        (logdir / "startup.log").write_text(
            f"frozen={getattr(sys, 'frozen', False)}\nexec={sys.executable}\ncwd={Path.cwd()}\nbundle={bundle_dir()}\nMEIPASS={getattr(sys, '_MEIPASS', None)}\nmodels_dir={md}\nml={ml.detail}\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    run_app()


if __name__ == "__main__":
    main()
