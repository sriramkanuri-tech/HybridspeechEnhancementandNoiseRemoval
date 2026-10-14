"""Enhancement workflow for offline 30-second recordings."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.recording import save_wav
from src.config.settings import AppConfig
from src.dsp.pipeline import EnhancementPipeline
from src.evaluation.metrics import rms
from src.ml.enhancer import MLEnhancer


def _mono(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32).ravel()


def _metrics(audio, sr, elapsed):
    a = _mono(audio)
    return {
        "duration": float(len(a) / sr),
        "rms": float(rms(a)),
        "peak": float(np.max(np.abs(a))) if a.size else 0.0,
        "process_time": float(elapsed),
        "rtf": float(elapsed / max(len(a) / sr, 1e-9)),
    }


def run_methods(audio, sr, session_dir, nr, vp, methods):
    audio = _mono(audio)
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    save_wav(session_dir / "original.wav", audio, sr, 1)
    ml = MLEnhancer()
    results = {}
    dsp_pipeline = EnhancementPipeline(noise_reduction=nr, voice_preservation=vp, sample_rate=sr, auto_gain=False)
    for method in methods:
        m = method.lower()
        t0 = time.perf_counter()
        try:
            if m == "dsp":
                out = dsp_pipeline.process_audio(audio.copy())
                elapsed = time.perf_counter() - t0
                save_wav(session_dir / "enhanced_dsp.wav", out, sr, 1)
                results["DSP"] = {"path": str(session_dir / "enhanced_dsp.wav"), "method": "dsp",
                                  **_metrics(out, sr, elapsed)}
            elif m == "ml":
                if not ml.available:
                    raise RuntimeError(ml.detail)
                out = ml.enhance_audio(audio.copy(), sr)
                elapsed = time.perf_counter() - t0
                save_wav(session_dir / "enhanced_ml.wav", out, sr, 1)
                results["ML"] = {"path": str(session_dir / "enhanced_ml.wav"), "method": "ml",
                                 **_metrics(out, sr, elapsed)}
            elif m == "hybrid":
                if not ml.available:
                    raise RuntimeError("ML unavailable: " + ml.detail)
                ml_out = ml.enhance_audio(audio.copy(), sr)
                hybrid_pipe = EnhancementPipeline(noise_reduction=nr * 0.5, voice_preservation=vp, sample_rate=sr, auto_gain=False)
                out = hybrid_pipe.process_audio(ml_out.copy())
                elapsed = time.perf_counter() - t0
                save_wav(session_dir / "enhanced_hybrid.wav", out, sr, 1)
                save_wav(session_dir / "ml_intermediate.wav", ml_out, sr, 1)
                results["Hybrid"] = {"path": str(session_dir / "enhanced_hybrid.wav"), "method": "hybrid",
                                     **_metrics(out, sr, elapsed)}
            else:
                raise ValueError("Unknown method: " + method)
        except Exception as e:
            results[m.upper()] = {"error": str(e)}
    # write metrics
    try:
        (session_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass
    try:
        from src.evaluation.plots import generate_all_plots
        generate_all_plots(session_dir, audio, results, sr)
    except Exception:
        pass
    return results


def run_all_methods(audio, sr, session_dir, nr, vp):
    return run_methods(audio, sr, session_dir, nr, vp, ["dsp", "ml", "hybrid"])


def compute_metrics(clean, enhanced, sr):
    # helper for external tables
    c = _mono(clean)
    e = _mono(enhanced)
    from src.evaluation.metrics import snr_db
    try:
        snr_in = snr_db(c, c)  # not meaningful for real recordings
    except Exception:
        snr_in = None
    return {"rms": float(rms(e)), "peak": float(np.max(np.abs(e))) if e.size else 0.0}
