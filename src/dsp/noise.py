"""Noise estimation and VAD helpers."""

from __future__ import annotations

import numpy as np


def frame_energy_db(frame: np.ndarray, eps: float = 1e-12) -> float:
    rms = float(np.sqrt(np.mean(np.square(frame)) + eps))
    return 20.0 * np.log10(rms + eps)


def estimate_noise_floor_mag(magnitude: np.ndarray, previous: np.ndarray | None = None,
                            alpha: float = 0.9, floor: float = 1e-8) -> np.ndarray:
    """Exponential moving average of magnitude. Skewed toward quieter frames externally."""
    mag = np.asarray(magnitude, dtype=np.float64)
    if previous is None:
        return mag.copy()
    prev = np.asarray(previous, dtype=np.float64)
    return np.maximum(alpha * prev + (1.0 - alpha) * mag, floor)


def voice_activity(magnitude: np.ndarray, noise_floor_mag: np.ndarray,
                   threshold_db: float = 6.0, eps: float = 1e-12) -> bool:
    """Crude VAD using total energy ratio."""
    signal_energy = float(np.mean(magnitude ** 2))
    noise_energy = float(np.mean(noise_floor_mag ** 2))
    if noise_energy < eps:
        return signal_energy > eps
    ratio_db = 10.0 * np.log10((signal_energy + eps) / (noise_energy + eps))
    return ratio_db > threshold_db


def calibrate_noise_profile(magnitudes: np.ndarray, percentile: float = 20.0) -> np.ndarray:
    """Estimate noise profile from a sequence of magnitudes (noise-only or mixed).

    Uses lower percentile to avoid speech dominance.
    """
    m = np.asarray(magnitudes, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    if m.shape[0] == 0:
        return np.ones(m.shape[-1] if m.ndim > 1 else 1, dtype=np.float64) * 1e-6
    # magnitude across time: shape (frames, bins) -> percentile over time
    return np.percentile(m, percentile, axis=0).astype(np.float64)


def update_noise_estimate(current_noise: np.ndarray, frame_mag: np.ndarray,
                         is_speech: bool, alpha_quiet: float = 0.85,
                         alpha_speech: float = 0.995, floor: float = 1e-8) -> np.ndarray:
    """Update noise estimate; slower during speech to avoid absorbing speech."""
    alpha = alpha_speech if is_speech else alpha_quiet
    merged = alpha * current_noise + (1.0 - alpha) * frame_mag
    return np.maximum(merged, floor)
