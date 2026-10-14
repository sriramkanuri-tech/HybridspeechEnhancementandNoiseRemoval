"""Spectral gain filters for speech enhancement."""

from __future__ import annotations

import numpy as np


def spectral_subtraction_gain(magnitude: np.ndarray, noise_mag: np.ndarray,
                             strength: float = 1.0, floor_factor: float = 0.02,
                             eps: float = 1e-12) -> np.ndarray:
    """Return gain (0..1-ish) using power spectral subtraction.

    magnitude: (..., bins) or (bins,)
    noise_mag: same shape or broadcastable.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    nse = np.asarray(noise_mag, dtype=np.float64)
    power = mag ** 2
    noise_power = nse ** 2
    subtracted = power - strength * noise_power
    floor_power = (floor_factor * nse) ** 2
    enhanced_power = np.maximum(subtracted, floor_power)
    # gain in magnitude domain
    gain = np.sqrt(enhanced_power / (power + eps))
    gain = np.clip(gain, 0.0, 1.0)
    return gain


def wiener_gain(magnitude: np.ndarray, noise_mag: np.ndarray,
                strength: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    mag = np.asarray(magnitude, dtype=np.float64)
    nse = np.asarray(noise_mag, dtype=np.float64)
    signal_power = mag ** 2
    noise_power = (strength * nse) ** 2
    gain = signal_power / (signal_power + noise_power + eps)
    gain = np.sqrt(np.clip(gain, 0.0, 1.0))
    return gain


def apply_gain(magnitude: np.ndarray, phase: np.ndarray, gain: np.ndarray) -> np.ndarray:
    stft = magnitude * gain * np.exp(1j * phase)
    return stft


def normalize_gain(audio: np.ndarray, target_rms: float = 0.05, max_gain: float = 4.0) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
    if rms < 1e-12:
        return audio.copy()
    g = target_rms / rms
    g = min(g, max_gain)
    return audio * g


def clip_protect(audio: np.ndarray, threshold: float = 0.98) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > threshold and peak > 0:
        audio = audio * (threshold / peak)
    return np.clip(audio, -1.0, 1.0)


def combine_gains(g_sub: np.ndarray, g_wien: np.ndarray, mix: float = 0.5) -> np.ndarray:
    mix = float(np.clip(mix, 0.0, 1.0))
    return mix * g_sub + (1.0 - mix) * g_wien
