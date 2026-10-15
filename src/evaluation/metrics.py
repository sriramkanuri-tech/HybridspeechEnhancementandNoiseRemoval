"""Evaluation metrics."""

from __future__ import annotations

import numpy as np


def _mono(x):
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=-1) if x.shape[-1] <= 8 else x.mean(axis=0)
    return np.asarray(x, dtype=np.float64).ravel()


def rms(x):
    x = _mono(x)
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def rms_dbfs(x):
    return float(20 * np.log10(rms(x) + 1e-12))


def peak_amplitude(x):
    x = _mono(x)
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def peak_dbfs(x):
    return float(20 * np.log10(peak_amplitude(x) + 1e-12))


def crest_factor(x):
    r = rms(x)
    p = peak_amplitude(x)
    return float(p / r if r > 0 else 0.0)


def clipping_pct(x, threshold=0.99):
    x = _mono(x)
    if x.size == 0:
        return 0.0
    return float(100.0 * np.mean(np.abs(x) >= threshold))


def snr_db(clean, noisy, clean_sr=None, noisy_sr=None):
    """Reference-based SNR in dB.

    clean and noisy must represent the same recording and time alignment. The
    signals are trimmed only to a common finite length; one signal is never used
    as its own clean reference. Return None when the measurement is undefined.
    """
    if clean_sr is not None and noisy_sr is not None and int(clean_sr) != int(noisy_sr):
        raise ValueError("clean and noisy sample rates must match for SNR")
    clean = _mono(clean)
    noisy = _mono(noisy)
    if clean.size == 0 or noisy.size == 0:
        return None
    n = min(clean.size, noisy.size)
    if n < 1:
        return None
    clean = clean[:n].astype(np.float64)
    noisy = noisy[:n].astype(np.float64)
    if not (np.all(np.isfinite(clean)) and np.all(np.isfinite(noisy))):
        return None
    signal_power = float(np.sum(clean * clean))
    noise_power = float(np.sum((noisy - clean) * (noisy - clean)))
    if not np.isfinite(signal_power) or not np.isfinite(noise_power):
        return None
    if noise_power <= 1e-18:
        # Exact equality cannot produce a finite, comparable SNR.
        return None
    if signal_power < 0.0:
        return None
    return float(10.0 * np.log10(max(signal_power, 0.0) / noise_power))


def snr_improvement(clean, noisy, enhanced):
    """SNR improvement in dB. Returns None if either reference measurement is unavailable."""
    base = snr_db(clean, noisy)
    out = snr_db(clean, enhanced)
    if base is None or out is None:
        return None
    value = out - base
    if not np.isfinite(value):
        return None
    return float(value)


def rt_factor(process_time, duration):
    duration = max(float(duration), 1e-9)
    return float(process_time / duration)


def frequency_band_energy(x, sr, low, high):
    x = _mono(x)
    if x.size == 0:
        return 0.0
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    # energy as sum of squared magnitude in band
    band = np.abs(fft[mask]) ** 2
    return float(np.sum(band) / (len(x) + 1e-12))


def safe_pct_change(old, new):
    if abs(old) < 1e-12:
        return None
    return float((new - old) / abs(old) * 100.0)


def _trim_for_metric(clean, enhanced):
    n = min(len(clean), len(enhanced))
    return clean[:n], enhanced[:n]


def pesq_score(clean, enhanced, sr):
    """Optional PESQ score. Returns None if unavailable or unsupported."""
    clean = _mono(clean)
    enhanced = _mono(enhanced)
    clean, enhanced = _trim_for_metric(clean, enhanced)
    if clean.size == 0 or enhanced.size == 0:
        return None
    try:
        from pesq import pesq as pesq_fn
    except Exception:
        return None
    try:
        mode = "wb" if int(sr) >= 16000 else "nb"
        return float(pesq_fn(int(sr), clean, enhanced, mode))
    except Exception:
        return None


def stoi_score(clean, enhanced, sr):
    """Optional STOI score. Returns None if unavailable or unsupported."""
    clean = _mono(clean)
    enhanced = _mono(enhanced)
    clean, enhanced = _trim_for_metric(clean, enhanced)
    if clean.size == 0 or enhanced.size == 0:
        return None
    try:
        from pystoi import stoi as stoi_fn
    except Exception:
        return None
    try:
        return float(stoi_fn(clean, enhanced, int(sr), extended=False))
    except Exception:
        return None