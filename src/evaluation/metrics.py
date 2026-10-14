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


def snr_db(clean, noisy):
    clean = _mono(clean)
    noise = _mono(noisy) - clean
    n = min(len(clean), len(noise))
    clean = clean[:n]
    noise = noise[:n]
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean(noise ** 2) + 1e-12)
    return float(10 * np.log10(signal_power / noise_power + 1e-12))


def snr_improvement(clean, noisy, enhanced):
    return snr_db(clean, enhanced) - snr_db(clean, noisy)


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