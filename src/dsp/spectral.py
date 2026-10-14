"""Spectral processing helpers and windowing."""

from __future__ import annotations

import numpy as np
from scipy import signal


def hann_window(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("window size must be positive")
    return signal.windows.hann(n, sym=False).astype(np.float64)


def stft_frames(audio: np.ndarray, n_fft: int = 1024, hop_length: int = 256,
                win_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (complex STFT matrix shape (frames, bins), window)."""
    audio = np.asarray(audio, dtype=np.float64).ravel()
    if win_length is None:
        win_length = n_fft
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    window = signal.windows.hann(win_length, sym=False)
    # pad
    pad = n_fft // 2
    padded = np.pad(audio, (pad, pad), mode="reflect")
    frames = (len(padded) - n_fft) // hop_length + 1
    if frames <= 0:
        frames = 1
    stft = np.zeros((frames, n_fft // 2 + 1), dtype=np.complex128)
    for i in range(frames):
        start = i * hop_length
        chunk = padded[start:start + n_fft]
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        windowed = chunk * window[:len(chunk)] if len(window) >= len(chunk) else chunk * np.ones_like(chunk)
        # use full window trimmed
        w = window * 1.0
        if len(w) != n_fft:
            wfull = np.zeros(n_fft)
            wfull[:min(len(w), n_fft)] = w[:min(len(w), n_fft)]
            w = wfull
        windowed = chunk * w
        stft[i] = np.fft.rfft(windowed, n=n_fft)
    return stft, window


def istft_frames(stft: np.ndarray, hop_length: int = 256, n_fft: int = 1024,
                 window: np.ndarray | None = None, length: int | None = None) -> np.ndarray:
    """Inverse STFT (overlap-add)."""
    stft = np.asarray(stft)
    if stft.ndim != 2:
        raise ValueError("stft must be 2D")
    frames, bins = stft.shape
    n_fft = (bins - 1) * 2
    if window is None:
        window = signal.windows.hann(n_fft, sym=False)
    if len(window) != n_fft:
        w = np.zeros(n_fft)
        w[:min(len(window), n_fft)] = window[:min(len(window), n_fft)]
        window = w
    pad = n_fft // 2
    total = (frames - 1) * hop_length + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for i in range(frames):
        spec = np.asarray(stft[i], dtype=np.complex128)
        chunk = np.fft.irfft(spec, n=n_fft).real
        start = i * hop_length
        chunk = chunk * window
        out[start:start + n_fft] += chunk
        norm[start:start + n_fft] += window ** 2
    # normalize
    nz = norm > 1e-8
    out[nz] /= norm[nz]
    # trim padding
    if length is not None:
        # total includes pad both sides; reconstruct original length by trimming pad
        start = pad
        end = start + length
        if end <= len(out):
            out = out[start:end]
        else:
            out = out[start:]
    else:
        out = out[pad:pad + ((frames - 1) * hop_length + n_fft - pad)]
    return out


def stft_mag_phase(audio, n_fft=1024, hop_length=256):
    stft, win = stft_frames(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    phase = np.angle(stft)
    return mag, phase, stft, win


def smooth_frequency(gain: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Smooth gain across frequency bins."""
    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim == 1:
        k = max(1, kernel_size)
        if k == 1:
            return gain.copy()
        pad = k // 2
        padded = np.pad(gain, (pad, pad), mode="edge")
        kernel = np.ones(k) / k
        return np.convolve(padded, kernel, mode="valid")
    # 2D: smooth over last axis
    out = np.zeros_like(gain)
    for i in range(gain.shape[0]):
        out[i] = smooth_frequency(gain[i], kernel_size)
    return out


def smooth_time(gain: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Temporal smoothing over frames (first axis)."""
    gain = np.asarray(gain, dtype=np.float64)
    if gain.ndim < 2:
        return gain.copy()
    out = gain.copy()
    for i in range(1, gain.shape[0]):
        out[i] = alpha * out[i - 1] + (1 - alpha) * gain[i]
    return out
