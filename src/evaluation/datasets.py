"""Synthetic test signal generators."""

from __future__ import annotations

import numpy as np


def synthetic_speech(sr: int = 16000, duration: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * duration)) / sr
    # amplitude-modulated harmonic tones resembling voiced speech
    f0 = 120.0
    sig = np.zeros_like(t)
    for h in range(1, 6):
        sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t + rng.random() * 0.2)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    env *= (t % 0.4 < 0.25).astype(float)
    sig = sig * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.3
    return sig.astype(np.float32)


def add_noise(clean, noise_type="white", snr_db=10.0, seed=1):
    rng = np.random.default_rng(seed)
    n = len(clean)
    if noise_type == "white":
        noise = rng.normal(size=n)
    elif noise_type == "babble":
        noise = rng.normal(size=n)
        for _ in range(3):
            noise += 0.5 * np.roll(rng.normal(size=n), rng.integers(0, n))
    elif noise_type == "changing":
        noise = rng.normal(size=n) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * np.arange(n) / 16000))
    else:
        noise = rng.normal(size=n)
    # scale to SNR
    clean_rms = np.sqrt(np.mean(clean ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)
    return (clean + noise).astype(np.float32), noise.astype(np.float32)
