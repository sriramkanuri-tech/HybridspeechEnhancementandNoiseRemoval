import numpy as np
import pytest

from src.dsp.pipeline import EnhancementPipeline
from src.dsp.filters import normalize_gain, clip_protect, spectral_subtraction_gain
from src.dsp.noise import calibrate_noise_profile, voice_activity


def test_process_shape_and_dtype():
    sr = 16000
    x = (0.2 * np.sin(2 * np.pi * 200 * np.arange(sr) / sr)).astype(np.float32)
    p = EnhancementPipeline(noise_reduction=0.5, voice_preservation=0.6, sample_rate=sr)
    y = p.process_audio(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


def test_silence_handling():
    x = np.zeros(4000, dtype=np.float32)
    p = EnhancementPipeline()
    y = p.process_audio(x)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.01


def test_clipping_protection():
    x = np.ones(1000, dtype=np.float32) * 0.99
    p = EnhancementPipeline(auto_gain=False)
    y = p.process_audio(x)
    assert np.max(np.abs(y)) <= 1.0


def test_noise_calibration_shape():
    mags = np.abs(np.random.default_rng(0).normal(size=(10, 513)))
    prof = calibrate_noise_profile(mags)
    assert prof.shape == (513,)
    assert np.all(prof >= 0)


def test_gain_bounds():
    mag = np.abs(np.random.default_rng(1).normal(size=64)) + 0.1
    nse = np.abs(np.random.default_rng(2).normal(size=64)) + 0.01
    g = spectral_subtraction_gain(mag, nse, strength=2.0)
    assert np.all(g >= 0) and np.all(g <= 1.0001)


def test_voice_activity_bool():
    nse = np.ones(10) * 0.01
    mag = np.ones(10) * 0.5
    assert isinstance(bool(voice_activity(mag, nse)), bool)


def test_normalize_gain():
    x = np.ones(100) * 0.01
    y = normalize_gain(x, target_rms=0.1, max_gain=5.0)
    assert np.sqrt(np.mean(y ** 2)) <= 0.1 + 1e-6
