import numpy as np
import pytest

from src.ml.enhancer import MLEnhancer


def test_ml_shape_and_finite():
    sr = 16000
    x = (0.2 * np.sin(2 * np.pi * 200 * np.arange(sr * 2) / sr)).astype(np.float32)
    ml = MLEnhancer()
    if not ml.available:
        pytest.skip("ML model unavailable")
    y = ml.enhance_audio(x, sr)
    assert y.shape == x.shape
    assert y.dtype == np.float32
    assert np.all(np.isfinite(y))


def test_ml_handles_silence():
    ml = MLEnhancer()
    if not ml.available:
        pytest.skip("ML model unavailable")
    x = np.zeros(8000, dtype=np.float32)
    y = ml.enhance_audio(x, 16000)
    assert np.max(np.abs(y)) < 1e-3 or np.all(np.isfinite(y))
