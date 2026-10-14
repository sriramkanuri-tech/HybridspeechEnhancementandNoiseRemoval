"""Optional ONNX-based ML speech enhancement using the supplied V3 CNN."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy import signal
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

MODEL_FILENAME = "speech_enhancement_cnn_v3.onnx"
logger = logging.getLogger("speech_enhancement.ml")
CKPT_FILENAME = "speech_enhancement_cnn_v3.pth"


def discover_model_path(model_path=None):
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    env = os.environ.get("SPEECH_ENHANCEMENT_MODEL")
    if env:
        candidates.append(Path(env))
    root = Path(__file__).resolve().parents[2]
    candidates.append(root / "models" / MODEL_FILENAME)
    candidates.append(Path("models") / MODEL_FILENAME)
    try:
        from src.config.paths import models_dir
        candidates.append(models_dir() / MODEL_FILENAME)
    except Exception:
        pass
    candidates.append(Path("C:/Users/srira/OneDrive/Desktop/Speech-Enhancement-and-Noise-Removal/src/Phase2/ML") / MODEL_FILENAME)
    for c in candidates:
        if c.is_file():
            return c
    return None


class V3OnnxProcessor:
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 256

    def __init__(self, onnx_path, x_mean=0.0, x_std=1.0, y_mean=0.0, y_std=1.0):
        self.onnx_path = str(onnx_path)
        self.x_mean = float(x_mean)
        self.x_std = max(float(x_std), 1e-8)
        self.y_mean = float(y_mean)
        self.y_std = max(float(y_std), 1e-8)
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def enhance_magnitude(self, magnitude):
        import time
        magnitude = np.asarray(magnitude, dtype=np.float32)
        normalized = (magnitude - self.x_mean) / self.x_std
        tensor = normalized[None, None, :, :].astype(np.float32)
        t0 = time.perf_counter()
        prediction = self.session.run([self.output_name], {self.input_name: tensor})[0][0, 0]
        dt = (time.perf_counter() - t0) * 1000.0
        enhanced = prediction * self.y_std + self.y_mean
        out = np.maximum(enhanced, 0.0).astype(np.float32)
        logger.debug("V3 inference input=%s output=%s duration_ms=%.2f", tensor.shape, out.shape, dt)
        return out

    def process_audio(self, audio, sample_rate=16000):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        orig_len = audio.size
        if orig_len == 0:
            return audio.copy()
        original_sr = int(sample_rate)
        target_sr = self.SAMPLE_RATE
        if original_sr != target_sr:
            n = max(1, int(round(orig_len * target_sr / original_sr)))
            audio16 = signal.resample(audio, n).astype(np.float32)
        else:
            audio16 = audio.copy()
        # STFT
        window = hann(self.N_FFT, sym=False)
        SFT = ShortTimeFFT(win=window, hop=self.HOP_LENGTH, fs=target_sr, fft_mode="onesided")
        spec = SFT.stft(audio16.astype(np.float32))
        mag = np.abs(spec).astype(np.float32)
        phase = np.angle(spec)
        enhanced_mag = self.enhance_magnitude(mag)
        spec_enh = enhanced_mag.astype(np.complex64) * np.exp(1j * phase.astype(np.float32))
        out16 = SFT.istft(spec_enh, k0=0, k1=audio16.size)
        out16 = np.real(out16).astype(np.float32).ravel()
        out16 = np.real(out16).astype(np.float32).ravel()
        if out16.size > audio16.size:
            out16 = out16[:audio16.size]
        elif out16.size < audio16.size:
            out16 = np.pad(out16, (0, audio16.size - out16.size))
        if original_sr != target_sr:
            out = signal.resample(out16, orig_len).astype(np.float32)
        else:
            out = out16
        if out.size != orig_len:
            if out.size > orig_len:
                out = out[:orig_len]
            else:
                out = np.pad(out, (0, orig_len - out.size))
        # clipping protection
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.98:
            out = (out / peak * 0.95).astype(np.float32)
        return out.astype(np.float32)


class MLEnhancer:
    def __init__(self, model_path=None):
        self.status_available = False
        self.status_detail = "ML model not loaded."
        self.processor = None
        mp = discover_model_path(model_path)
        if mp is None:
            self.status_detail = "ONNX model not found."
            return
        # Normalization stats from the supplied checkpoint (hardcoded to avoid
        # requiring torch at runtime). Values: X_mean 0.1973, X_std 0.3839,
        # Y_mean 0.1638, Y_std 0.3616.
        stats = {"x_mean": 0.19731217622756958, "x_std": 0.3838746249675751,
                 "y_mean": 0.1638099104166031, "y_std": 0.36157527565956116}
        try:
            self.processor = V3OnnxProcessor(
                mp,
                x_mean=stats.get("x_mean", 0.0),
                x_std=stats.get("x_std", 1.0),
                y_mean=stats.get("y_mean", 0.0),
                y_std=stats.get("y_std", 1.0),
            )
            self.status_available = True
            self.status_detail = f"Loaded V3 ONNX model from {mp}."
        except Exception as e:
            self.status_detail = f"Failed to load ONNX model: {e}"
            self.processor = None
            self.status_available = False

    @property
    def available(self):
        return self.status_available

    @property
    def detail(self):
        return self.status_detail

    def enhance_audio(self, audio, sample_rate=16000):
        if self.processor is None:
            raise RuntimeError(self.status_detail)
        return self.processor.process_audio(audio, sample_rate)
