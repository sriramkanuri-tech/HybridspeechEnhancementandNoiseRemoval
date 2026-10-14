"""Speech enhancement pipeline wrapping the V4.1 reducer."""

from __future__ import annotations

import numpy as np

from .v41_reducer import V41Reducer


class EnhancementPipeline:
    def __init__(
        self,
        noise_reduction: float = 0.55,
        voice_preservation: float = 0.72,
        sample_rate: int = 16000,
        auto_gain: bool = False,
        target_rms: float = 0.05,
        smoothing: float = 0.85,
        frame_size: int = 2048,
        hop_size: int = 512,
        noise_profile=None,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.noise_reduction = float(np.clip(noise_reduction, 0.0, 1.0))
        self.voice_preservation = float(np.clip(voice_preservation, 0.0, 1.0))
        self.sample_rate = sample_rate
        self.auto_gain = auto_gain
        self.target_rms = target_rms
        self.smoothing = smoothing
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.reducer = V41Reducer(
            sample_rate=sample_rate,
            frame_size=frame_size,
            hop_size=hop_size,
            reduction_strength=self.noise_reduction,
            minimum_gain=0.35 + 0.6 * self.voice_preservation,
            smoothing=smoothing,
        )

    def reset(self):
        try:
            self.reducer.reset()
        except Exception:
            pass
        try:
            self.noise_profile = None
        except Exception:
            pass

    def update_params(self, noise_reduction=None, voice_preservation=None, smoothing=None):
        if noise_reduction is not None:
            self.noise_reduction = float(np.clip(noise_reduction, 0.0, 1.0))
            self.reducer.reduction_strength = self.noise_reduction
        if voice_preservation is not None:
            self.voice_preservation = float(np.clip(voice_preservation, 0.0, 1.0))
            self.reducer.minimum_gain = 0.35 + 0.6 * self.voice_preservation
        if smoothing is not None:
            self.smoothing = float(np.clip(smoothing, 0.0, 0.999))
            self.reducer.smoothing = self.smoothing

    def _process_channel(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return audio.copy()
        out = self.reducer.process(audio)
        if self.auto_gain:
            from .filters import normalize_gain, clip_protect
            out = normalize_gain(out, target_rms=self.target_rms, max_gain=2.0)
            out = clip_protect(out)
        return out

    def process_audio(self, audio):
        audio = np.asarray(audio)
        if audio.ndim == 1:
            return self._process_channel(audio).astype(np.float32)
        if audio.ndim == 2:
            if audio.shape[0] < audio.shape[1] and audio.shape[0] <= 8:
                out_ch = [self._process_channel(audio[c]) for c in range(audio.shape[0])]
                return np.stack(out_ch, axis=0).astype(np.float32)
            out = np.zeros_like(audio, dtype=np.float32)
            for c in range(audio.shape[1]):
                out[:, c] = self._process_channel(audio[:, c])
            return out.astype(np.float32)
        raise ValueError("audio must be 1D or 2D")

    def calibrate(self, noise_audio):
        audio = np.asarray(noise_audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1) if audio.shape[-1] <= 8 else np.mean(audio, axis=0)
        self.reducer.estimate_noise(np.asarray(audio, dtype=np.float32).ravel())
