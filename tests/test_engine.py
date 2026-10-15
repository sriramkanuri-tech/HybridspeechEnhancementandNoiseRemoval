import numpy as np
import pytest

from src.config.settings import AppConfig
from src.dsp.pipeline import EnhancementPipeline
from src.audio.engine import AudioEngine


def test_engine_status_invalid_device():
    cfg = AppConfig(sample_rate=48000, channels=1, block_size=480, buffer_blocks=4,
                    input_device=99999, output_device=99998)
    pipe = EnhancementPipeline(sample_rate=48000)
    eng = AudioEngine(cfg, pipe)
    eng.start()
    st = eng.get_status()
    assert st.running is False
    assert st.error is not None
    eng.stop()


def test_engine_config_update():
    cfg = AppConfig(sample_rate=16000, channels=1, block_size=256, buffer_blocks=2,
                    input_device=None, output_device=None)
    pipe = EnhancementPipeline(sample_rate=16000)
    eng = AudioEngine(cfg, pipe)
    assert eng.cfg.block_size == 256


class _FakeML:
    def __init__(self, value=0.25, fail=False):
        self.status_available = True
        self.available = True
        self.detail = "fake ml"
        self.value = value
        self.fail = fail

    def enhance_audio(self, audio, sample_rate=16000):
        audio = np.asarray(audio)
        if self.fail:
            raise RuntimeError("boom")
        return np.full(audio.shape, self.value, dtype=np.float32)


def test_engine_process_block_uses_ml():
    cfg = AppConfig(sample_rate=16000, channels=1, block_size=4, buffer_blocks=2,
                    input_device=None, output_device=None, use_ml=True)
    pipe = EnhancementPipeline(sample_rate=16000)
    ml = _FakeML(value=0.25)
    eng = AudioEngine(cfg, pipe, ml)
    eng._apply_config_snapshot()
    assert eng._use_ml is True
    out = eng._process_block(np.ones(4, dtype=np.float32))
    assert out.shape == (4, 1)
    assert np.allclose(out, 0.25)


def test_engine_process_block_ml_failure():
    cfg = AppConfig(sample_rate=16000, channels=1, block_size=4, buffer_blocks=2,
                    input_device=None, output_device=None, use_ml=True)
    pipe = EnhancementPipeline(sample_rate=16000)
    ml = _FakeML(fail=True)
    eng = AudioEngine(cfg, pipe, ml)
    eng._apply_config_snapshot()
    out = eng._process_block(np.ones(4, dtype=np.float32))
    assert out.shape == (4, 1)
    assert np.all(np.isfinite(out))
    st = eng.get_status()
    assert st.error is not None
    assert "bypass" in st.error.lower()


def test_engine_process_block_dsp():
    cfg = AppConfig(sample_rate=16000, channels=1, block_size=4, buffer_blocks=2,
                    input_device=None, output_device=None, use_ml=False,
                    noise_reduction=0.5, voice_preservation=0.6)
    pipe = EnhancementPipeline(sample_rate=16000)
    eng = AudioEngine(cfg, pipe)
    eng._apply_config_snapshot()
    x = (0.2 * np.sin(2 * np.pi * 50 * np.arange(4) / 16000)).astype(np.float32)
    out = eng._process_block(x)
    assert out.shape == (4, 1)
    assert np.all(np.isfinite(out))


def test_pipeline_reset():
    p = EnhancementPipeline(sample_rate=16000)
    p._process_channel(np.zeros(1000, dtype=np.float32))
    p.reset()
    assert p.reducer.noise_power is None or p.reducer.previous_gain.shape[0] == (p.reducer.frame_size // 2 + 1)
