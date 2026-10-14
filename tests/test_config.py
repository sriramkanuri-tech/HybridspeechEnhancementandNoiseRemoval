from src.config.settings import AppConfig, default_config_path
import tempfile
from pathlib import Path

def test_config_roundtrip():
    cfg = AppConfig(noise_reduction=0.3, voice_preservation=0.8)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        cfg.save(p)
        loaded = AppConfig.load(p)
        assert loaded.noise_reduction == 0.3
        assert loaded.voice_preservation == 0.8

def test_config_missing():
    cfg = AppConfig.load(Path("nonexistent.json"))
    assert cfg.sample_rate == 48000
