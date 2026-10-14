"""Application settings and configuration management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AppConfig:
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    monitor_device: Optional[int] = None
    monitor_enabled: bool = False
    use_hybrid: bool = False
    sample_rate: int = 48000
    channels: int = 1
    block_size: int = 480
    buffer_blocks: int = 8
    use_ml: bool = False
    noise_reduction: float = 0.6
    voice_preservation: float = 0.7
    gate_threshold_db: float = -45.0
    auto_gain: bool = True
    target_rms: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls()
        return cls.from_dict(data)


def default_config_path() -> Path:
    base = Path.home() / ".speech-enhancement-noise-removal"
    base.mkdir(parents=True, exist_ok=True)
    return base / "config.json"
