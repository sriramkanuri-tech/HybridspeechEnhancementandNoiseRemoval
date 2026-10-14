"""Microphone recording helpers."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def record_audio(
    duration_seconds: float,
    sample_rate: int,
    channels: int = 1,
    device=None,
    stop_event: threading.Event | None = None,
) -> np.ndarray:
    """Record audio synchronously. Returns (frames, channels) float32 array."""
    if stop_event is None:
        stop_event = threading.Event()
    frames = int(sample_rate * duration_seconds)
    chunks = []
    q = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        q.put(indata.copy())

    stream = sd.InputStream(
        device=device,
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
        callback=callback,
        blocksize=1024,
    )
    stream.start()
    start = time.time()
    collected = 0
    target = frames
    data = np.zeros((0, channels), dtype=np.float32) if channels > 1 else np.zeros(0, dtype=np.float32)
    buffer = []
    try:
        while collected < target and (stop_event is None or not stop_event.is_set()):
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                if not stream.active:
                    break
                continue
            buffer.append(chunk)
            collected += len(chunk)
            if collected >= target:
                break
        # trim
        if buffer:
            arr = np.concatenate(buffer, axis=0)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1) if channels == 1 else arr
            arr = arr[:target]
            if channels == 1:
                arr = arr.reshape(-1)
            data = arr.astype(np.float32)
        else:
            data = (np.zeros(target, dtype=np.float32) if channels == 1 else np.zeros((target, channels), dtype=np.float32))
    finally:
        stream.stop()
        stream.close()
    return data


def save_wav(path, audio, sample_rate, channels=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if channels == 1:
        audio = audio.reshape(-1)
    sf.write(str(path), audio, sample_rate)


def unique_session_dir(root="outputs"):
    ts = time.strftime("session_%Y%m%d_%H%M%S")
    import uuid
    d = Path(root) / f"{ts}_{uuid.uuid4().hex[:6]}"
    d.mkdir(parents=True, exist_ok=True)
    return d
