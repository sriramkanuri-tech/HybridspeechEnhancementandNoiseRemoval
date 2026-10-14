"""Real-time audio engine.

The engine keeps all heavy processing out of the PortAudio callbacks.  The
callbacks only copy frames and move them through bounded queues, while a
separate worker thread performs enhancement.  This prevents callback overruns
and avoids unbounded latency.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import psutil
import sounddevice as sd

from src.config.settings import AppConfig
from .devices import query_devices

logger = logging.getLogger("speech_enhancement.audio")


@dataclass
class EngineStatus:
    running: bool = False
    error: Optional[str] = None
    input_active: bool = True
    output_active: bool = True
    input_rms: float = 0.0
    output_rms: float = 0.0
    input_peak: float = 0.0
    output_peak: float = 0.0
    underflows: int = 0
    overflows: int = 0
    cpu_percent: float = 0.0
    process_ms: float = 0.0
    latency_ms: float = 0.0
    queue_depth: int = 0
    input_device_actual: Optional[int] = None
    output_device_actual: Optional[int] = None
    sample_rate_actual: Optional[int] = None
    channels_actual: Optional[int] = None
    model_mode: str = "dsp"


class AudioEngine:
    def __init__(self, cfg: AppConfig, pipeline, ml_enhancer=None):
        self.cfg = cfg
        self.pipeline = pipeline
        self.ml = ml_enhancer
        self.status = EngineStatus()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._in_stream = None
        self._out_stream = None
        self._monitor_stream = None
        self._lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._proc = psutil.Process()
        self._last_cpu = time.time()
        self._started = False
        self._metrics_thread = None
        self._in_q: queue.Queue = queue.Queue(maxsize=getattr(cfg, "buffer_blocks", 8) or 8)
        self._out_q: queue.Queue = queue.Queue(maxsize=getattr(cfg, "buffer_blocks", 8) or 8)
        self._monitor_q: queue.Queue = queue.Queue(maxsize=getattr(cfg, "buffer_blocks", 8) or 8)
        self._monitor_stream = None

    # ------------------------------------------------------------------ utils

    def _rms(self, x):
        x = np.asarray(x)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

    def _queue_size(self, q):
        try:
            return q.qsize()
        except NotImplementedError:
            # macOS may not support qsize.
            return 0

    def _apply_config_snapshot(self):
        """Copy mutable config values used by the worker onto local attributes."""
        self._block_size = int(self.cfg.block_size)
        self._sample_rate = int(self.cfg.sample_rate)
        self._channels = int(self.cfg.channels)
        self._buffer_blocks = max(1, int(self.cfg.buffer_blocks))
        self._use_ml = bool(getattr(self.cfg, "use_ml", False))
        self._use_hybrid = bool(getattr(self.cfg, "use_hybrid", False))
        self._monitor_enabled = bool(getattr(self.cfg, "monitor_enabled", False))
        self._monitor_device = getattr(self.cfg, "monitor_device", None)
        self._noise_reduction = float(getattr(self.cfg, "noise_reduction", 0.6))
        self._voice_preservation = float(getattr(self.cfg, "voice_preservation", 0.7))

    def _reset_status(self):
        self.status = EngineStatus()
        self._in_q = queue.Queue(maxsize=self._buffer_blocks)
        self._out_q = queue.Queue(maxsize=self._buffer_blocks)
        self._monitor_q = queue.Queue(maxsize=self._buffer_blocks)

    def _update_status_value(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)

    # ---------------------------------------------------------------- callbacks

    def input_callback(self, indata, frames, time_info, status):
        # Minimal work only: copy, meter, enqueue.
        try:
            data = np.asarray(indata).astype(np.float32, copy=True)
        except Exception:
            return
        try:
            if data.ndim > 1:
                mono = data.mean(axis=1)
            else:
                mono = data.ravel()
            with self._lock:
                self.status.input_rms = self._rms(mono)
                self.status.input_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        except Exception:
            pass

        if getattr(status, "input_overflow", False):
            with self._lock:
                self.status.overflows += 1

        try:
            self._in_q.put_nowait(data)
        except queue.Full:
            with self._lock:
                self.status.overflows += 1

    def output_callback(self, outdata, frames, time_info, status):
        try:
            data = self._out_q.get_nowait()
        except queue.Empty:
            with self._lock:
                self.status.underflows += 1
            outdata[:] = 0.0
            return
        out = np.asarray(data)
        # Shape/channel adaptation.
        if out.ndim == 0:
            out = out.reshape(1, 1)
        if out.dtype != outdata.dtype:
            out = out.astype(outdata.dtype, copy=False)
        need = outdata.shape
        if out.shape != need:
            flat = out.reshape(-1)
            required = int(np.prod(need))
            if flat.size >= required:
                arr = flat[:required].reshape(need)
            else:
                arr = np.zeros(required, dtype=out.dtype)
                arr[:flat.size] = flat
                arr = arr.reshape(need)
            outdata[:] = arr
        else:
            outdata[:] = out

        try:
            if outdata.ndim > 1:
                mono = outdata.mean(axis=1)
            else:
                mono = outdata.ravel()
            with self._lock:
                self.status.output_rms = self._rms(mono)
                self.status.output_peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        except Exception:
            pass

        if getattr(status, "output_underflow", False):
            with self._lock:
                self.status.underflows += 1

    def monitor_callback(self, outdata, frames, time_info, status):
        try:
            data = self._monitor_q.get_nowait()
        except queue.Empty:
            with self._lock:
                self.status.underflows += 1
            outdata[:] = 0.0
            return
        out = np.asarray(data)
        if out.dtype != outdata.dtype:
            out = out.astype(outdata.dtype, copy=False)
        if out.shape != outdata.shape:
            flat = out.reshape(-1)
            required = int(np.prod(outdata.shape))
            if flat.size >= required:
                arr = flat[:required].reshape(outdata.shape)
            else:
                arr = np.zeros(required, dtype=out.dtype)
                arr[:flat.size] = flat
                arr = arr.reshape(outdata.shape)
            outdata[:] = arr
        else:
            outdata[:] = out

    # ---------------------------------------------------------------- workers

    def _process_block(self, block: np.ndarray) -> np.ndarray:
        """Process one captured block and return output with the same frame count."""
        x = np.asarray(block)
        if x.ndim == 2 and x.shape[1] > 1 and self._channels == 1:
            x = x.mean(axis=1, keepdims=True)
        if x.ndim == 2:
            audio_in = x[:, 0] if x.shape[1] == 1 else x.mean(axis=1)
        else:
            audio_in = x
        audio_in = np.asarray(audio_in, dtype=np.float32).reshape(-1)
        bs = self._block_size

        t0 = time.perf_counter()
        if self._use_hybrid:
            if self.ml is None or not getattr(self.ml, "available", False):
                raise RuntimeError(getattr(self.ml, "detail", "ML model is not available."))
            try:
                self.pipeline.update_params(noise_reduction=self._noise_reduction * 0.5,
                                            voice_preservation=self._voice_preservation)
            except Exception:
                pass
            ml_out = self.ml.enhance_audio(audio_in, self._sample_rate)
            y = self.pipeline.process_audio(np.asarray(ml_out, dtype=np.float32))
            mode = "hybrid"
        elif self._use_ml:
            if self.ml is None or not getattr(self.ml, "available", False):
                raise RuntimeError(getattr(self.ml, "detail", "ML model is not available."))
            y = self.ml.enhance_audio(audio_in, self._sample_rate)
            mode = "ml"
        else:
            # Reset DSP state is managed at start/stop. Keep params current.
            try:
                self.pipeline.update_params(
                    noise_reduction=self._noise_reduction,
                    voice_preservation=self._voice_preservation,
                )
            except Exception:
                pass
            y = self.pipeline.process_audio(audio_in)
            mode = "dsp"

        y = np.asarray(y, dtype=np.float32).reshape(-1)

        # Channel conversion.
        if self._channels == 2:
            if y.ndim == 1:
                y = np.column_stack([y, y])
            if y.ndim == 2 and y.shape[1] == 1:
                y = np.column_stack([y[:, 0], y[:, 0]])
        else:
            if y.ndim > 1:
                y = y.reshape(-1)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim == 2 and y.shape[1] != self._channels:
            if self._channels == 1:
                y = y[:, 0].reshape(-1, 1)
            else:
                y = np.column_stack([y[:, 0], y[:, 0]])

        # Frame count match to avoid callback desync.
        if y.shape[0] != bs:
            if y.shape[0] > bs:
                y = y[:bs, ...]
            else:
                pad = bs - y.shape[0]
                if y.ndim == 2:
                    y = np.pad(y, ((0, pad), (0, 0)))
                else:
                    y = np.pad(y, (0, pad))
                y = y.reshape(-1, self._channels if self._channels > 1 else 1)

        y = np.asarray(y, dtype=np.float32)
        dt = (time.perf_counter() - t0) * 1000.0
        logger.debug("processed block mode=%s frames=%d duration_ms=%.2f", mode, y.shape[0], dt)
        self._update_status_value(process_ms=dt, model_mode=mode)
        return y.astype(np.float32)

    def _processor(self):
        while not self._stop_event.is_set():
            # Detect stream deactivation/device disconnection without blocking callbacks.
            if self._in_stream is not None or self._out_stream is not None:
                in_ok = True
                out_ok = True
                try:
                    if self._in_stream is not None:
                        in_ok = bool(getattr(self._in_stream, "active", True))
                    if self._out_stream is not None:
                        out_ok = bool(getattr(self._out_stream, "active", True))
                except Exception:
                    in_ok = False
                    out_ok = False
                self._update_status_value(input_active=in_ok, output_active=out_ok)
                if not in_ok or not out_ok:
                    self._update_status_value(error="Audio device disconnected or stopped.")
                    self._stop_event.set()
                    return
            try:
                block = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                processed = self._process_block(block)
            except Exception as e:
                logger.exception("Audio processing failed")
                self._update_status_value(error=str(e))
                self._stop_event.set()
                return
            # Queue depth diagnostic.
            out_depth = self._queue_size(self._out_q)
            self._update_status_value(queue_depth=out_depth)
            try:
                self._out_q.put(processed, timeout=0.2)
            except queue.Full:
                with self._lock:
                    self.status.overflows += 1
                logger.warning("Virtual output queue full; dropping processed block")
            if self._monitor_enabled:
                try:
                    self._monitor_q.put(processed.copy(), timeout=0.2)
                except queue.Full:
                    with self._lock:
                        self.status.overflows += 1
                    logger.warning("Monitor output queue full; dropping processed block")            # CPU metering.
            try:
                with self._lock:
                    self.status.cpu_percent = self._proc.cpu_percent(interval=None)
            except Exception:
                pass

    # ---------------------------------------------------------------- public

    def start(self):
        with self._stream_lock:
            if self._started:
                logger.warning("start() ignored: engine already running")
                return
            self._apply_config_snapshot()
            self._stop_event.clear()
            self._reset_status()
            self._update_status_value(input_device_actual=self.cfg.input_device,
                                      output_device_actual=self.cfg.output_device,
                                      sample_rate_actual=self._sample_rate,
                                      channels_actual=self._channels,
                                      model_mode="ml" if self._use_ml else "dsp")
            self._proc.cpu_percent(interval=None)

            streams_started = False
            try:
                in_stream = sd.InputStream(
                    device=self.cfg.input_device,
                    channels=self._channels,
                    samplerate=self._sample_rate,
                    blocksize=self._block_size,
                    dtype="float32",
                    callback=self.input_callback,
                )
                out_stream = sd.OutputStream(
                    device=self.cfg.output_device,
                    channels=self._channels,
                    samplerate=self._sample_rate,
                    blocksize=self._block_size,
                    dtype="float32",
                    callback=self.output_callback,
                )
                in_stream.start()
                out_stream.start()
                monitor_stream = None
                if self._monitor_enabled and self._monitor_device is not None:
                    monitor_stream = sd.OutputStream(
                        device=self._monitor_device,
                        channels=self._channels,
                        samplerate=self._sample_rate,
                        blocksize=self._block_size,
                        dtype="float32",
                        callback=self.monitor_callback,
                    )
                    monitor_stream.start()
                streams_started = True
            except Exception as e:
                logger.exception("Failed to open audio streams")
                # Clean up any partially-open stream.
                for s in (in_stream if "in_stream" in locals() else None,
                          out_stream if "out_stream" in locals() else None,
                          monitor_stream if "monitor_stream" in locals() else None):
                    try:
                        if s is not None:
                            s.stop()
                            s.close()
                    except Exception:
                        pass
                self._update_status_value(error=str(e))
                return

            self._in_stream = in_stream
            self._out_stream = out_stream
            self._monitor_stream = monitor_stream
            self._started = True
            self.status.running = True

            # Store actual opened identifiers for diagnostics.
            try:
                self._update_status_value(
                    input_device_actual=getattr(in_stream, "device", self.cfg.input_device),
                    output_device_actual=getattr(out_stream, "device", self.cfg.output_device),
                    sample_rate_actual=int(getattr(in_stream, "samplerate", self._sample_rate)),
                    channels_actual=int(getattr(in_stream, "channels", self._channels)),
                )
            except Exception:
                pass

            self._thread = threading.Thread(target=self._processor, daemon=True, name="audio-processor")
            self._thread.start()

            logger.info(
                "Audio engine started input=%s output=%s sample_rate=%d channels=%d "
                "block_size=%d buffer_blocks=%d use_ml=%s",
                self.cfg.input_device, self.cfg.output_device, self._sample_rate,
                self._channels, self._block_size, self._buffer_blocks, self._use_ml,
            )
            # Opened device diagnostic for GUI logs is emitted separately by app.
            logger.info(
                "Opened input device=%s output device=%s samplerate=%d",
                getattr(in_stream, "device", None), getattr(out_stream, "device", None),
                int(getattr(in_stream, "samplerate", self._sample_rate)),
            )

    def stop(self):
        with self._stream_lock:
            if not self._started:
                # Still make sure status is cleared.
                self.status.running = False
                return
            self._stop_event.set()
            self.status.running = False
            for s in (self._in_stream, self._out_stream, self._monitor_stream):
                try:
                    if s is not None:
                        s.stop()
                        s.close()
                except Exception as e:
                    logger.debug("Stream close error: %s", e)
            self._in_stream = None
            self._out_stream = None
            self._monitor_stream = None
            if self._thread is not None:
                self._thread.join(timeout=3.0)
                self._thread = None
            self._started = False

        # Reset DSP state so a subsequent run starts with a fresh noise estimate.
        try:
            if hasattr(self.pipeline, "reset"):
                self.pipeline.reset()
            elif hasattr(self.pipeline, "reducer") and hasattr(self.pipeline.reducer, "reset"):
                self.pipeline.reducer.reset()
        except Exception as e:
            logger.debug("Pipeline reset failed: %s", e)

        logger.info("Audio engine stopped")

    def get_status(self) -> EngineStatus:
        with self._lock:
            return EngineStatus(**self.status.__dict__)

    # Convenience helpers used by tests/UI.
    def is_running(self) -> bool:
        return self._started
