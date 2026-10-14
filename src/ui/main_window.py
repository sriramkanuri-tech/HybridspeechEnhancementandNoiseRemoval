"""Main desktop window with live processing and offline recording/analysis."""

from __future__ import annotations

import os
import sys
import threading
import webbrowser

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except Exception:
    Figure = None
    FigureCanvas = None
    MATPLOTLIB_AVAILABLE = False
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QSlider, QProgressBar, QTextEdit, QTabWidget,
    QGroupBox, QMessageBox, QTableWidget, QTableWidgetItem, QFileDialog, QCheckBox,
)

from src.audio.devices import (
    input_devices, output_devices,
    physical_input_devices, virtual_output_devices, physical_output_devices,
)
from src.audio.engine import AudioEngine
from src.audio.recording import record_audio, save_wav, unique_session_dir
from src.config.settings import AppConfig, default_config_path
from src.dsp.pipeline import EnhancementPipeline
from src.evaluation.workflow import run_methods
from src.ml.enhancer import MLEnhancer


class RecordWorker(QThread):
    progress = Signal(int, float)
    finished = Signal(object, float, bool, str)

    def __init__(self, duration, sample_rate, channels, device, stop_event):
        super().__init__()
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.stop_event = stop_event
        self._cancel = False

    def cancel(self):
        self._cancel = True
        if self.stop_event is not None:
            self.stop_event.set()

    def run(self):
        ev = self.stop_event
        started = time.time()
        audio = record_audio(self.duration, self.sample_rate, self.channels, self.device, ev)
        elapsed = time.time() - started
        cancelled = self._cancel or (ev is not None and ev.is_set() and elapsed < self.duration * 0.95)
        self.finished.emit(audio, elapsed, cancelled, "")


class EnhanceWorker(QThread):
    progress = Signal(str)
    finished = Signal(dict, str)

    def __init__(self, audio, sr, session_dir, nr, vp, methods):
        super().__init__()
        self.audio = audio
        self.sr = sr
        self.session_dir = session_dir
        self.nr = nr
        self.vp = vp
        self.methods = methods

    def run(self):
        try:
            from src.evaluation.workflow import run_methods
            res = run_methods(self.audio, self.sr, self.session_dir, self.nr, self.vp, self.methods)
            self.finished.emit(res, "")
        except Exception as e:
            self.finished.emit({}, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Enhancement and Noise Removal")
        self.resize(1200, 900)

        self.cfg = AppConfig.load(default_config_path())
        self.ml = MLEnhancer()
        self.pipeline = EnhancementPipeline(
            noise_reduction=self.cfg.noise_reduction,
            voice_preservation=self.cfg.voice_preservation,
            sample_rate=self.cfg.sample_rate,
            auto_gain=self.cfg.auto_gain,
            target_rms=self.cfg.target_rms,
        )
        self.engine = AudioEngine(self.cfg, self.pipeline, self.ml)

        self.session_dir = None
        self.original_audio = None
        self.results = {}
        self.record_stop_event = None
        self.record_worker = None
        self.enhance_worker = None

        self._build_ui()
        self._load_device_lists()
        self._apply_config_to_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_status)
        self.timer.start(100)

        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._update_recording_progress)
        self.rec_start_time = None
        self.rec_duration = 30.0

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        # Live tab (existing)
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)

        dev_group = QGroupBox("Audio Devices and Virtual Microphone")
        dev_layout = QGridLayout(dev_group)
        dev_layout.addWidget(QLabel("Physical Microphone Input:"), 0, 0)
        self.input_combo = QComboBox()
        dev_layout.addWidget(self.input_combo, 0, 1)
        dev_layout.addWidget(QLabel("Virtual Microphone Destination:"), 1, 0)
        self.virtual_output_combo = QComboBox()
        dev_layout.addWidget(self.virtual_output_combo, 1, 1)
        self.monitor_checkbox = QCheckBox("Enable local monitoring (headphones recommended)")
        dev_layout.addWidget(self.monitor_checkbox, 2, 0, 1, 2)
        dev_layout.addWidget(QLabel("Monitoring Output:"), 3, 0)
        self.monitor_combo = QComboBox()
        dev_layout.addWidget(self.monitor_combo, 3, 1)
        self.output_combo = self.virtual_output_combo
        virtual_note = QLabel("Virtual destination is a playback/render endpoint. Meeting apps use the matching capture endpoint (for example, a Voicemeeter Out device), not this endpoint.")
        virtual_note.setWordWrap(True)
        dev_layout.addWidget(virtual_note, 4, 0, 1, 2)
        live_layout.addWidget(dev_group)

        mode_group = QGroupBox("Processing")
        mode_layout = QGridLayout(mode_group)
        mode_layout.addWidget(QLabel("Mode:"), 0, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["DSP only", "ML enhanced", "Hybrid (ML + DSP)"])
        mode_layout.addWidget(self.mode_combo, 0, 1)
        mode_layout.addWidget(QLabel("Noise Reduction:"), 1, 0)
        self.nr_slider = QSlider(Qt.Horizontal)
        self.nr_slider.setRange(0, 100)
        mode_layout.addWidget(self.nr_slider, 1, 1)
        self.nr_label = QLabel()
        mode_layout.addWidget(self.nr_label, 1, 2)
        mode_layout.addWidget(QLabel("Voice Preservation:"), 2, 0)
        self.vp_slider = QSlider(Qt.Horizontal)
        self.vp_slider.setRange(0, 100)
        mode_layout.addWidget(self.vp_slider, 2, 1)
        self.vp_label = QLabel()
        mode_layout.addWidget(self.vp_label, 2, 2)
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Processing")
        self.stop_btn = QPushButton("Stop Processing")
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        mode_layout.addLayout(btn_layout, 3, 0, 1, 3)
        live_layout.addWidget(mode_group)

        meter_group = QGroupBox("Levels")
        meter_layout = QGridLayout(meter_group)
        meter_layout.addWidget(QLabel("Input:"), 0, 0)
        self.in_meter = QProgressBar()
        self.in_meter.setRange(0, 100)
        meter_layout.addWidget(self.in_meter, 0, 1)
        meter_layout.addWidget(QLabel("Output:"), 1, 0)
        self.out_meter = QProgressBar()
        self.out_meter.setRange(0, 100)
        meter_layout.addWidget(self.out_meter, 1, 1)
        live_layout.addWidget(meter_group)

        stats_group = QGroupBox("Status")
        stats_layout = QGridLayout(stats_group)
        self.status_label = QLabel("Idle")
        stats_layout.addWidget(QLabel("Status:"), 0, 0)
        stats_layout.addWidget(self.status_label, 0, 1)
        self.mode_status = QLabel("DSP only")
        stats_layout.addWidget(QLabel("Processing mode:"), 1, 0)
        stats_layout.addWidget(self.mode_status, 1, 1)
        self.ml_status = QLabel(self.ml.detail)
        stats_layout.addWidget(QLabel("ML model:"), 2, 0)
        stats_layout.addWidget(self.ml_status, 2, 1)
        self.cpu_label = QLabel("CPU: 0%")
        stats_layout.addWidget(QLabel("CPU:"), 3, 0)
        stats_layout.addWidget(self.cpu_label, 3, 1)
        self.latency_label = QLabel("Latency: 0 ms")
        stats_layout.addWidget(QLabel("Latency:"), 4, 0)
        stats_layout.addWidget(self.latency_label, 4, 1)
        self.uf_label = QLabel("Underflows: 0")
        stats_layout.addWidget(QLabel("Underflows:"), 5, 0)
        stats_layout.addWidget(self.uf_label, 5, 1)
        self.of_label = QLabel("Overflows: 0")
        stats_layout.addWidget(QLabel("Overflows:"), 6, 0)
        stats_layout.addWidget(self.of_label, 6, 1)
        self.feedback_label = QLabel("Use headphones to avoid microphone feedback.")
        stats_layout.addWidget(QLabel("Feedback:"), 7, 0)
        stats_layout.addWidget(self.feedback_label, 7, 1)
        live_layout.addWidget(stats_group)

        meet_group = QGroupBox("Meeting Integration")
        meet_layout = QHBoxLayout(meet_group)
        self.meet_google_btn = QPushButton("Google Meet")
        self.meet_teams_btn = QPushButton("Microsoft Teams")
        self.meet_zoom_btn = QPushButton("Zoom")
        for b in (self.meet_google_btn, self.meet_teams_btn, self.meet_zoom_btn):
            meet_layout.addWidget(b)
        live_layout.addWidget(meet_group)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        live_layout.addWidget(self.log)

        self.tabs.addTab(live_tab, "Live Processing")

        # Recording tab
        rec_tab = QWidget()
        rec_layout = QVBoxLayout(rec_tab)

        rec_group = QGroupBox("30-Second Recording")
        recg = QGridLayout(rec_group)
        recg.addWidget(QLabel("Input device:"), 0, 0)
        self.rec_device_combo = QComboBox()
        recg.addWidget(self.rec_device_combo, 0, 1)
        recg.addWidget(QLabel("Sample rate:"), 1, 0)
        self.rec_sr_combo = QComboBox()
        self.rec_sr_combo.addItems(["16000", "44100", "48000"])
        recg.addWidget(self.rec_sr_combo, 1, 1)
        recg.addWidget(QLabel("Channels:"), 2, 0)
        self.rec_ch_combo = QComboBox()
        self.rec_ch_combo.addItems(["1", "2"])
        recg.addWidget(self.rec_ch_combo, 2, 1)
        recg.addWidget(QLabel("Duration (s):"), 3, 0)
        self.rec_dur_label = QLabel("30")
        recg.addWidget(self.rec_dur_label, 3, 1)
        self.rec_progress = QProgressBar()
        self.rec_progress.setRange(0, 100)
        recg.addWidget(self.rec_progress, 4, 0, 1, 2)
        self.rec_status = QLabel("Ready")
        recg.addWidget(self.rec_status, 5, 0, 1, 2)
        rbtn = QHBoxLayout()
        self.record_btn = QPushButton("Record 30 Seconds")
        self.stop_rec_btn = QPushButton("Stop Recording")
        self.cancel_rec_btn = QPushButton("Cancel Recording")
        self.stop_rec_btn.setEnabled(False)
        self.cancel_rec_btn.setEnabled(False)
        rbtn.addWidget(self.record_btn)
        rbtn.addWidget(self.stop_rec_btn)
        rbtn.addWidget(self.cancel_rec_btn)
        recg.addLayout(rbtn, 6, 0, 1, 2)
        rec_layout.addWidget(rec_group)

        play_group = QGroupBox("Playback")
        playg = QGridLayout(play_group)
        self.play_original_btn = QPushButton("Play Original")
        self.play_dsp_btn = QPushButton("Play DSP")
        self.play_ml_btn = QPushButton("Play ML")
        self.play_hybrid_btn = QPushButton("Play Hybrid")
        self.stop_play_btn = QPushButton("Stop Playback")
        for i, b in enumerate([self.play_original_btn, self.play_dsp_btn, self.play_ml_btn, self.play_hybrid_btn]):
            playg.addWidget(b, i // 2, i % 2)
        playg.addWidget(self.stop_play_btn, 2, 0, 1, 2)
        rec_layout.addWidget(play_group)

        enh_group = QGroupBox("Enhancement & Analysis")
        enhg = QGridLayout(enh_group)
        self.enhance_dsp_btn = QPushButton("Enhance DSP")
        self.enhance_ml_btn = QPushButton("Enhance ML")
        self.enhance_hybrid_btn = QPushButton("Enhance Hybrid")
        self.compare_all_btn = QPushButton("Compare All Methods")
        self.open_folder_btn = QPushButton("Open Output Folder")
        for i, b in enumerate([self.enhance_dsp_btn, self.enhance_ml_btn, self.enhance_hybrid_btn, self.compare_all_btn, self.open_folder_btn]):
            enhg.addWidget(b, i, 0)
        rec_layout.addWidget(enh_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(6)
        self.metrics_table.setHorizontalHeaderLabels(["Method", "Duration (s)", "RMS", "Peak", "Proc Time (s)", "RTF"])
        rec_layout.addWidget(self.metrics_table)

        if MATPLOTLIB_AVAILABLE:
            self.results_figure = Figure(figsize=(6, 3), tight_layout=True)
            self.results_canvas = FigureCanvas(self.results_figure)
            rec_layout.addWidget(self.results_canvas)
        else:
            self.results_figure = None
            self.results_canvas = None

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        rec_layout.addWidget(self.results_text)

        self.tabs.addTab(rec_tab, "Recording & Analysis")

        # signals
        self.start_btn.clicked.connect(self.start_processing)
        self.stop_btn.clicked.connect(self.stop_processing)
        self.nr_slider.valueChanged.connect(self._nr_changed)
        self.vp_slider.valueChanged.connect(self._vp_changed)
        self.input_combo.currentIndexChanged.connect(self._device_changed)
        self.virtual_output_combo.currentIndexChanged.connect(self._device_changed)
        self.monitor_combo.currentIndexChanged.connect(self._device_changed)
        self.monitor_checkbox.toggled.connect(self._device_changed)
        self.meet_google_btn.clicked.connect(lambda: webbrowser.open("https://meet.google.com"))
        self.meet_teams_btn.clicked.connect(lambda: webbrowser.open("https://teams.microsoft.com"))
        self.meet_zoom_btn.clicked.connect(lambda: webbrowser.open("https://zoom.us"))

        self.record_btn.clicked.connect(self.start_recording)
        self.stop_rec_btn.clicked.connect(self.stop_recording)
        self.cancel_rec_btn.clicked.connect(self.cancel_recording)
        self.play_original_btn.clicked.connect(lambda: self.play_file("original.wav"))
        self.play_dsp_btn.clicked.connect(lambda: self.play_file("enhanced_dsp.wav"))
        self.play_ml_btn.clicked.connect(lambda: self.play_file("enhanced_ml.wav"))
        self.play_hybrid_btn.clicked.connect(lambda: self.play_file("enhanced_hybrid.wav"))
        self.stop_play_btn.clicked.connect(self.stop_playback)
        self.enhance_dsp_btn.clicked.connect(lambda: self.enhance_single("dsp"))
        self.enhance_ml_btn.clicked.connect(lambda: self.enhance_single("ml"))
        self.enhance_hybrid_btn.clicked.connect(lambda: self.enhance_single("hybrid"))
        self.compare_all_btn.clicked.connect(self.compare_all)
        self.open_folder_btn.clicked.connect(self.open_output_folder)

    def _load_device_lists(self):
        self.input_combo.clear()
        for d in physical_input_devices():
            self.input_combo.addItem(f"[{d.index}] {d.name} ({d.hostapi})", d.index)
        if not self.input_combo.count():
            for d in input_devices():
                self.input_combo.addItem(f"[{d.index}] {d.name} ({d.hostapi})", d.index)
        self.virtual_output_combo.clear()
        for d in virtual_output_devices():
            self.virtual_output_combo.addItem(f"[{d.index}] {d.name} ({d.hostapi})", d.index)
        if not self.virtual_output_combo.count():
            for d in output_devices():
                self.virtual_output_combo.addItem(f"[{d.index}] {d.name} ({d.hostapi})", d.index)
        self.monitor_combo.clear()
        for d in physical_output_devices():
            self.monitor_combo.addItem(f"[{d.index}] {d.name} ({d.hostapi})", d.index)
        if self.cfg.input_device is not None:
            idx = self.input_combo.findData(self.cfg.input_device)
            if idx >= 0:
                self.input_combo.setCurrentIndex(idx)
        if self.cfg.output_device is not None:
            idx = self.virtual_output_combo.findData(self.cfg.output_device)
            if idx >= 0:
                self.virtual_output_combo.setCurrentIndex(idx)
        if getattr(self.cfg, "monitor_device", None) is not None:
            idx = self.monitor_combo.findData(self.cfg.monitor_device)
            if idx >= 0:
                self.monitor_combo.setCurrentIndex(idx)
        # recording device list
        self.rec_device_combo.clear()
        for d in input_devices():
            self.rec_device_combo.addItem(f"[{d.index}] {d.name}", d.index)
        if self.cfg.input_device is not None:
            idx = self.rec_device_combo.findData(self.cfg.input_device)
            if idx >= 0:
                self.rec_device_combo.setCurrentIndex(idx)

    def _apply_config_to_ui(self):
        self.nr_slider.setValue(int(self.cfg.noise_reduction * 100))
        self.vp_slider.setValue(int(self.cfg.voice_preservation * 100))
        self._nr_changed(self.nr_slider.value())
        self._vp_changed(self.vp_slider.value())
        if getattr(self.cfg, "monitor_enabled", False):
            self.monitor_checkbox.setChecked(True)
        else:
            self.monitor_checkbox.setChecked(False)
        # Restore mode including hybrid.
        if getattr(self.cfg, "use_hybrid", False):
            self.mode_combo.setCurrentIndex(2)
        else:
            self.mode_combo.setCurrentIndex(1 if (self.cfg.use_ml and self.ml.available) else 0)

    def _nr_changed(self, v):
        self.nr_label.setText(f"{v}%")
        self.cfg.noise_reduction = v / 100.0
        self.pipeline.update_params(noise_reduction=self.cfg.noise_reduction)

    def _vp_changed(self, v):
        self.vp_label.setText(f"{v}%")
        self.cfg.voice_preservation = v / 100.0
        self.pipeline.update_params(voice_preservation=self.cfg.voice_preservation)

    def _device_changed(self, *_):
        self.cfg.input_device = self.input_combo.currentData()
        self.cfg.output_device = self.virtual_output_combo.currentData()
        self.cfg.monitor_device = self.monitor_combo.currentData()
        self.cfg.monitor_enabled = self.monitor_checkbox.isChecked()

    def start_processing(self):
        self._device_changed()
        if self.engine.is_running():
            self.log.append("Processing is already running.")
            return
        if self.cfg.input_device is None or self.cfg.output_device is None:
            QMessageBox.warning(self, "Devices", "Select input and output devices.")
            return
        mode_idx = self.mode_combo.currentIndex()
        if mode_idx in (1, 2) and not self.ml.available:
            self.log.append("ML model is not available; DSP-only mode selected.")
            self.mode_combo.setCurrentIndex(0)
            mode_idx = 0
        self.cfg.use_ml = (mode_idx == 1)
        self.cfg.use_hybrid = (mode_idx == 2)
        self.cfg.monitor_enabled = self.monitor_checkbox.isChecked()
        if self.virtual_output_combo.count() == 0:
            QMessageBox.warning(self, "Virtual Mic", "No virtual audio destination detected. Install/configure Voicemeeter or another virtual cable first.")
            return
        if self.cfg.output_device is None:
            QMessageBox.warning(self, "Virtual Mic", "Select a virtual audio destination.")
            return
        self.engine.cfg = self.cfg
        self.pipeline.update_params(noise_reduction=self.cfg.noise_reduction, voice_preservation=self.cfg.voice_preservation)
        self.engine.start()
        st = self.engine.get_status()
        if st.error:
            self.log.append("ERROR: " + str(st.error))
            QMessageBox.critical(self, "Audio Error", str(st.error))
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running")
        self._save_config()
        self.feedback_label.setText("Use headphones; speaker output can cause feedback.")
        self.log.append("Routed input %s -> output %s" % (self.cfg.input_device, self.cfg.output_device))
        self.log.append("Diagnostic: actual input=%s output=%s sample_rate=%s channels=%s"
                        % (st.input_device_actual, st.output_device_actual, st.sample_rate_actual, st.channels_actual))
        self.log.append("Use headphones. Do not place speakers near the microphone.")

    def stop_processing(self):
        self.engine.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Stopped")

    def _update_status(self):
        st = self.engine.get_status()
        self.in_meter.setValue(int(min(100, st.input_rms * 1000 * 2)))
        self.out_meter.setValue(int(min(100, st.output_rms * 1000 * 2)))
        self.cpu_label.setText(f"CPU: {st.cpu_percent:.1f}%")
        self.latency_label.setText(f"Latency: {st.process_ms:.1f} ms")
        self.uf_label.setText(f"Underflows: {st.underflows}")
        self.of_label.setText(f"Overflows: {st.overflows}")
        if self.cfg.use_ml:
            mode_text = "ML enhanced (active)" if self.ml.available else "ML unavailable (DSP not used)"
        else:
            mode_text = "DSP only"
        if st.error:
            mode_text = "Error: " + str(st.error)
        self.mode_status.setText(mode_text)
        self.ml_status.setText(self.ml.detail)
        if st.error:
            self.status_label.setText("Error: " + st.error)
            self.log.append("ERROR: " + st.error)

    def _save_config(self):
        try:
            self.cfg.save(default_config_path())
        except Exception as e:
            self.log.append("Config save failed: " + str(e))

    # Recording
    def start_recording(self):
        try:
            dur = float(self.rec_dur_label.text())
        except Exception:
            dur = 30.0
        self.rec_duration = dur
        self.rec_progress.setValue(0)
        self.rec_status.setText("Recording...")
        self.record_stop_event = threading.Event()
        self.record_btn.setEnabled(False)
        self.stop_rec_btn.setEnabled(True)
        self.cancel_rec_btn.setEnabled(True)
        sr = int(self.rec_sr_combo.currentText())
        ch = int(self.rec_ch_combo.currentText())
        dev = self.rec_device_combo.currentData()
        self.rec_start_time = time.time()
        self.rec_timer.start(200)
        self.record_worker = RecordWorker(dur, sr, ch, dev, self.record_stop_event)
        self.record_worker.finished.connect(self.recording_finished)
        self.record_worker.start()

    def _update_recording_progress(self):
        if self.rec_start_time is None:
            return
        elapsed = time.time() - self.rec_start_time
        frac = min(1.0, elapsed / self.rec_duration)
        self.rec_progress.setValue(int(frac * 100))
        self.rec_status.setText(f"Recording... {elapsed:.1f}/{self.rec_duration:.0f}s")
        if elapsed >= self.rec_duration:
            self.stop_recording()

    def stop_recording(self):
        if self.record_stop_event is not None:
            self.record_stop_event.set()
        self.rec_status.setText("Finalizing...")

    def cancel_recording(self):
        if self.record_worker is not None:
            self.record_worker.cancel()
        self.rec_status.setText("Cancelling...")
        self.rec_timer.stop()

    def recording_finished(self, audio, elapsed, cancelled, err):
        self.rec_timer.stop()
        self.record_btn.setEnabled(True)
        self.stop_rec_btn.setEnabled(False)
        self.cancel_rec_btn.setEnabled(False)
        if err:
            self.rec_status.setText("Error: " + err)
            QMessageBox.critical(self, "Recording Error", err)
            return
        if cancelled:
            self.rec_status.setText("Cancelled")
            return
        sr = int(self.rec_sr_combo.currentText())
        ch = int(self.rec_ch_combo.currentText())
        self.session_dir = unique_session_dir("outputs")
        self.original_audio = np.asarray(audio)
        save_wav(self.session_dir / "original.wav", self.original_audio, sr, ch)
        # metadata
        import json
        meta = {
            "duration_seconds": float(len(self.original_audio) / sr if self.original_audio.ndim == 1 else len(self.original_audio) / sr),
            "sample_rate": sr,
            "channels": ch,
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": self.rec_device_combo.currentText(),
            "cancelled": False,
        }
        (self.session_dir / "session_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.rec_status.setText(f"Saved: {self.session_dir}")
        self.results_text.append(f"Recording saved to {self.session_dir}")
        self.results = {"original": self.session_dir / "original.wav"}

    def play_file(self, name):
        if self.session_dir is None:
            QMessageBox.warning(self, "No session", "Record audio first.")
            return
        path = self.session_dir / name
        if not path.exists():
            QMessageBox.warning(self, "Missing file", f"{name} not found.")
            return
        try:
            data, sr = sf.read(str(path))
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            sd.stop()
            sd.play(data, sr)
        except Exception as e:
            QMessageBox.critical(self, "Playback Error", str(e))

    def stop_playback(self):
        try:
            sd.stop()
        except Exception:
            pass

    def enhance_single(self, method):
        if self.original_audio is None:
            QMessageBox.warning(self, "No recording", "Record 30-second audio first.")
            return
        sr = int(self.rec_sr_combo.currentText())
        ch = int(self.rec_ch_combo.currentText())
        # Ensure 2D handling okay; models operate on mono. Convert.
        audio = self.original_audio
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self._run_enhancement(audio, sr, [method])

    def compare_all(self):
        if self.original_audio is None:
            QMessageBox.warning(self, "No recording", "Record 30-second audio first.")
            return
        sr = int(self.rec_sr_combo.currentText())
        audio = self.original_audio
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self._run_enhancement(audio, sr, ["dsp", "ml", "hybrid"])

    def _run_enhancement(self, audio, sr, methods):
        if self.enhance_worker is not None and self.enhance_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Enhancement already running.")
            return
        self.results_text.append(f"Running enhancement: {methods}")
        self.enhance_worker = EnhanceWorker(audio, sr, self.session_dir, self.cfg.noise_reduction, self.cfg.voice_preservation, methods)
        self.enhance_worker.finished.connect(self.enhancement_finished)
        self.enhance_worker.start()

    def enhancement_finished(self, res, err):
        if err:
            self.results_text.append("ERROR: " + err)
            QMessageBox.critical(self, "Enhancement Error", err)
            return
        self.results = res
        # populate metrics table
        self.metrics_table.setRowCount(0)
        for method, info in res.items():
            if "path" not in info:
                continue
            row = self.metrics_table.rowCount()
            self.metrics_table.insertRow(row)
            self.metrics_table.setItem(row, 0, QTableWidgetItem(method))
            self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{info.get('duration', 0):.3f}"))
            self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{info.get('rms', 0):.5f}"))
            self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{info.get('peak', 0):.5f}"))
            self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{info.get('process_time', 0):.4f}"))
            self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{info.get('rtf', 0):.4f}"))
        self.results_text.append("Enhancement complete. Files saved in " + str(self.session_dir))
        self._plot_results(res)
        # try generate report
        try:
            from src.evaluation.report import generate_pdf_report
            generate_pdf_report(self.session_dir, res)
            self.results_text.append("PDF report generated.")
        except Exception as e:
            self.results_text.append("PDF generation failed: " + str(e))

    def _plot_results(self, res):
        if not (MATPLOTLIB_AVAILABLE and self.results_canvas is not None):
            return
        try:
            fig = self.results_figure
            fig.clear()
            ax = fig.add_subplot(111)
            methods = []
            rms_values = []
            for method, info in res.items():
                if "error" in info:
                    continue
                methods.append(method)
                rms_values.append(float(info.get("rms", 0.0)))
            if methods:
                ax.bar(methods, rms_values, color="#4a90d9")
                ax.set_title("Enhanced output RMS by method")
                ax.set_ylabel("RMS")
                ax.tick_params(axis="x", rotation=20)
            else:
                ax.text(0.5, 0.5, "No successful enhancement results", ha="center", va="center")
                ax.set_axis_off()
            self.results_canvas.draw_idle()
        except Exception as e:
            self.results_text.append("Plot update failed: " + str(e))

    def open_output_folder(self):
        if self.session_dir is None:
            QMessageBox.warning(self, "No session", "Record audio first.")
            return
        os.startfile(str(self.session_dir))

    def closeEvent(self, event):
        self.engine.stop()
        if self.record_worker is not None and self.record_worker.isRunning():
            self.record_worker.cancel()
            self.record_worker.wait(2000)
        if self.enhance_worker is not None and self.enhance_worker.isRunning():
            self.enhance_worker.wait(5000)
        self._save_config()
        super().closeEvent(event)


def run_app():
    app = QApplication(sys.argv)
    app.setApplicationName("Speech Enhancement and Noise Removal")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
