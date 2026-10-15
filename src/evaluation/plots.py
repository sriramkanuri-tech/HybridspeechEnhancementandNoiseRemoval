"""Plot generation for offline enhancement comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import rms, rms_dbfs, peak_amplitude, peak_dbfs, clipping_pct, frequency_band_energy, snr_db, snr_improvement


def _mono(a):
    a = np.asarray(a)
    if a.ndim > 1:
        a = a.mean(axis=-1) if a.shape[-1] <= 8 else a.mean(axis=0)
    return np.asarray(a, dtype=np.float64).ravel()


def _read(path):
    import soundfile as sf
    data, sr = sf.read(str(path))
    return _mono(data), sr


def generate_all_plots(session_dir, original, results, sr, clean=None):
    session_dir = Path(session_dir)
    original = _mono(original)
    sr = float(sr)
    successful = {name: info for name, info in results.items()
                  if isinstance(info, dict) and info.get("path") and Path(info["path"]).exists()}

    # 1. Waveform comparison two-panel
    try:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, sharey=True)
        n = min(len(original), 8000)
        t = np.arange(n) / sr
        axes[0].plot(t, original[:n], color="steelblue", linewidth=0.8)
        axes[0].set_title("Original Recording Waveform")
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)
        # plot first successful enhanced
        if successful:
            name, info = next(iter(successful.items()))
            data, _ = _read(info["path"])
            m = min(len(data), n)
            axes[1].plot(t[:m], data[:m], color="darkorange", linewidth=0.8)
            axes[1].set_title(f"Enhanced Recording Waveform ({name})")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(session_dir / "waveform_comparison.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # 2. RMS comparison + RMS over time
    try:
        names = ["Original"] + list(successful.keys())
        values = [rms(original)] + [info.get("rms", 0.0) for info in successful.values()]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(names, values, color=["gray"] + ["tab:blue"] * len(successful))
        ax1.set_ylabel("RMS")
        ax1.set_title("Overall RMS Comparison")
        ax1.tick_params(axis="x", rotation=30)
        # RMS over time with same window/hop
        win = 1024
        hop = 512
        def rms_series(x):
            out = []
            for i in range(0, len(x) - win + 1, hop):
                out.append(rms(x[i:i + win]))
            return np.asarray(out)
        tx = np.arange(len(rms_series(original))) * hop / sr
        ax2.plot(tx, rms_series(original), label="Original", alpha=0.8)
        for name, info in successful.items():
            data, _ = _read(info["path"])
            ty = np.arange(len(rms_series(data))) * hop / sr
            ax2.plot(ty, rms_series(data), label=name, alpha=0.8)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("RMS")
        ax2.set_title("RMS Over Time")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(session_dir / "rms_comparison.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # 3. Peak / clipping
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        pnames = ["Original"] + list(successful.keys())
        peaks = [peak_amplitude(original)] + [peak_amplitude(_read(info["path"])[0]) for info in successful.values()]
        clips = [clipping_pct(original)] + [clipping_pct(_read(info["path"])[0]) for info in successful.values()]
        ax1.bar(pnames, peaks, color=["gray"] + ["tab:blue"] * len(successful))
        ax1.set_ylabel("Peak amplitude")
        ax1.set_title("Peak Amplitude Comparison")
        ax1.tick_params(axis="x", rotation=30)
        ax2.bar(pnames, clips, color=["gray"] + ["tab:red"] * len(successful))
        ax2.set_ylabel("Clipping % (>=0.99)")
        ax2.set_title("Estimated Clipping Percentage")
        ax2.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(session_dir / "peak_comparison.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # 4. Spectrograms
    try:
        nplots = 1 + len(successful)
        fig, axes = plt.subplots(1, nplots, figsize=(4 * nplots, 4))
        if nplots == 1:
            axes = [axes]
        from matplotlib import mlab
        for ax, (title, data) in zip(axes, [("Original", original)] + [(n, _read(i["path"])[0]) for n, i in successful.items()]):
            S, freqs, bins, im = ax.specgram(data, NFFT=256, Fs=sr, noverlap=128)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
        plt.tight_layout()
        plt.savefig(session_dir / "spectrogram_comparison.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # 5. Frequency-band energy
    try:
        bands = [(0, 250, "Low"), (250, 2000, "Mid"), (2000, sr / 2, "High")]
        x = np.arange(len(bands))
        width = 0.8 / (1 + len(successful))
        fig = plt.figure(figsize=(8, 4))
        vals_orig = [frequency_band_energy(original, sr, lo, hi) for lo, hi, _ in bands]
        plt.bar(x - (len(successful) * width) / 2, vals_orig, width, label="Original")
        for idx, (name, info) in enumerate(successful.items()):
            data, _ = _read(info["path"])
            vals = [frequency_band_energy(data, sr, lo, hi) for lo, hi, _ in bands]
            plt.bar(x - (len(successful) * width) / 2 + (idx + 1) * width, vals, width, label=name)
        plt.xticks(x, [b[2] for b in bands])
        plt.ylabel("Band energy")
        plt.title("Frequency-Band Energy Comparison")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(session_dir / "frequency_band_comparison.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # 6. Processing performance
    try:
        names = list(successful.keys())
        times = [info.get("process_time", 0.0) for info in successful.values()]
        rtfs = [info.get("rtf", 0.0) for info in successful.values()]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(names, times, color="tab:green")
        ax1.set_ylabel("Seconds")
        ax1.set_title("Processing Time")
        ax1.tick_params(axis="x", rotation=30)
        ax2.bar(names, rtfs, color="tab:purple")
        ax2.axhline(1.0, color="red", linestyle="--", label="RTF=1")
        ax2.set_ylabel("RTF")
        ax2.set_title("Real-Time Factor")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(session_dir / "processing_performance.png", dpi=110)
        plt.close()
    except Exception:
        pass

    # Optional reference-based SNR graphs when a clean reference is available.
    try:
        if clean is not None:
            clean_a = _mono(clean)
            noisy_a = _mono(original)
            method_outputs = {}
            successful = {name: info for name, info in results.items()
                          if isinstance(info, dict) and info.get("path") and Path(info["path"]).exists()}
            for name, info in successful.items():
                try:
                    data, _ = _read(info["path"])
                    method_outputs[name] = data
                except Exception:
                    continue
            generate_snr_graphs(session_dir, clean_a, noisy_a, method_outputs, sr)
    except Exception:
        pass

    # compatibility aliases for expected files
    try:
        # ensure rms and peak aliases used earlier remain present
        pass
    except Exception:
        pass


def generate_snr_graphs(session_dir, clean, noisy, method_outputs, sr, title_suffix="", filename_prefix=""):
    """Save before/after and improvement SNR graphs when a clean reference exists."""
    session_dir = Path(session_dir)
    clean = _mono(clean).astype(np.float64)
    noisy = _mono(noisy).astype(np.float64)
    sr = float(sr)
    # Avoid invalid arrays and unequal sample-rate measurement contexts through sr.
    if clean.size < 8 or noisy.size < 8:
        return
    clean = clean[: min(len(clean), len(noisy))]
    noisy = noisy[: min(len(noisy), len(clean))]
    base_snr = snr_db(clean, noisy)
    methods = list(method_outputs.keys())
    before_values = [base_snr for _ in methods]
    after_values = []
    improvements = []
    for method in methods:
        audio = method_outputs.get(method)
        if audio is None:
            after_values.append(None)
            improvements.append(None)
            continue
        try:
            arr = _mono(audio).astype(np.float64)
            arr = arr[: min(len(clean), len(arr))]
            arr = arr[: len(clean)]
            out_snr = snr_db(clean, arr)
        except Exception:
            out_snr = None
        after_values.append(out_snr)
        improvements.append(snr_improvement(clean, noisy, arr) if out_snr is not None else None)

    # Before-and-after grouped bar chart.
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        valid_methods = methods
        x = np.arange(len(valid_methods))
        width = 0.36
        before_bar_positions = x - width / 2.0
        after_bar_positions = x + width / 2.0
        plotted_before = [(before_bar_positions[i], before_values[i]) for i in range(len(methods))
                         if before_values[i] is not None]
        if plotted_before:
            ax.bar([p[0] for p in plotted_before], [p[1] for p in plotted_before], width,
                   color="#8c8c8c", label="Before enhancement (noisy input)")
        for i, value in enumerate(after_values):
            if value is not None:
                ax.bar(after_bar_positions[i], value, width, color="#1f77b4", label="After enhancement" if i == 0 else "")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("SNR (dB)")
        ax.set_title("SNR Before and After Enhancement" + title_suffix)
        ax.grid(True, axis="y", alpha=0.3)
        # De-duplicate legend labels.
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique:
                unique[l] = h
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        # Add labels.
        for i in range(len(methods)):
            if before_values[i] is not None:
                ax.text(before_bar_positions[i], before_values[i], f"{before_values[i]:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=0)
            if after_values[i] is not None:
                ax.text(after_bar_positions[i], after_values[i], f"{after_values[i]:.2f}",
                        ha="center", va="bottom", fontsize=8, rotation=0)
            else:
                ax.text(after_bar_positions[i], 0.0, "Unavailable", ha="center",
                        va="bottom", fontsize=8, color="#b22222", rotation=90)
        plt.tight_layout()
        plt.savefig(session_dir / f"{filename_prefix}snr_before_after.png", dpi=110)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass

    # SNR improvement chart.
    try:
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(methods))
        valid_indices = [i for i, v in enumerate(improvements) if v is not None]
        if valid_indices:
            ax.bar([x[i] for i in valid_indices], [improvements[i] for i in valid_indices],
                   color="#2a9d8f", width=0.6, label="SNR improvement (dB)")
        for i in valid_indices:
            ax.text(x[i], improvements[i], f"{improvements[i]:.2f}", ha="center",
                    va="bottom" if improvements[i] >= 0 else "top", fontsize=8)
        ax.axhline(0.0, color="#333333", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel("SNR improvement (dB)")
        ax.set_title("SNR Improvement by Method" + title_suffix)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(improvements):
            if v is None:
                ax.text(x[i], 0.0, "Unavailable", ha="center", va="bottom",
                        fontsize=8, color="#b22222", rotation=90)
        plt.tight_layout()
        plt.savefig(session_dir / f"{filename_prefix}snr_improvement.png", dpi=110)
        plt.close()
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
