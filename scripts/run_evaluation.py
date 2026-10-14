"""Offline audio quality evaluation and plotting."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.settings import AppConfig
from src.dsp.pipeline import EnhancementPipeline
from src.evaluation.datasets import synthetic_speech, add_noise
from src.evaluation.metrics import snr_db, snr_improvement, rms
from src.ml.enhancer import MLEnhancer

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    sr = 16000
    clean = synthetic_speech(sr=sr, duration=3.0, seed=0)
    cases = [("white", 10.0), ("babble", 8.0), ("changing", 12.0)]
    ml = MLEnhancer()
    rows = []
    for i, (nt, snr) in enumerate(cases):
        noisy, noise = add_noise(clean, noise_type=nt, snr_db=snr, seed=i + 1)
        pipe = EnhancementPipeline(noise_reduction=0.6, voice_preservation=0.7, sample_rate=sr, auto_gain=False)
        enhanced_dsp = pipe.process_audio(noisy.copy())
        enhanced_ml = ml.enhance_audio(noisy.copy(), sr) if ml.available else None

        sin = snr_db(clean, noisy)
        sdsp = snr_db(clean, enhanced_dsp)
        sml = snr_db(clean, enhanced_ml) if enhanced_ml is not None else float("nan")
        rows.append((nt, snr, sin, sdsp, sml))
        print(f"{nt}: noisy={sin:.2f} DSP={sdsp:.2f} ML={sml:.2f}")

        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes[0, 0].plot(clean[:2000]); axes[0, 0].set_title("Clean segment")
        axes[0, 1].plot(noisy[:2000]); axes[0, 1].set_title("Noisy segment")
        axes[1, 0].plot(enhanced_dsp[:2000]); axes[1, 0].set_title("DSP enhanced")
        axes[1, 1].specgram(noisy, Fs=sr, NFFT=256, noverlap=128)
        axes[1, 1].set_title("Noisy spectrogram")
        plt.tight_layout()
        plt.savefig(RESULTS / f"spectrogram_{nt}.png", dpi=100)
        plt.close()

        fig2 = plt.figure(figsize=(10, 4))
        plt.plot(clean[:1000], label="clean")
        plt.plot(noisy[:1000], label="noisy", alpha=0.6)
        plt.plot(enhanced_dsp[:1000], label="DSP enhanced", alpha=0.8)
        if enhanced_ml is not None:
            plt.plot(enhanced_ml[:1000], label="ML enhanced", alpha=0.8)
        plt.legend()
        plt.title(f"Waveforms: {nt} noise")
        plt.tight_layout()
        plt.savefig(RESULTS / f"waveform_{nt}.png", dpi=100)
        plt.close()

    fig = plt.figure(figsize=(9, 4))
    labels = [r[0] for r in rows]
    before = [r[2] for r in rows]
    dspv = [r[3] for r in rows]
    x = np.arange(len(labels))
    width = 0.25
    plt.bar(x - width, before, width, label="Noisy SNR")
    plt.bar(x, dspv, width, label="DSP SNR")
    if ml.available:
        mlv = [r[4] for r in rows]
        plt.bar(x + width, mlv, width, label="ML SNR")
    plt.xticks(x, labels)
    plt.ylabel("SNR (dB)")
    plt.title("SNR Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "snr_comparison.png", dpi=100)
    plt.close()

    with open(RESULTS / "metrics.txt", "w", encoding="utf-8") as f:
        f.write("noise_type target_snr noisy_snr dsp_snr ml_snr dsp_improvement ml_improvement\n")
        for r in rows:
            dsp_imp = r[3] - r[2]
            ml_imp = (r[4] - r[2]) if not np.isnan(r[4]) else float("nan")
            f.write(f"{r[0]} {r[1]} {r[2]:.3f} {r[3]:.3f} {r[4]:.3f} {dsp_imp:.3f} {ml_imp:.3f}\n")
    print(open(RESULTS / "metrics.txt").read())

    # Performance
    noisy, _ = add_noise(clean, "white", 10.0, seed=9)
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        pipe.process_audio(noisy.copy())
    dt = (time.perf_counter() - t0) / n_runs
    block_ms = len(noisy) / sr * 1000
    rtf = dt * 1000 / block_ms
    with open(RESULTS / "performance.txt", "w", encoding="utf-8") as f:
        f.write(f"audio_duration_s={len(noisy)/sr:.3f}\nprocess_time_s={dt:.4f}\nrtf={rtf:.4f}\nml_available={ml.available}\nml_detail={ml.detail}\n")
    print(f"DSP RTF={rtf:.4f}")
    if ml.available:
        t0 = time.perf_counter()
        for _ in range(3):
            ml.enhance_audio(noisy.copy(), sr)
        dtml = (time.perf_counter() - t0) / 3
        print(f"ML process_time_s={dtml:.4f} RTF={dtml*1000/block_ms:.4f}")


if __name__ == "__main__":
    main()
