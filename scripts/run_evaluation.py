"""Offline audio quality evaluation and plotting."""

from __future__ import annotations

import json
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
from src.evaluation.plots import generate_snr_graphs
from src.ml.enhancer import MLEnhancer

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _valid(v):
    return v is not None and np.isfinite(v)


def main():
    sr = 16000
    clean = synthetic_speech(sr=sr, duration=3.0, seed=0)
    cases = [("white", 10.0), ("babble", 8.0), ("changing", 12.0)]
    ml = MLEnhancer()
    rows = []
    records = []
    for i, (nt, target_snr) in enumerate(cases):
        noisy, noise = add_noise(clean, noise_type=nt, snr_db=target_snr, seed=i + 1)
        pipe = EnhancementPipeline(noise_reduction=0.6, voice_preservation=0.7, sample_rate=sr, auto_gain=False)
        enhanced_dsp = pipe.process_audio(noisy.copy())
        enhanced_ml = ml.enhance_audio(noisy.copy(), sr) if ml.available else None
        enhanced_hybrid = None
        if ml.available:
            hybrid_pipe = EnhancementPipeline(noise_reduction=0.3, voice_preservation=0.7, sample_rate=sr, auto_gain=False)
            hybrid_ml = ml.enhance_audio(noisy.copy(), sr)
            enhanced_hybrid = hybrid_pipe.process_audio(hybrid_ml.copy())

        sin = snr_db(clean, noisy)
        sdsp = snr_db(clean, enhanced_dsp)
        sml = snr_db(clean, enhanced_ml) if enhanced_ml is not None else None
        shy = snr_db(clean, enhanced_hybrid) if enhanced_hybrid is not None else None
        dsp_imp = snr_improvement(clean, noisy, enhanced_dsp)
        ml_imp = snr_improvement(clean, noisy, enhanced_ml) if enhanced_ml is not None else None
        hybrid_imp = snr_improvement(clean, noisy, enhanced_hybrid) if enhanced_hybrid is not None else None

        rows.append((nt, target_snr, sin, sdsp, sml, shy))
        print(f"{nt}: noisy={sin} DSP={sdsp} ML={sml} Hybrid={shy}")

        # Existing diagnostic figures.
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
        if enhanced_hybrid is not None:
            plt.plot(enhanced_hybrid[:1000], label="Hybrid", alpha=0.8)
        plt.legend()
        plt.title(f"Waveforms: {nt} noise")
        plt.tight_layout()
        plt.savefig(RESULTS / f"waveform_{nt}.png", dpi=100)
        plt.close()

        method_outputs = {"DSP": enhanced_dsp}
        if enhanced_ml is not None:
            method_outputs["ML"] = enhanced_ml
        if enhanced_hybrid is not None:
            method_outputs["Hybrid"] = enhanced_hybrid
        generate_snr_graphs(RESULTS, clean, noisy, method_outputs, sr,
                            title_suffix=f" ({nt} noise, {target_snr:.1f} dB nominal)",
                            filename_prefix=f"{nt}_")
        records.append({
            "noise_type": nt,
            "target_snr_db": float(target_snr),
            "noisy_snr_db": None if sin is None else float(sin),
            "dsp_snr_db": None if sdsp is None else float(sdsp),
            "ml_snr_db": None if sml is None else (float(sml) if np.isfinite(sml) else None),
            "hybrid_snr_db": None if shy is None else (float(shy) if np.isfinite(shy) else None),
            "dsp_improvement_db": None if dsp_imp is None else float(dsp_imp),
            "ml_improvement_db": None if ml_imp is None else float(ml_imp),
            "hybrid_improvement_db": None if hybrid_imp is None else float(hybrid_imp),
        })

    # Overview across conditions.
    fig = plt.figure(figsize=(9, 4))
    labels = [r[0] for r in rows]
    before = [r[2] for r in rows]
    dspv = [r[3] for r in rows]
    mlv = [r[4] for r in rows]
    hyb = [r[5] for r in rows]
    x = np.arange(len(labels))
    width = 0.18
    plt.bar(x - width, before, width, label="Noisy SNR")
    plt.bar(x, dspv, width, label="DSP SNR")
    if ml.available:
        plt.bar(x + width, mlv, width, label="ML SNR")
    if any(h is not None for h in hyb):
        plt.bar(x + 2 * width, [h if _valid(h) else float("nan") for h in hyb], width, label="Hybrid SNR")
    plt.xticks(x, labels)
    plt.ylabel("SNR (dB)")
    plt.title("SNR Comparison Across Noise Conditions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "snr_comparison.png", dpi=100)
    plt.close()

    metrics_path = RESULTS / "snr_results.json"
    metrics_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    with (RESULTS / "snr_results.csv").open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Noise Type", "Target SNR (dB)", "Noisy SNR (dB)", "DSP SNR (dB)", "ML SNR (dB)",
                        "Hybrid SNR (dB)", "DSP Improvement (dB)", "ML Improvement (dB)", "Hybrid Improvement (dB)"])
        for row in records:
            writer.writerow([
                row["noise_type"], row["target_snr_db"], row["noisy_snr_db"], row["dsp_snr_db"],
                row["ml_snr_db"], row["hybrid_snr_db"], row["dsp_improvement_db"],
                row["ml_improvement_db"], row["hybrid_improvement_db"],
            ])
    text = "noise_type target_snr noisy_snr dsp_snr ml_snr hybrid_snr dsp_improvement ml_improvement hybrid_improvement\n"
    for r in records:
        text += (f"{r['noise_type']} {r['target_snr_db']} {r['noisy_snr_db']} "
                  f"{r['dsp_snr_db']} {r['ml_snr_db']} {r['hybrid_snr_db']} "
                  f"{r['dsp_improvement_db']} {r['ml_improvement_db']} {r['hybrid_improvement_db']}\n")
    (RESULTS / "metrics.txt").write_text(text, encoding="utf-8")
    print((RESULTS / "metrics.txt").read_text())
    json.dump(records, open(RESULTS / "snr_results.json", "w", encoding="utf-8"), indent=2)

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