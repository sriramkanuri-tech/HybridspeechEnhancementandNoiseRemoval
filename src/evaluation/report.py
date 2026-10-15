"""PDF report generation with embedded before/after graphs."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

STYLES = getSampleStyleSheet()


def _img(path, width=460):
    if not Path(path).is_file():
        return None
    try:
        return Image(str(path), width=width, height=width * 0.6)
    except Exception:
        return None


def generate_pdf_report(session_dir, results, clean=None, noisy=None, sr=None):
    session_dir = Path(session_dir)
    out = session_dir / "comparison_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, title="Speech Enhancement Report")
    story = []

    story.append(Paragraph("Speech Enhancement and Noise Removal - Session Report", STYLES["Heading1"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Session directory: {session_dir.name}", STYLES["Normal"]))
    story.append(Paragraph("Metrics are calculated from the actual recorded and processed WAV files.", STYLES["Normal"]))
    story.append(Spacer(1, 12))

    # Summary table
    data = [["Method", "Status", "Duration(s)", "RMS", "Peak", "ProcTime(s)", "RTF"]]
    for method, info in results.items():
        if not isinstance(info, dict):
            continue
        if "error" in info:
            data.append([method, "Failed", "N/A", "N/A", "N/A", "N/A", "N/A"])
        else:
            data.append([
                method,
                "Success",
                f"{info.get('duration', 0):.3f}",
                f"{info.get('rms', 0):.6f}",
                f"{info.get('peak', 0):.5f}",
                f"{info.get('process_time', 0):.4f}",
                f"{info.get('rtf', 0):.5f}",
            ])
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph("RMS reduction can indicate noise suppression or speech attenuation. A lower RMS is not automatically better.", STYLES["Italic"]))

    # Graphs with captions
    graph_items = [
        ("waveform_comparison.png", "Waveform Comparison",
         "Top: original recording. Bottom: enhanced recording. Waveforms show amplitude over time and help reveal speech preservation, discontinuities, clipping, and excessive gain."),
        ("rms_comparison.png", "RMS Comparison",
         "Left: overall RMS per method. Right: RMS over time using the same window and hop length. RMS measures effective signal energy; lower RMS may indicate noise reduction or speech attenuation."),
        ("peak_comparison.png", "Peak Amplitude and Clipping",
         "Compares maximum absolute amplitude and estimated clipping percentage using threshold 0.99. A high peak does not necessarily mean better quality; clipping can indicate distortion."),
        ("spectrogram_comparison.png", "Spectrogram Comparison",
         "Shows time-frequency energy. Consistent STFT parameters are used. Look for preserved speech structures such as formants and harmonics while reducing unwanted energy."),
        ("frequency_band_comparison.png", "Frequency-Band Energy",
         "Compares energy in low, mid, and high bands. A reduction in a band is not automatically beneficial because speech may also occupy that band."),
        ("processing_performance.png", "Processing Time and RTF",
         "RTF below 1 means enhancement completed faster than audio duration. This does not by itself prove live streaming suitability."),
    ]
    for filename, title, caption in graph_items:
        path = session_dir / filename
        if not path.is_file():
            continue
        story.append(PageBreak())
        story.append(Paragraph(title, STYLES["Heading2"]))
        img = _img(path)
        if img is not None:
            story.append(img)
        story.append(Spacer(1, 6))
        story.append(Paragraph(caption, STYLES["Normal"]))

    has_snr = any(isinstance(info, dict) and (
        "snr_output" in info or "snr_input" in info
    ) for info in results.values())
    snr_rows = [["Method", "Input SNR (dB)", "Output SNR (dB)", "Improvement (dB)"]]
    for method, info in results.items():
        if not isinstance(info, dict) or "error" in info:
            continue
        if "snr_output" not in info and "snr_input" not in info:
            continue
        snr_rows.append([
            method,
            "Unavailable" if info.get("snr_input") is None else f"{info.get('snr_input'):.3f}",
            "Unavailable" if info.get("snr_output") is None else f"{info.get('snr_output'):.3f}",
            "Unavailable" if info.get("snr_improvement") is None else f"{info.get('snr_improvement'):.3f}",
        ])
    story.append(Paragraph("SNR Comparison", STYLES["Heading2"]))
    if has_snr and len(snr_rows) > 1:
        story.append(Paragraph(
            "SNR improvement is calculated as output SNR minus noisy input SNR using the same clean reference. "
            "Database-style line RMS and peak values are not treated as SNR. Method settings are saved with the session.",
            STYLES["Normal"],
        ))
        snr_table = Table(snr_rows, repeatRows=1)
        snr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ]))
        story.append(snr_table)
        story.append(Spacer(1, 8))
        snr_graphs = [
            ("snr_before_after.png", "SNR Before and After Enhancement",
             "Grouped comparison of noisy-input SNR and enhanced-output SNR for each successful method. Methods without a calculable reference-based SNR are marked unavailable."),
            ("snr_improvement.png", "SNR Improvement by Method",
             "SNR improvement is output SNR minus noisy-input SNR. Positive values indicate an improved signal-to-noise ratio under the measured clean reference."),
        ]
        for filename, title, caption in snr_graphs:
            path = session_dir / filename
            if not path.is_file():
                continue
            story.append(PageBreak())
            story.append(Paragraph(title, STYLES["Heading2"]))
            img = _img(path)
            if img is not None:
                story.append(img)
            story.append(Spacer(6, 4))
            story.append(Paragraph(caption, STYLES["Normal"]))
    else:
        story.append(Paragraph(
            "Reference-based SNR unavailable: no clean reference was supplied for this session. "
            "RMS, peak, waveform, spectrogram, and processing metrics remain available for live recordings.",
            STYLES["Normal"],
        ))
        story.append(Spacer(1, 8))
    story.append(PageBreak())
    story.append(Paragraph("Metric limitations", STYLES["Heading2"]))
    story.append(Paragraph(
        "Reference-based SNR, PESQ, and STOI are not computed here because real microphone recordings do not include a clean speech reference. "
        "RMS and spectrogram evidence alone cannot prove intelligibility improvement. Unavailable methods are not represented as successful.",
        STYLES["Normal"],
    ))
    doc.build(story)
    return out
