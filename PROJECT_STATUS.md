# Project Status (Final Verification)

- Original repository untouched: `C:\Users\srira\OneDrive\Desktop\Speech-Enhancement-and-Noise-Removal`
- Workspace: `C:\Users\srira\OneDrive\Desktop\SPproject`

## Automated tests
- Command: `python -m pytest -q`
- Result: 13 passed, 0 failed, 0 skipped
- Tests cover DSP, config, engine, ML loading/inference.

## Evaluation
- Command: `python scripts/run_evaluation.py`
- DSP SNR improvement (synthetic speech): +0.54 dB white, +0.63 dB babble, +0.12 dB changing
- ML SNR improvement (synthetic speech): +3.92 dB white, +4.13 dB babble, +2.94 dB changing
- RTF: DSP 0.0044, ML 0.0107

## Microphone hardware test
- 30-second recording at 16000 Hz passed.
- Reloaded WAV shape: (480000,), duration 30.000 seconds.
- Hybrid/DSP/ML 30-second outputs all retained exact duration and shape.

## Executable
- Built at `dist/SpeechEnhancementNoiseRemoval/SpeechEnhancementNoiseRemoval.exe`
- Launch test: windowed process started; no stderr; startup log confirms bundle path and ML loading.

## Known limitations
- QtMultimedia is unavailable in this environment; sounddevice handles recording/playback.
- Real-room microphone quality and subjective intelligibility still require listening tests.
- No clean speech reference for real microphone recordings, so objective SNR cannot be reported for live recordings.