# Portable Installation Guide

## Package
`SpeechEnhancementNoiseRemoval_Portable.zip`

## Extract
1. Copy the ZIP file to the target Windows computer.
2. Right-click the ZIP file and choose **Extract All**.
3. Extract the entire `SpeechEnhancementNoiseRemoval` folder.
4. Keep every extracted file together. Do not move only the `.exe` file.

## Run
Open:
`SpeechEnhancementNoiseRemoval\SpeechEnhancementNoiseRemoval.exe`

The executable requires the `_internal` folder and the bundled model files.

## External requirements
- Windows x64 compatible with Python 3.13 packaged runtime.
- Optional virtual audio driver such as Voicemeeter/VB-Audio for virtual microphone mode.
- Google Meet, Microsoft Teams, or Zoom for meeting use.
- No source-code workspace or Python installation is required.

## Virtual microphone setup
1. Install and configure Voicemeeter or another suitable virtual audio device.
2. Start Speech Enhancement and Noise Removal.
3. Select the physical microphone.
4. Select a virtual render endpoint, such as a Voicemeeter In device.
5. Select a processing method.
6. Click **Start Processing**.
7. In the meeting application, select the matching capture endpoint and real headphones/speakers.
8. Disable monitoring unless using headphones.

## Troubleshooting
- If Windows blocks the app, check SmartScreen and allow the unsigned application.
- If the GUI does not open, verify the entire folder was extracted.
- If the model cannot be found, verify the bundled `models` folder exists inside `_internal`.
- If ML is unavailable, use DSP mode.
- For feedback, use headphones and avoid routing monitoring into the microphone.

## Limitations
- No virtual audio driver is bundled.
- Virtual meeting-app pairing has not been verified on another computer.
- Acoustic feedback behavior depends on physical speaker/microphone placement.
PY