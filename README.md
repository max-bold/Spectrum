# Spectrum Analyzer

A real-time audio spectrum analyzer built with Python and Dear PyGui, designed for audio analysis and signal processing applications.

![Screenshot](info/Screenshot%202025-11-21%20201724.png)


## Features

- **Real-time FFT Analysis**: Live frequency domain visualization with logarithmic frequency scale
- **Multiple Recording Channels**: Support for up to 5 simultaneous recordings
- **Time Domain Visualization**: Real-time level meters and waveform display
- **Audio I/O Management**: Comprehensive audio input/output device management
- **Signal Generation**: Built-in log sweep and pink noise generators for testing and calibration
- **Professional Audio Support**: Compatible with various audio interfaces and devices
- **Cross-platform**: Works on Windows with ASIO support, Linux, and macOS (need testing)

## Requirements

- Python 3.7+
- Audio interface (built-in or external)
- Windows OS (with optional ASIO driver support)

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Release builds

Tagged pushes build release artifacts automatically with GitHub Actions:

```bash
git tag v0.2.3
git push origin v0.2.3
```

The release workflow publishes separate Windows, macOS Intel, and macOS Apple
Silicon zip files.

### macOS Gatekeeper

Current macOS release builds are packaged as `.app` bundles, but they are not
yet notarized by Apple. On first launch, Finder may show: "Apple cannot check it
for malicious software." This is expected until the project has Apple Developer
credentials for Developer ID signing and notarization.

To open a downloaded macOS build, right-click the app and choose **Open**, then
confirm the prompt. Advanced users can also remove the quarantine attribute:

```bash
xattr -dr com.apple.quarantine /path/to/BM_Spectrum_v0.2.3_macos-x86_64.app
```

## Usage

To start the spectrum analyzer, run:

```bash
py run.py
```

### Main Interface

The application features a dual-pane interface:
- **Left Panel**: Real-time FFT plot showing frequency response (20 Hz - 20 kHz)
- **Right Panel**: Control panel with audio settings, recording options, and level meters

### Key Controls

- **Run/Stop**: Start or stop audio analysis
- **Record Management**: Enable/disable multiple recording tracks
- **Audio Settings**: Configure input/output devices and sample rates
- **Generator Controls**: Access built-in signal generators for testing

## Project Structure

- `run.py` - Application entrypoint
- `spectrum_app/gui.py` - Dear PyGui interface construction
- `spectrum_app/models.py` - Application enums and shared model types
- `spectrum_app/themes.py` - Dear PyGui themes
- `spectrum_app/cbs.py` - Compatibility facade for callbacks and app state
- `spectrum_app/` - Application state, callbacks, file I/O, analysis orchestration, and UI sync logic
- `utils/` - Utility modules for audio processing and DSP
- `sandbox/` - Development and testing scripts
- `info/` - Documentation and reference materials

## Technical Details

The analyzer uses:
- **Dear PyGui** for the user interface
- **SoundDevice** for audio I/O operations
- **NumPy/SciPy** for signal processing and FFT calculations
- **Matplotlib** for additional plotting capabilities

## Contributing

This is an active development project. Feel free to explore the `sandbox/` directory for experimental features and development scripts.

## License

This project is open source. Please refer to the repository for license details.
