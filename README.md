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

## Usage

To start the spectrum analyzer, run:

```bash
py gui.py
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

- `gui.py` - Main application interface
- `cbs.py` - Core callback functions and audio processing
- `utils/` - Utility modules for audio processing, analysis, and UI themes
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