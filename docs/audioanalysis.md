# audioanalysis

`audioanalysis` is the reusable DSP and audio-measurement library layer extracted from Spectrum.

Audio arrays use `(samples, channels)` shape throughout the library. Mono
audio is represented as `(samples, 1)`. Use `as_channels` at API boundaries
when user input needs to be converted into that shape.

The package is intentionally split into pure analysis modules and optional audio-device integration:

- `audioanalysis.generators` - test signals and continuous generator workers.
- `audioanalysis.smoothing` - logarithmic smoothing windows and grids.
- `audioanalysis.spectrum` - periodogram/Welch analysis and reference-channel math.
- `audioanalysis.levels` - peak normalization, channel-shape conversion, and level-meter calculations.
- `audioanalysis.audioio` - `sounddevice` device and play/record helpers.
- `audioanalysis.thd` - harmonic distortion primitives.
- `audioanalysis.impedance` - reusable impedance math.

Install locally while developing:

```bash
pip install -e .
```

Minimal example:

```python
from audioanalysis import FrequencyBand, SpectrumConfig, analyze_spectrum, log_chirp, power_db

sample_rate = 48_000
signal = log_chirp(sample_rate, sample_rate, FrequencyBand(20, 20_000), pad=0, fade=0)
# signal.as_array().shape == (48000, 1)
result = analyze_spectrum(signal.as_array(), SpectrumConfig(sample_rate=sample_rate))
level_db = power_db(result.values)
```
