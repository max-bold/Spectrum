# audioanalysis

`audioanalysis` is the reusable DSP and audio-measurement library layer extracted from Spectrum.

Public APIs that accept or return audio signals use `ASignal`. Its internal
array shape is always `(samples, channels)`; mono audio uses `(samples, 1)`.
Plain arrays remain appropriate for non-audio results such as spectra and
frequency grids.

The package is intentionally split into pure analysis modules and optional audio-device integration:

- `audioanalysis.generators` - mono test-signal generators, including
  `periodic_pink_noise()` for FFT-bin-aligned periodic excitation.
- `audioanalysis.smoothing` - logarithmic smoothing windows and grids.
- `audioanalysis.spectrum` - periodogram/Welch analysis and reference-channel math.
- `audioanalysis.rta` - configurable multichannel periodogram analysis on a
  logarithmically smoothed frequency grid and equal-log-band density
  compensation.
- `audioanalysis.phase` - two-channel transfer response, weighted delay
  estimation, delay compensation, and shared phase display transformations.
- `audioanalysis.levels` - peak normalization, channel-shape conversion, and level-meter calculations.
- `audioanalysis.audioio` - `sounddevice` device and play/record helpers.
- `audioanalysis.thd` - conventional spectrum-based THD and the semi-analog
  swept THD+N method, including sweep generation, adaptive-mask calibration,
  and frequency-resolved analysis.
- `audioanalysis.impedance` - signal generation, two-stage calibration, and
  impedance calculation.
- `audioanalysis.impedance_model` - SPICE-equivalent model fitting and table
  formatting. The automatic fit currently needs testing.

## Logarithmic smoothing width

`audioanalysis.smoothing.log_window()` and the smoothing helpers interpret
`width` as the full width at half maximum (FWHM), in octaves. Tapered windows
are truncated at approximately −30 dB (`weight = 0.001`). The flat window is
the discontinuous exception: its support is exactly `width` octaves.

`grid_smooth()` applies the logarithmic-frequency Jacobian when averaging a
linearly spaced FFT grid. This keeps the window centered in log-frequency even
though the source bins are uniformly spaced in hertz.

Install locally while developing:

```bash
pip install -e .
```

Minimal example:

```python
from audioanalysis import FrequencyBand, SpectrumConfig, analyze_spectrum, log_chirp, power_db

sample_rate = 48_000
signal = log_chirp(sample_rate, sample_rate, FrequencyBand(20, 20_000))
# signal.as_array().shape == (48000, 1)
result = analyze_spectrum(signal, SpectrumConfig())
level_db = power_db(result.values)
```
