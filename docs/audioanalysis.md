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
  formatting; see the fitting procedure below.

## SPICE impedance model fitting

`audioanalysis.impedance_model` models a series resistance `Re`, a series
inductance `Le`, and a series chain of parallel RLC sections. In the example
schematic, `Re` and `Le` correspond to `R3` and `L3`; each remaining section
contains its own parallel `Rx`, `Lx`, and `Cx`.

For angular frequency `omega = 2*pi*f`, the complete circuit impedance is:

```text
Z(f) = Re + j*omega*Le
       + sum(1 / (1/Rx + 1/(j*omega*Lx) + j*omega*Cx))
```

Every optimization step evaluates this whole complex circuit and then takes
its magnitude. Changing one section therefore affects the fitted response of
the whole model. Only measured `|Z|` is fitted; measured phase is not included
in the objective.

### Progressive section selection

`fit_impedance_auto()` starts with the series RL model, without resonant
sections. If its error exceeds the target, it adds one section at a time and
fits the complete model again. It first chooses the most under-fitted prominent
peak in the measured magnitude. When no unused measured peak has a positive
residual, it searches for a positive local maximum of
`ln(|Z_measured|) - ln(|Z_model|)`. This can add sections for overlapping
resonances that do not form separate peaks in the original measurement.
New anchors must be more than three sample indices from existing anchors.

The default target is `target_log_error=0.02`, and the default maximum is ten
sections. Ten is a limit, not a required model size. Growth stops when the
target is reached, the section limit is reached, no further resonance candidate
is found, or a further optimization fails. If the target cannot be reached,
the best converged candidate satisfying `min_sections` is returned. Failure
before such a candidate exists raises an exception.

### Two fitting stages for each candidate

1. Initialize `Re` from the minimum measured magnitude and `Le` from the
   high-frequency response. Set each section's `Lx` to **1 mH** and compute
   `Cx = 1 / ((2*pi*f0)^2 * Lx)` for its selected resonance frequency. Fit
   `Re`, `Le`, and all section resistances jointly, keeping `Lx` and `Cx` fixed.
2. If RMS log error still exceeds the target, also fit the section inductances
   within **1 uH to 1 H**. Recompute each capacitance as its inductance changes,
   preserving its resonance frequency. Keep this refined stage if it reaches
   the target or improves RMS log error by more than 1%; otherwise keep the
   fixed-inductance stage.

Resonance frequencies remain fixed throughout both stages. Section
inductances control peak widths as well as their interaction with section
resistances. Parameters are optimized in logarithmic coordinates with bounds,
using `scipy.optimize.least_squares` and the robust `soft_l1` loss.

### Accuracy, API, and application settings

The reported residual and accuracy are defined as:

```text
residual[i] = ln(|Z_model[i]|) - ln(|Z_measured[i]|)
rms_log_error = sqrt(mean(residual**2))
max_abs_log_error = max(abs(residual))
```

A target of `0.02` is displayed as **2% RMS log error**. For small deviations
this approximates RMS relative error; it does not guarantee that the error at
every frequency is below 2%. Maximum log error is reported separately.

```python
from audioanalysis import fit_impedance_auto, format_spice_table

# frequency and measured_magnitude are matching one-dimensional arrays.
best, candidates = fit_impedance_auto(
    frequency,
    measured_magnitude,
    target_log_error=0.02,
    min_sections=0,
    max_sections=10,
    max_evaluations=2000,
)
values = format_spice_table(best)
```

`candidates` contains one selected result for each successfully evaluated
section count, including the initial RL model. `min_sections` specifies the
minimum acceptable section count. Non-finite and non-positive data points are
removed before fitting; at least three valid points must remain.

`fit_impedance(..., sections=N)` fits a specified number of detected prominent
measured peaks and raises an error if fewer peaks are available. Passing
`refine_inductances=False` keeps section inductances at 1 mH. This explicit
section-count API uses a fixed refinement threshold of `0.02`.

`FitResult.physical_params` stores `[Re, Le, R, f0, Q, ...]`, where
`Q = R / (2*pi*f0*L)`. `inductances_refined` identifies the selected fitting
stage, and automatic fits also store `target_log_error` and `stop_reason`:

- `target_reached` — the requested RMS log error was reached.
- `section_limit` — growth reached the section limit without reaching the target.
- `no_resonance` — no additional resonance candidate was found.
- `optimization_failed` — a further fit failed; the best valid model was retained.

`format_spice_table()` converts the model into displayed component values:
resistance in ohms, inductance in mH, and capacitance in uF, with three
significant digits. It does not generate a SPICE netlist.

In the application, **Settings -> Impedance** exposes the target RMS log error
as a percentage, defaulting to **2%**, with an allowed range of **0.1–20%**.
The preference is persisted in application settings. **Tools -> SPICE Fit**
fits a completed impedance measurement in a background thread and displays the
component values, selected stage, target, achieved errors, and stop reason.
Legacy cached fits or fits computed for a different target are recalculated on
the next request. A new measurement or completed reprocessing clears the cache.

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
