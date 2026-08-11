# Semi-Analog Sweep THD+N Method

The THD+N module follows the operating principle of an analog distortion
analyzer: it tracks a swept fundamental, removes the fundamental region, and
treats the remaining signal as residual distortion plus noise.

## Definition

The implementation uses the IEC residual/total amplitude ratio:

```text
THD+N = residual RMS / total RMS
      = sqrt(residual energy / total energy)
```

The denominator is the complete measured signal, not only the fundamental.
Both the per-frame curve and the integrated result use this definition.

## Measurement signal

The generator produces a fixed-level logarithmic sine sweep. The configured
duration is the time spent crossing the analysis band itself. Fade-in and
fade-out durations are separate application settings and are added to the
generated signal:

```text
generator duration = fade in + analysis-band duration + fade out
```

The logarithmic sweep rate is calculated from the analysis band and its
duration. Start and stop frequencies are then extended automatically so that
fade-in ends exactly at the lower band edge and fade-out starts exactly at the
upper band edge. There is no manually selected sweep-expansion factor.

## Frame analysis

The recording is divided into overlapping Hann-windowed STFT frames. The
defaults are:

- window length: 1 second;
- overlap: 90%;
- fade-in/fade-out: 0.5/0.5 seconds;
- result smoothing: 0.1 octave;
- result points: 1200.

Only complete STFT frames are analyzed. In each frame the largest
positive-frequency bin is taken as the current fundamental `f0`.

For a real one-sided FFT, DC and Nyquist are excluded and every other bin is
weighted by two to restore the energy represented by its negative-frequency
partner. Total frame energy is therefore:

```text
E_total[n] = sum_k P[n, k]
```

## Fixed fundamental rejection window

The old empirical mask calibration and frequency-dependent mask fit are no
longer used. A flat rejection response is placed around the tracked
fundamental:

```text
response(f) = 0,  f0 / c <= f <= f0 * c
              1,  otherwise
```

The default ratio is `c = 1.5`. It can be changed with **Rejection window
ratio** in the THD settings. The same ratio is used at every fundamental
frequency.

Residual frame energy and the raw result are:

```text
E_residual[n] = sum_k P[n, k] * response(f[k])^2
THD+N[n]      = sqrt(E_residual[n] / E_total[n])
```

Energy integration covers the complete one-sided spectrum except DC and
Nyquist. The configured frequency band selects tracked fundamental positions
for the displayed curve and integrated result; it does not truncate residual
energy.

## Frequency curve and integrated result

Raw frame ratios are associated with their tracked fundamental frequencies.
Their squared values are averaged on a logarithmic frequency grid and the
square root is taken afterward, giving RMS smoothing.

The scalar displayed in the status bar is calculated from unsmoothed frame
energies:

```text
THD+N_integrated = sqrt(sum_n E_residual[n] / sum_n E_total[n])
```

## Assumptions and limitations

- The fundamental must be the strongest component in every useful frame.
- A wider rejection window lowers fundamental leakage but can hide nearby
  noise or distortion.
- The fixed ratio is deliberately simple and still needs validation across
  sample rates, window lengths, sweep speeds, and real devices.
- The method reports total residual energy; it does not separate individual
  harmonic orders.
- Fixed recording delay is tolerated because `f0` is tracked from the recorded
  signal rather than inferred from playback time.
