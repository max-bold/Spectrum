# Audio analysis library

## Goal

The rewrite should produce both the Spectrum application and a reusable Python
library for applied audio analysis. Existing work in `audioanalysis/` is the
starting point; the library is developed through real application use rather
than as a separate duplicate effort.

## Boundary

`audioanalysis` contains reusable, application-independent code:

- spectrum, level, phase, impedance, THD and THD+N calculations;
- smoothing, weighting, windowing and calibration math;
- signal generators and neutral data/configuration types;
- validation and numerical transformations useful outside Spectrum.

It must not depend on:

- Dear PyGui or DPG item identifiers;
- `SpectrumApplication`, modules, project files or UI state;
- application navigation, persistence or plot layout.

Physical device ownership and application audio-session lifecycle remain in the
application core. Low-level audio helpers belong in the library only when their
API is useful independently of Spectrum.

## Development rule

When implementing or migrating a measurement module:

1. Identify reusable math and data types.
2. Implement them in `audioanalysis` with focused unit tests.
3. Expose a small documented public API.
4. Use that API from the application module.
5. Keep orchestration, state, persistence and GUI code in `spectrum_app`.

The application is the integration client of the library. Algorithms must not
be copied into a module merely to avoid defining a proper library API.

## Quality requirements

- The package is importable and testable without starting Spectrum or DPG.
- Public functions use NumPy arrays and neutral Python data types where
  practical.
- Numerical conventions, units, array shapes and error behavior are documented.
- Algorithm tests include known signals and tolerances, not only application
  callback tests.
- Breaking public-API changes are deliberate and recorded.
