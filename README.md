# BM Spectrum

BM Spectrum is a modular audio measurement application built with Python and
Dear PyGui. Version 0.3.1 extends the independent measurement-module
architecture introduced in v0.3 and the reusable `audioanalysis` DSP library.

## Measurement modules

- **Spectrum** — logarithmic sweep or pink-noise spectrum measurement;
- **Impedance** — two-channel impedance magnitude and phase measurement;
- **Phase** — acoustic/electrical phase and delay measurement;
- **THD+N** — semi-analog swept residual distortion measurement;
- **RTA** — continuously updated real-time spectrum analyzer.

Measurements are stored independently inside a project and can display several
results on the same plot. Projects use the `.bms` format; individual
measurements can be transferred between projects as `.bmm` files. The plot can
also be exported directly to PNG.

## Requirements

- Python 3.12 or newer;
- an audio input/output device supported by PortAudio;
- Windows is the primary tested platform. macOS builds are produced by CI but
  still require broader hardware testing.

Install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m spectrum_app
```

`python run.py` is an equivalent entrypoint.

## Development

Run the test suite:

```bash
python -m unittest discover -s tests
```

The main source directories are:

- `spectrum_app/` — application core, GUI and measurement modules;
- `audioanalysis/` — reusable signal generation and analysis functions;
- `tests/` — application and DSP tests;
- `docs/design/` — architecture decisions and module contract.

Each measurement module owns its controls, state and workers, while audio
device access, projects, settings and the shared plot are managed by the
application core. See [`docs/design/module_contract.md`](docs/design/module_contract.md)
before adding a module.

## Release builds

Pushing a version tag builds Windows, macOS Intel and macOS Apple Silicon
archives and publishes them to GitHub Releases:

```bash
git tag -a v0.3.1 -m "BM Spectrum v0.3.1"
git push origin v0.3.1
```

macOS bundles are currently not notarized. They may need to be opened through
Finder's **Open** context-menu command on first launch.
