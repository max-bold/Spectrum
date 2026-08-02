# Accepted decisions

## Repository branches

- `main` contains the stable `v0.2.3` application.
- `dev` contains experiments, new measurement modules, and the next
  architecture.
- Focused feature branches may be created from `dev` when useful.
- Stable releases reach `main` only after development has been organized and
  verified.

## Application and modules

- The main application is represented by `SpectrumApplication`.
- Measurement modules are classes derived from a common abstract base class.
- Every module receives the full `SpectrumApplication`, not a restricted module
  context.
- Public application subsystems form the supported module API; underscore-prefixed
  implementation details are not stable API.
- Module `__init__` stores local fields only. Access requiring a fully initialized
  application belongs in lifecycle methods.
- Per-measurement state, module controls, running measurement jobs, and plot
  renderers are separate objects rather than one large module class.
- A module is a long-lived object, not a `Thread`. It creates a separate worker
  for each measurement run and processes worker results in `update()`.
- Each module package explicitly exports its implementation as `MODULE_CLASS`
  from `__init__.py`; the module manager does not guess among imported helper
  classes.
- `Measurement.module_state` is the module's only persistent per-measurement
  state container. Its dictionary holds control values, raw measurements,
  calibration data, and any other measurement-owned data without
  application-defined subdivisions.
- `ModuleManager` only discovers modules and returns `module(id)`. The
  application owns initialization and shutdown; the measurement interface owns
  activation and delegates `update()`, `start_measurement()`, and
  `stop_measurement()`.
- Each module owns its worker lifecycle, error reporting, audio-stream cleanup,
  and the `app_state.measuring` flag, including cleanup after failures.
- Modules update `Measurement.graphs` on the main thread and then set
  `app.app_state.graph_data_changed` so the shared plot can redraw.
- The main action always calls `module.start_measurement()`. A module supplies
  its idle `measurement_button_label`; the application overrides it with
  `STOP` while `app_state.measuring` is true.

## Dependencies and resources

- Modules do not import or control other measurement modules directly.
- Reusable DSP, measurement math, generators, and neutral result types are
  developed in `audioanalysis` and consumed by the application modules.
- `audioanalysis` must remain independent of Dear PyGui, `SpectrumApplication`,
  project files, and module-specific UI state.
- Application code must not duplicate algorithms already exposed by
  `audioanalysis`.
- Physical audio I/O belongs to the application core and is exposed through an
  abstract application-level interface.
- Modules use `app.audio_input` and `app.audio_output`; SoundDevice streams and
  device discovery remain private to the core.
- Audio sample rate and channel count are read-only device defaults available
  immediately after device selection. Block size is a user recommendation, not
  a restriction on module reads and writes.
- Audio-device discovery supports hot plugging. PortAudio refresh is suspended
  while any application audio stream is open and resumes after the last stream
  closes.
- DPG calls are confined to UI-facing application and view code; background
  measurement code returns data rather than manipulating widgets.
- Sound recordings and generated audio stored for recalculation use
  `audioanalysis.ASignal`, not bare NumPy arrays. The Spectrum module stores
  both its completed recording and the exact generated signal in
  `Measurement.module_state`.
- Spectrum input and output run independently from spectral analysis. Online
  Welch calculations use a dedicated analyzer thread and a synchronized
  single-latest-snapshot slot, so delayed calculations cannot build a backlog.
- The Impedance module keeps its generated signals, raw two-channel recordings,
  calibration results, and calculated complex impedance in
  `Measurement.module_state`. Filtering changes recalculate every calibration
  stage and the result from those recordings without a new capture.
- Impedance exposes its experimental SPICE fit only while the module is active.
  The fit runs outside the UI thread and is explicitly marked as needing tests.
- THD+N uses a fixed-level `0.9` logarithmic sweep and records logical input A.
  Input and output blocking calls run on separate threads; adaptive
  mask calibration and STFT analysis run on a third analyzer thread. Only the
  final THD+N percentage curve is published as shared `GraphData`.
- Audio modules see two routed logical input channels (A/B) and one mono output.
  Physical input selection and mono fan-out to enabled device outputs belong to
  `AudioService` and application settings, never to measurement modules.
- THD keeps its raw recording, generated sweep, level history, mask fit, and
  result in `Measurement.module_state`. Its STFT and mask preferences live in
  `AppSettings` under the `thd` namespace.
- Phase measures logical input A relative to electrical reference B using a
  fixed-level `0.9` logarithmic sweep. Audio capture and playback use separate
  workers, and transfer-function analysis uses a third worker. The module
  publishes only unwrapped phase as shared `GraphData`; its compact response
  plot, fitted delay, raw recording, and generator remain in `module_state`.
- Phase delay is fitted within a user-selected subrange of the measurement
  band. A manual distance correction is converted with 343 m/s and added to
  the fitted delay before compensation. Phase wrapping and degrees-per-decade
  conversion remain responsibilities of the shared plot.
- Export commands share the extensible `File -> Export` submenu. Plot PNG
  export captures the Dear PyGui framebuffer and crops the rendered plot with
  HiDPI scaling; it does not rebuild the graph through Matplotlib.
- Module-wide preferences live under the module's namespace in `AppSettings`
  and are saved with the application settings, independently of project files.
  Spectrum stores its generator choice, Welch bucket size, and optional online
  Welch switch there.

## Module loading

- Initial development uses built-in modules with a common lifecycle.
- A module manager will later discover modules from the modules directory.
- Hot reload is not required for the first implementation.

## Project storage

- Next-version `.bms` files contain a pickled `AppState` and are written through
  an atomic temporary-file replacement.
- Project loading validates the state and required module IDs before replacing
  the current session. It never restores an in-progress measurement.
- Pickle project files are trusted local data and must not be opened from
  untrusted sources. Import of the older JSON `.bms` format may be added later.
- Individual measurements use the versioned JSON `.bmm` interchange format.
  Raw audio is stored as little-endian `float32`, compressed with zlib, and
  encoded as base64; calculated NumPy arrays preserve their dtype. File
  compression and I/O run outside the GUI thread.
- Imported measurements receive new measurement and graph IDs, are appended to
  the current project, activated, and made visible. Import is rejected when its
  module is unavailable.

## Open decisions

- Plot workspace, layer compatibility, and axis allocation API.
- Project schema and calibration-storage boundaries.
- Reusable typed signal/slot implementation. Until it is selected, modules use
  bounded lock-protected slots as defined by the module contract.
- Exact module discovery manifest and failure-isolation behavior.
