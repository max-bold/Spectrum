# Application architecture

## Direction

The next version uses an object-oriented application architecture:

```text
SpectrumApplication
|- audio services
|- project and measurements
|- module manager
|- plot workspace
|- settings and application events
`- main window
```

Measurement modules are classes derived from a common abstract base class. A
module receives the complete `SpectrumApplication` instance. The application
remains a composition of focused services rather than implementing every
operation itself.

The normative lifecycle, state ownership, audio, worker-thread, and
synchronization rules are defined in the
[measurement module contract](module_contract.md).

## Responsibilities

The application core owns:

- the Dear PyGui lifecycle and main window;
- audio-device discovery and abstract audio sessions;
- project loading and saving;
- module discovery and application-level lifecycle;
- shared settings, background execution, and UI-thread dispatch;
- the plot workspace and composition of visible layers.

A measurement module owns:

- measurement and calculation logic;
- module-specific controls;
- module-specific state and calibration logic;
- serialization and migration of its data;
- renderers that expose its results to the plot workspace.

All persistent per-measurement data owned by a module lives in the single
`Measurement.module_state` dictionary. The module defines and interprets its
keys; the application core does not split that data into additional settings,
recording, calibration, or result-state objects. This dictionary may contain
module control values, reusable raw data, and calibration data. Calculated plot
series remain in `Measurement.graphs` because they are consumed by the shared
plot workspace. Module-wide preferences live in the module's namespace inside
`AppSettings` and are not loaded from a project.

Modules must not depend directly on other measurement modules.

Reusable signal-processing algorithms belong to the independent
`audioanalysis` package. Measurement modules compose those algorithms with
application state, audio sessions, and UI instead of maintaining private DSP
implementations.

## Object lifetime

A module describes a measurement type and is not the storage for one particular
measurement. A project can contain multiple measurement instances of the same or
different module types. DPG item identifiers belong to disposable view objects,
not to persistent measurement state.

Module construction is phased to avoid accessing a partially initialized
application. The manager owns one module instance per measurement type:

```text
module.__init__()
application builds the UI
module.initialize(application)
module.activate(measurement)
...
module.deactivate()
module.shutdown()
```

The module object is not a `Thread`. Each measurement run may create its own
worker, while `module.update()` transfers results to application state from the
main thread.

The module owns the complete runtime of its measurement. It sets
`app.app_state.measuring` while work is active and clears it after normal
completion, user interruption, or failure. It also reports its own errors and
closes every audio stream it opened. `ModuleManager` only discovers modules and
provides lookup by ID. The application initializes and shuts down every module;
the measurement panel activates the selected module and delegates
start/stop/update calls. Neither component supervises the module's worker or
repairs its runtime state.

## Audio

The core owns the physical audio interface. Modules use abstract operations such
as open, read, write, and close without depending on SoundDevice-specific
parameters or stream objects.

## Visualization requirements

The application must support:

- several plot panels at the same time;
- multiple measurement layers in one plot;
- combinations such as spectrum with phase or impedance;
- several stored measurements and switching between them.

The exact plot/axis/layer API remains open. The design must account for Dear
PyGui's single X-axis and maximum of three Y-axes per plot.

## Events and threading

The preferred public model is typed signals/slots and lifecycle callbacks such
as update and completion. Background audio and calculations must not manipulate
DPG directly. The exact mechanism that marshals worker-thread emissions onto the
UI thread remains open and must not become a general untyped event queue.

## Project storage

The project stores the complete `AppState`, including multiple measurements,
their module state, calculated graphs, visibility, and current selection.
`AppSettings` remains separate and is not changed when a project is opened.
