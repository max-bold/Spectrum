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

## Responsibilities

The application core owns:

- the Dear PyGui lifecycle and main window;
- audio-device discovery and abstract audio sessions;
- project loading and saving;
- module discovery and lifecycle;
- shared settings, background execution, and UI-thread dispatch;
- the plot workspace and composition of visible layers.

A measurement module owns:

- measurement and calculation logic;
- module-specific controls;
- module-specific state and calibration logic;
- serialization and migration of its data;
- renderers that expose its results to the plot workspace.

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
application:

```text
module.__init__(application)
module.on_load()
application builds the UI
module.on_ui_ready()
module.on_activate()
...
module.shutdown()
```

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

The project continues to store multiple measurements and their selection. The
exact next-version schema, ownership of calibration data, and persistence of the
plot workspace remain open design questions.
