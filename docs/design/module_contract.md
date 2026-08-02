# Measurement module contract

This document defines the contract for measurement modules in Spectrum. It is
normative for new modules and for changes to existing modules. The exact helper
classes used to implement the contract may evolve, but ownership, thread
affinity, cleanup, and state-transfer rules must remain intact.

The main design rule is:

> A module coordinates a measurement, but it does not own the application,
> physical audio backend, GUI event loop, or reusable measurement math.

## Responsibility boundaries

The application core owns:

- the Dear PyGui context and main loop;
- the main window, shared measurement selector, status bar, and plot;
- `AppState`, project loading, and project saving;
- `AppSettings` and its persistence;
- audio-device discovery, PortAudio lifetime, and physical streams;
- module discovery and application-wide module lifecycle.

A measurement module owns:

- its measurement workflow and validation;
- its controls and module-specific windows or menu items;
- acquisition and calculation worker lifetimes;
- interpretation of its `Measurement.module_state` dictionary;
- conversion of completed results into `GraphData`;
- user-facing errors and status messages produced by its work;
- restoring its idle state after completion, cancellation, or failure.

`audioanalysis` owns reusable, application-independent math. A function belongs
there when it can operate on `ASignal`, arrays, configuration values, and
neutral result objects without importing Dear PyGui or `spectrum_app`.

## Recommended package shape

```text
spectrum_app/modules/example/
|- __init__.py   # exports MODULE_CLASS
|- module.py     # lifecycle and state-machine coordinator
|- jobs.py       # acquisition and calculation workers
|- view.py       # DPG controls and module-specific visualization
`- settings.py   # optional application-wide module preferences
```

This is a responsibility split, not a requirement to create empty files. Keep
small modules small and add files only when the corresponding responsibility
exists.

The package root must explicitly export one module class:

```python
from spectrum_app.modules.example.module import ExampleModule

MODULE_CLASS = ExampleModule
```

The manager discovers and instantiates `MODULE_CLASS`. It does not infer a
class by scanning imports and it does not supervise measurement work.

## `BaseModule` lifecycle

Every module derives from `BaseModule`. A module object is long-lived and is
not itself a `Thread`. There is one module object per measurement type, while a
project may contain many `Measurement` instances of that type.

The normal call sequence is:

```text
__init__
initialize(app)
    activate(measurement)
        update() repeated once per GUI frame
        start_measurement()
        update() repeated while workers run
        stop_measurement() when requested
    deactivate()
    ...another activate/deactivate pair may follow...
shutdown()
```

All lifecycle methods are called on the main/UI thread.

### `__init__()`

`__init__()` must only create module-local fields and synchronization
primitives. The application is not available yet. Do not query devices, create
DPG items, start threads, or access settings here.

Good constructor fields include:

- `None` references for future view and worker objects;
- `Lock`, `Event`, and bounded pending-result slots;
- a monotonically increasing operation revision;
- transient status needed by the coordinator.

### `initialize(app)`

`initialize()` receives the complete `SpectrumApplication`. It must call
`super().initialize(app)` and may create resources that live for the entire
application session, such as a module settings window.

Items that should exist only while the module is selected belong in
`activate()`, not `initialize()`. For example, Impedance adds `Tools -> SPICE
Fit` on activation and removes it on deactivation.

Initialization must not start a measurement or open audio streams.

### `activate(measurement)`

`activate()` must:

1. call `super().activate(measurement)`;
2. supply defaults for missing `module_state` keys;
3. repair serialized transient states such as `measuring` or `stopping`;
4. build controls in the application-provided hosts;
5. restore calculated graphs when reusable stored results exist;
6. add activation-scoped Tools and Settings items.

Activation may be repeated with different measurements. Never assume the
previously active measurement has the same module state.

### `measurement_button_label`

The module supplies the idle label for the application's main action button.
The default is `MEASURE`. A workflow may return another label, such as
`Calibrate`. While `app.app_state.measuring` is true, the application always
overrides the label with `STOP`.

The main action always enters through `start_measurement()`, even when the
current action is calibration rather than a regular measurement.

### `start_measurement()`

`start_measurement()` is a non-blocking command. It validates the current
state, constructs a worker, marks the operation active, and starts the worker.
It must return to the GUI loop immediately.

It must not:

- perform blocking `open`, `read`, or `write` calls on the UI thread;
- run FFT, STFT, optimization, smoothing, or whole-record calculations;
- wait for a worker with `join()`;
- call DPG from a worker callback.

Starting an already active operation should be ignored or reported clearly,
not create a second acquisition.

### `stop_measurement()`

`stop_measurement()` requests cooperative cancellation and returns quickly. It
must be safe when no worker is running and safe when called repeatedly.

For a blocking audio worker, setting an event alone is insufficient: the
module must also close the streams it opened so a blocked `read()` or `write()`
can return. Cleanup must not depend on another audio block arriving.

### `update()`

`update()` is the bridge back to the UI thread. It should:

1. take pending worker messages under a short lock;
2. release the lock;
3. reject stale messages using their operation revision;
4. update `Measurement.module_state` and `Measurement.graphs`;
5. set `app.app_state.graph_data_changed` when graph data changed;
6. update the module view and application status;
7. finish or advance the workflow.

Never run heavy calculations while holding the lock or inside `update()`.
Never call a worker callback while holding a lock that `update()` also needs.

### `deactivate()`

`deactivate()` is a hard ownership boundary. It must leave no active audio
session, no live acquisition that can affect application state, and no
activation-scoped DPG items.

The safe order is:

1. invalidate the current operation revision;
2. request cancellation;
3. close input and output streams;
4. join owned acquisition threads with a bounded timeout;
5. clear pending messages and worker references;
6. clear `app.app_state.measuring`;
7. destroy the view and activation-scoped menu items;
8. call `super().deactivate()`.

Do not rely on the measurement panel calling `stop_measurement()` first.
`deactivate()` must be correct on its own.

### `shutdown()`

`shutdown()` releases application-lifetime module resources. It must be safe
after partial initialization and after a previous `deactivate()`. It should
stop remaining workers, destroy module settings windows if applicable, and
call `super().shutdown()`.

Cleanup methods should be idempotent. Error recovery frequently exercises
partial lifecycles.

## Persistent and transient state

`Measurement.module_state` is the module's only persistent per-measurement
container. The module owns its key names and migrations. Typical contents are:

- control values that must follow the measurement;
- generated `ASignal` instances;
- raw recorded `ASignal` instances;
- calibration data;
- neutral calculation result objects and arrays;
- a persistent workflow state such as `uncalibrated`, `calibrated`, or
  `completed`.

It must not contain:

- threads, locks, events, callbacks, or queues;
- DPG item IDs or view objects;
- `AudioInput`, `AudioOutput`, SoundDevice, or stream objects;
- references to `SpectrumApplication`;
- a transient claim that an operation is still running after project load.

Project files pickle `AppState`. Therefore every stored value must be
pickle-compatible and meaningful after application restart. Generated and
recorded audio is stored as `ASignal`, not a bare array with a separate sample
rate.

Only the UI thread should mutate `module_state`. A worker returns a value; the
module applies it during `update()`. This avoids locking a large, nested,
serialized dictionary and prevents projects from observing half-written
results.

Application-wide module preferences belong in the module's namespace in
`AppSettings`. They must be JSON-serializable and are not restored from a
project. A useful rule is:

- if a value describes this particular measurement, store it in
  `module_state`;
- if it describes how the user generally wants the module to behave, store it
  in `AppSettings`.

## Audio-interface contract

Modules use only:

```text
app.audio_input.open() -> bool
app.audio_input.read(samples) -> ndarray
app.audio_input.close() -> bool
app.audio_input.sample_rate
app.audio_input.channels
app.audio_input.block_size

app.audio_output.open() -> bool
app.audio_output.write(data)
app.audio_output.close() -> bool
app.audio_output.sample_rate
app.audio_output.channels
app.audio_output.block_size
```

A module must never import `sounddevice`, create a PortAudio stream, choose a
device, refresh devices, or pass SoundDevice-specific stream parameters.

`sample_rate` and `channels` are read-only defaults of the selected device and
are available before `open()`. `block_size` is the user's recommended transfer
size. It is not a requirement that all module reads and writes use exactly that
size.

### Opening streams

Opening a stream may wait for an in-progress device refresh. This is why
`open()` belongs in the acquisition worker, never in a DPG callback.

When both directions are needed:

1. validate advertised input/output sample rates and channel counts;
2. open the first stream;
3. open the second stream;
4. if either operation fails, close both directions in `finally`;
5. start actual transfer only after both opens succeeded.

An active audio session suspends device refresh. Closing the final stream lets
the updater resume. Leaking one stream therefore leaks both a device resource
and the hot-plug synchronization reservation.

### Reading and writing concurrently

Input `read()` and output `write()` are blocking operations. A single loop that
writes a block, reads a block, and performs analysis serializes three unrelated
deadlines. It is vulnerable to underflow, overflow, and increasing latency.

For duplex measurements, use independent input and output execution paths:

```text
acquisition coordinator
|- input loop:  read -> append recording -> publish level
`- output loop: slice generator -> adapt channels -> write
```

One of these loops may be the coordinator thread and the other a child thread.
Both must share a cancellation event and report their first failure to the
coordinator. The coordinator owns final closure and emits exactly one
completion message.

Output data must be shaped for the selected output channel count. Input workers
should retain only the channels required by the measurement, but must preserve
the sample rate in the resulting `ASignal`.

### Audio-worker cleanup pattern

Every audio worker should follow this structure:

```python
def run(self) -> None:
    recording = None
    error = None
    try:
        if not self.audio_input.open():
            raise RuntimeError("Cannot open audio input")
        if not self.audio_output.open():
            raise RuntimeError("Cannot open audio output")
        self._run_input_and_output()
        recording = self._build_recording()
    except Exception as exc:
        if not self.cancel_event.is_set():
            error = str(exc)
    finally:
        cancelled = self.cancel_event.is_set()
        self.cancel_event.set()
        self.audio_output.close()
        self.audio_input.close()
        self._join_child_workers()
        self.on_complete(recording, error, cancelled)
```

The callback is invoked once, including on cancellation and partial-open
failure. Closing both directions is harmless because core close operations are
idempotent.

## Heavy-calculation contract

Whole-record Welch, STFT, smoothing over large grids, nonlinear optimization,
model fitting, and recalculation of stored data are heavy work. They must not
run in:

- a DPG callback;
- `update()`;
- an audio input or output loop;
- a lock-protected critical section.

Use a dedicated calculation worker. Acquisition completion should transfer an
immutable input snapshot to that worker and return. The calculation worker must
not access DPG or mutate `Measurement`.

For online analysis, the producer may generate snapshots faster than analysis
can consume them. Use a synchronized single-latest-snapshot slot:

```text
audio producer -> [latest snapshot, capacity 1] -> analyzer
```

Replacing an obsolete preview is correct. Building an unbounded FIFO queue is
not: it increases memory use and guarantees that the displayed result falls
behind real time. Final completion is not a preview and must have its own slot
that cannot be overwritten by online updates.

Long algorithms should expose cancellation checkpoints where practical. If a
third-party call cannot be interrupted, invalidate its revision so its result
is ignored, and do not block deactivation indefinitely waiting for it. Such a
worker must still hold no audio stream and no DPG resource.

## Worker-to-UI synchronization

Worker callbacks run on worker threads. A callback may only publish a typed
message into a bounded synchronized slot and return. It must not:

- call DPG;
- modify `AppState`, `Measurement`, graphs, or module controls;
- select another measurement or module;
- show an error dialog;
- perform another expensive calculation inline.

The current implementation uses `Lock`-protected pending slots. A future typed
signal/slot helper may replace this mechanism, but it must preserve the same
thread boundary and bounded-delivery semantics.

A recommended message contains:

- operation revision;
- operation kind;
- neutral result or recording;
- error text, if any;
- cancellation state.

Use the lock only to swap references:

```python
def _receive_completion(self, message: Completion) -> None:
    with self._lock:
        self._pending_completion = message

def _take_completion(self) -> Completion | None:
    with self._lock:
        message = self._pending_completion
        self._pending_completion = None
    return message
```

Do validation, state mutation, graph creation, and UI updates after releasing
the lock.

### Operation revisions

Every asynchronous operation should carry a monotonically increasing revision
or generation token:

```text
start operation -> increment revision -> capture token in callbacks
cancel/deactivate -> increment revision
update receives result -> apply only if token == current revision
```

This prevents a late result from an old measurement, cancelled calculation, or
previous activation from corrupting the newly active measurement. Checking
only `thread.is_alive()` is not sufficient because a worker can finish and
publish between two UI frames.

## Measurement state machine

Use an explicit operation enum and a small persistent workflow state. Avoid a
collection of loosely related booleans such as `is_running`, `is_calibrating`,
`has_result`, and `waiting` whose combinations can become invalid.

For every operation define:

- allowed source states;
- active state;
- success state;
- cancellation fallback state;
- failure fallback state;
- which raw inputs and results are retained or invalidated.

Capture the fallback before changing the workflow. Calibration stage 2, for
example, usually falls back to `waiting_reference`, while a cancelled regular
measurement falls back to `calibrated`.

`app.app_state.measuring` is the application-wide statement that the active
module owns an operation controlled by the main `STOP` button. The module must:

- set it before or atomically with starting the worker;
- leave it true while acquisition or its required final calculation is active;
- clear it after success, cancellation, start failure, worker failure,
  deactivation, and shutdown.

Do not use this flag as the only internal state machine.

## Graph publication

Workers return neutral results. During `update()`, the module creates or updates
`GraphData` in the active `Measurement` and then sets:

```python
app.app_state.graph_data_changed = True
```

The shared plot owns axes, scale selection, phase wrapping, and rendering. A
module supplies raw series and semantic `AxisSpec` values. It must not reach
into the shared plot and create DPG series directly.

Prefer updating existing `GraphData` objects so their IDs and visibility remain
stable across recalculation. When graphs are removed, remove their IDs from
`visible_graph_ids` as well.

## Errors and cancellation

Expected operational failures should become concise user-facing status text.
Preserve the exception for tests or diagnostics when useful, but never let a
worker exception silently terminate a thread.

On every exit path verify all four outcomes:

- streams are closed;
- child workers are stopped or made stale;
- `app_state.measuring` is false when the operation is over;
- the workflow returns to a valid state.

Cancellation is not an error. Do not display a cancellation requested by the
user as an audio failure.

Avoid broad `except BaseException` in module code. Catch `Exception` unless the
worker is deliberately recording fatal interpreter-level failures for later
re-raising.

## Best practices checklist

Before considering a module complete, verify:

- [ ] `MODULE_CLASS` explicitly exports one `BaseModule` subclass.
- [ ] The module object is not a `Thread`.
- [ ] `__init__()` does not use the application or DPG.
- [ ] Every DPG call happens on the main thread.
- [ ] Audio is accessed only through `app.audio_input` and `app.audio_output`.
- [ ] Blocking input and output have independent execution paths.
- [ ] Heavy calculations have a worker separate from audio transfer.
- [ ] Online previews use a capacity-one latest-value slot.
- [ ] Final completion cannot be overwritten by a preview.
- [ ] Pending messages are protected by a short-held lock.
- [ ] Every message carries an operation revision.
- [ ] Workers never mutate `Measurement` or `AppState`.
- [ ] Raw recordings and generators are stored as `ASignal` when recalculation
      is required.
- [ ] `module_state` contains no runtime objects.
- [ ] `start_measurement()` returns immediately.
- [ ] `stop_measurement()`, `deactivate()`, and `shutdown()` are idempotent.
- [ ] Blocking audio calls are unblocked by closing streams during stop.
- [ ] Partial audio-open failure closes the other direction.
- [ ] Success, failure, cancellation, and deactivation all clear measuring
      state correctly.
- [ ] Graph changes are published on the main thread with
      `graph_data_changed`.
- [ ] Activation-scoped Tools and Settings entries are removed on deactivate.
- [ ] Project load repairs serialized transient workflow states.

## Minimum verification for a new module

A new module should have four test layers:

1. **Pure math tests** in `audioanalysis`: known signals, numerical reference
   values, validation, channel shape, and sample-rate handling.
2. **Worker tests** with fake `AudioInput` and `AudioOutput`: open failure,
   arbitrary block sizes, cancellation, underflow/overflow propagation, and
   guaranteed closure.
3. **Module tests** with a fake DPG backend: lifecycle, menu cleanup, state
   transitions, stale revisions, graph publication, and project restoration.
4. **One end-to-end synthetic workflow**: generator -> fake duplex audio -> raw
   `ASignal` -> calculation worker -> `GraphData`.

Real-device testing remains necessary before release, but it is not a substitute
for deterministic worker and synchronization tests.

## Anti-patterns

Do not introduce:

- direct `sounddevice` imports inside a measurement module;
- one giant thread that performs write, read, analysis, and UI work;
- background DPG calls that appear to work until timing changes;
- an unbounded generic event queue;
- polling a mutable result object without a lock;
- storing live worker objects in project state;
- using daemon threads as the cleanup strategy;
- waiting indefinitely in `deactivate()` or on the UI thread;
- applying a worker result without confirming its operation revision;
- recalculating reusable DSP privately inside the application module.

When a new requirement appears to need one of these patterns, stop and revise
the contract or shared infrastructure explicitly instead of adding a local
exception that future modules will copy.
