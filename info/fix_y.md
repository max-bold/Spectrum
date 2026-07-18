# Manual Scaling of the Phase/Dir Axis

## Goal

The impedance measurement plot uses two vertical axes:

- the left `Z` axis, which must automatically include zero and allow manual
  zooming and panning;
- the right `Phase/Dir` axis.

In `Angle` mode, the right axis is intentionally fixed to `-180..180 deg`. In
`Dir` mode, the range must be fitted to the phase derivative data and remain
available for manual zooming and panning afterward.

Automatic scaling in `Dir` mode currently works. Manual zooming and panning of
the right axis do not work. The left `Z` axis can be scaled with the mouse as
expected.

The issue is reproducible with Dear PyGui 2.1.1 on Windows.

## Current Implementation

The relevant code is located in:

- `spectrum_app/impedance/gui.py` — plot and axis creation;
- `spectrum_app/impedance/cbs.py` — Angle/Dir switching and axis limit control;
- `spectrum_app/impedance/imp_measure.py` — range calculation.

The right axis uses a real secondary axis slot:

```python
dpg.plot_axis(
    dpg.mvYAxis2,
    opposite=True,
    no_side_switch=True,
)
```

`opposite=True` is required to keep `mvYAxis2` on the right. Without it, DPG
2.1.1 displays the axis on the left.

In `Dir` mode, the application calculates the minimum and maximum finite
values with 5% headroom, calls `set_axis_limits()`, and calls
`set_axis_limits_auto()` after two render-loop iterations. The frame counter
follows the working approach used by the main application UI.

In `Angle` mode, only the following fixed limits are applied:

```python
dpg.set_axis_limits(phase_axis, -180.0, 180.0)
```

## Attempted Solutions

### set_axis_limits_auto and fit_axis_data in one update

The first implementation called these functions when switching to `Dir`:

```python
dpg.set_axis_limits_auto(phase_axis)
dpg.fit_axis_data(phase_axis)
```

The range remained at `-180..180`, and manual scaling did not work. A likely
reason is that removal of fixed limits is applied during a later render frame,
while the fit request is processed against the old limits.

### Deferred fit using set_frame_callback

Another implementation removed the limits and called `fit_axis_data()` on the
next frame through `set_frame_callback()`. The result did not change.

It was also considered that multiple callbacks registered for the same frame
could replace each other, so Z and Phase updates were combined into one frame
callback. This did not solve the issue either.

### A second axis created as another mvYAxis

Originally, the right axis was created by adding `mvYAxis` a second time. DPG
placed it on the right visually, but it was not a real independent axis slot.

A minimal test with data spanning `-900..700` showed that `fit_axis_data()` was
ignored for this axis: `get_axis_limits()` still returned `(-180, 180)`.

### mvYAxis2 without opposite

Changing the axis to `mvYAxis2` created a separate slot but moved it to the
left. Automatic and manual scaling also did not work in the tested version.
The undocumented-in-the-short-reference `opposite` configuration option was
found later.

### The two-frame approach from the main application

The main application uses a counter of two render-loop iterations after
`set_axis_limits()`, followed by `set_axis_limits_auto()`. It does not use a
frame callback.

This approach was copied into the impedance UI. `fit_axis_data()` still did
not work for the repeated `mvYAxis`, so the `Dir` range was calculated in
Python instead.

For test data spanning `-900..700`, the calculated limits were `-980..780`.
The range remained at those values after the delayed
`set_axis_limits_auto()`. This fixed automatic scaling.

### mvYAxis2 with opposite=True

The current implementation uses `mvYAxis2`, `opposite=True`, and
`no_side_switch=True`. A minimal test confirmed the following configuration:

```text
opposite=True
no_side_switch=True
lock_min=False
lock_max=False
```

Despite these values, testing in the complete application showed that the
right axis still cannot be scaled manually. The automatically calculated
range works correctly.

## Suggested Next Steps

1. Reproduce the issue in a standalone DPG 2.1.1 application and test mouse
   zoom directly on `mvYAxis2` with `opposite=True`.
2. Compare mouse input while hovering over the axis itself and over the plot
   area. With multiple Y axes, ImPlot may route general plot zoom only to the
   primary axis.
3. Inspect plot input flags and modifiers such as `no_inputs`, `zoom_mod`,
   `vertical_mod`, and `override_mod`, as well as the axis context menu.
4. Run the same minimal example with the current Dear PyGui release. The
   documentation and recent releases contain changes related to secondary
   axes, while the application currently uses version 2.1.1.
5. If this is a DPG 2.1.1 limitation or bug, either update DPG or add explicit
   range controls and a Reset/Auto button for `Dir`.

The issue is tracked in `TODO.md` for version 0.2.4.
