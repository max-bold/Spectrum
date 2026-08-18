from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg
import numpy as np

from spectrum_app.gui.controls import LevelMeter, add_level_meter

if TYPE_CHECKING:
    from spectrum_app.modules.thd.module import THDModule


class THDView:
    ROOT = "module::thd::controls"
    BOTTOM = "module::thd::bottom"
    METER = "module::thd::level_meter"
    LEVEL_PLOT = "module::thd::level_plot"
    LEVEL_X_AXIS = "module::thd::level_x_axis"
    LEVEL_Y_AXIS = "module::thd::level_y_axis"
    LEVEL_SERIES = "module::thd::level_series"

    def __init__(self, module: "THDModule") -> None:
        self.module = module
        self.level_meter: LevelMeter | None = None

    def build(
        self,
        controls_parent: int | str,
        bottom_parent: int | str,
        state: dict[str, Any],
    ) -> None:
        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=controls_parent,
            tag=self.ROOT,
        ):
            dpg.add_text("Frequency range, Hz")
            dpg.add_input_intx(
                size=2,
                default_value=list(state["band"]),
                width=-1,
                callback=self._set_band,
            )
            dpg.add_text("Band duration, s")
            dpg.add_input_float(
                default_value=state["duration"],
                min_value=1.0,
                min_clamped=True,
                step=0,
                width=-1,
                callback=self._set_duration,
            )
            dpg.add_text("Smoothing, octaves")
            dpg.add_input_float(
                default_value=state["smoothing_octaves"],
                min_value=0.01,
                min_clamped=True,
                step=0,
                width=-1,
                callback=self._set_smoothing,
            )

        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=bottom_parent,
            tag=self.BOTTOM,
            horizontal=True,
        ):
            self.level_meter = add_level_meter(
                self.BOTTOM,
                self.METER,
                self.LEVEL_PLOT,
                labels=("A",),
            )
            with dpg.plot(  # pyright: ignore[reportGeneralTypeIssues]
                tag=self.LEVEL_PLOT,
                width=-1,
                height=-1,
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    tag=self.LEVEL_X_AXIS,
                )
                dpg.add_plot_axis(
                    dpg.mvYAxis,
                    tag=self.LEVEL_Y_AXIS,
                )
                dpg.add_line_series(
                    [],
                    [],
                    tag=self.LEVEL_SERIES,
                    label="A",
                    parent=self.LEVEL_Y_AXIS,
                )
        dpg.set_axis_limits(self.LEVEL_Y_AXIS, 0.0, 1.0)
        self.update_levels(
            np.asarray(state["level_time"], dtype=np.float64),
            np.asarray(state["level_values"], dtype=np.float64),
            0.0,
            duration=float(state["duration"]),
        )

    def destroy(self) -> None:
        for item in (self.ROOT, self.BOTTOM):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.level_meter = None

    def update(self) -> None:
        if self.level_meter is not None:
            self.level_meter.resize()

    def set_enabled(self, enabled: bool) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.configure_item(self.ROOT, enabled=enabled)

    def update_levels(
        self,
        times: np.ndarray,
        levels: np.ndarray,
        current: float,
        *,
        duration: float,
    ) -> None:
        if self.level_meter is not None:
            self.level_meter.set_levels(current)
        dpg.set_value(
            self.LEVEL_SERIES,
            [times.tolist(), levels.tolist()],
        )
        maximum_time = max(duration + 0.7, float(times[-1]) if times.size else 0.0)
        dpg.set_axis_limits(self.LEVEL_X_AXIS, 0.0, maximum_time)

    def _set_band(self, sender: int | str, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_duration(self, sender: int | str, value: float, user_data=None) -> None:
        duration = self.module.set_setting("duration", value)
        dpg.set_value(sender, duration)

    def _set_smoothing(self, sender: int | str, value: float, user_data=None) -> None:
        smoothing = self.module.set_setting("smoothing_octaves", value)
        dpg.set_value(sender, smoothing)
