from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg
import numpy as np

from spectrum_app.gui.controls import LevelMeter, add_level_meter

if TYPE_CHECKING:
    from spectrum_app.modules.phase.module import PhaseModule


class PhaseView:
    ROOT = "module::phase::controls"
    BOTTOM = "module::phase::bottom"
    METER = "module::phase::level_meter"
    LEVEL_PLOT = "module::phase::level_plot"
    LEVEL_X_AXIS = "module::phase::level_x_axis"
    LEVEL_Y_AXIS = "module::phase::level_y_axis"
    LEVEL_SERIES_A = "module::phase::level_series_a"
    LEVEL_SERIES_B = "module::phase::level_series_b"
    DELAY_TEXT = "module::phase::delay"
    BAND = "module::phase::band"
    DURATION = "module::phase::duration"
    SMOOTHING = "module::phase::smoothing"
    POINTS = "module::phase::points"
    DELAY_CORRECTION = "module::phase::delay_correction"

    def __init__(self, module: "PhaseModule") -> None:
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
                tag=self.BAND,
                default_value=list(state["band"]),
                width=-1,
                callback=self._set_band,
            )
            dpg.add_text("Band duration, s")
            dpg.add_input_float(
                tag=self.DURATION,
                default_value=state["duration"],
                min_value=0.1,
                min_clamped=True,
                step=0,
                width=-1,
                callback=self._set_duration,
            )
            dpg.add_text("Smoothing, octaves")
            dpg.add_input_float(
                tag=self.SMOOTHING,
                default_value=state["smoothing_octaves"],
                min_value=0.01,
                min_clamped=True,
                step=0,
                width=-1,
                callback=self._set_smoothing,
            )
            dpg.add_text("Points")
            dpg.add_input_int(
                tag=self.POINTS,
                default_value=state["points"],
                min_value=2,
                min_clamped=True,
                step=0,
                width=-1,
                callback=self._set_points,
            )
            dpg.add_text("Calculated delay, m")
            dpg.add_text("--", tag=self.DELAY_TEXT, wrap=-1)
            dpg.add_text("Delay correction, m")
            dpg.add_input_float(
                tag=self.DELAY_CORRECTION,
                default_value=state["delay_correction_meters"] or 0.0,
                step=0,
                width=-1,
                callback=self._set_delay_correction,
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
                labels=("A", "B"),
            )
            with dpg.plot(  # pyright: ignore[reportGeneralTypeIssues]
                tag=self.LEVEL_PLOT,
                width=-1,
                height=-1,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag=self.LEVEL_X_AXIS)
                dpg.add_plot_axis(dpg.mvYAxis, tag=self.LEVEL_Y_AXIS)
                dpg.add_line_series(
                    [],
                    [],
                    label="A",
                    tag=self.LEVEL_SERIES_A,
                    parent=self.LEVEL_Y_AXIS,
                )
                dpg.add_line_series(
                    [],
                    [],
                    label="B",
                    tag=self.LEVEL_SERIES_B,
                    parent=self.LEVEL_Y_AXIS,
                )
        dpg.set_axis_limits(self.LEVEL_Y_AXIS, 0.0, 1.0)
        self.update_result(
            state.get("estimated_delay_meters"),
            state.get("delay_correction_meters"),
        )
        levels = np.asarray(state["level_values"], dtype=np.float64)
        current = (
            (float(levels[-1, 0]), float(levels[-1, 1]))
            if levels.ndim == 2 and levels.shape[0] and levels.shape[1] >= 2
            else (0.0, 0.0)
        )
        self.update_levels(
            np.asarray(state["level_time"], dtype=np.float64),
            levels,
            current,
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
        current: tuple[float, float],
        *,
        duration: float,
    ) -> None:
        if self.level_meter is not None:
            self.level_meter.set_levels(*current)
        channel_a = levels[:, 0] if levels.ndim == 2 and levels.shape[1] >= 1 else []
        channel_b = levels[:, 1] if levels.ndim == 2 and levels.shape[1] >= 2 else []
        dpg.set_value(
            self.LEVEL_SERIES_A,
            [times[: len(channel_a)].tolist(), np.asarray(channel_a).tolist()],
        )
        dpg.set_value(
            self.LEVEL_SERIES_B,
            [times[: len(channel_b)].tolist(), np.asarray(channel_b).tolist()],
        )
        maximum_time = max(duration, float(times[-1]) if times.size else 0.0)
        dpg.set_axis_limits(self.LEVEL_X_AXIS, 0.0, maximum_time)

    def update_result(
        self,
        delay_meters: Any,
        correction_meters: Any,
    ) -> None:
        if delay_meters is None:
            text = "--"
        else:
            text = f"{float(delay_meters):.3f} m"
        dpg.set_value(self.DELAY_TEXT, text)
        dpg.set_value(
            self.DELAY_CORRECTION,
            float(correction_meters) if correction_meters is not None else 0.0,
        )

    def _set_band(self, sender, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_duration(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("duration", value))

    def _set_smoothing(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("smoothing_octaves", value))

    def _set_points(self, sender, value: int, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("points", value))

    def _set_delay_correction(self, sender, value: float, user_data=None) -> None:
        correction = self.module.set_setting("delay_correction_meters", value)
        dpg.set_value(sender, correction)
