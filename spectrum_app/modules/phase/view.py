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
    RESPONSE_PLOT = "module::phase::response_plot"
    RESPONSE_X_AXIS = "module::phase::response_x_axis"
    RESPONSE_Y_AXIS = "module::phase::response_y_axis"
    RESPONSE_SERIES = "module::phase::response_series"
    DELAY_TEXT = "module::phase::delay"
    BAND = "module::phase::band"
    DURATION = "module::phase::duration"
    SMOOTHING = "module::phase::smoothing"
    POINTS = "module::phase::points"
    DELAY_FIT = "module::phase::delay_fit"
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
            dpg.add_text("Duration, s")
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
            dpg.add_text("Delay fit range, Hz")
            dpg.add_input_intx(
                size=2,
                tag=self.DELAY_FIT,
                default_value=list(state["delay_fit_band"]),
                width=-1,
                callback=self._set_delay_fit,
            )
            dpg.add_text("Calculated delay")
            dpg.add_text("--", tag=self.DELAY_TEXT, wrap=-1)
            dpg.add_text("Delay correction, m")
            dpg.add_input_float(
                tag=self.DELAY_CORRECTION,
                default_value=state["delay_correction_meters"],
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
                self.RESPONSE_PLOT,
                labels=("A", "B"),
            )
            with dpg.plot(  # pyright: ignore[reportGeneralTypeIssues]
                tag=self.RESPONSE_PLOT,
                width=-1,
                height=-1,
            ):
                dpg.add_plot_axis(dpg.mvXAxis, tag=self.RESPONSE_X_AXIS)
                dpg.add_plot_axis(dpg.mvYAxis, tag=self.RESPONSE_Y_AXIS)
                dpg.add_line_series(
                    [],
                    [],
                    tag=self.RESPONSE_SERIES,
                    parent=self.RESPONSE_Y_AXIS,
                )
        dpg.configure_item(self.RESPONSE_X_AXIS, scale=dpg.mvPlotScale_Log10)
        self.update_result(
            state.get("result_frequency"),
            state.get("result_magnitude_db"),
            state.get("estimated_delay_seconds"),
            state.get("estimated_delay_meters"),
            band=state["band"],
        )
        self.update_levels((0.0, 0.0))

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

    def update_levels(self, levels: tuple[float, float]) -> None:
        if self.level_meter is not None:
            self.level_meter.set_levels(*levels)

    def update_result(
        self,
        frequency: Any,
        magnitude_db: Any,
        delay_seconds: Any,
        delay_meters: Any,
        *,
        band: tuple[int, int],
    ) -> None:
        x = (
            np.asarray(frequency, dtype=np.float64)
            if frequency is not None
            else np.empty(0)
        )
        y = (
            np.asarray(magnitude_db, dtype=np.float64)
            if magnitude_db is not None
            else np.empty(0)
        )
        dpg.set_value(self.RESPONSE_SERIES, [x.tolist(), y.tolist()])
        dpg.set_axis_limits(self.RESPONSE_X_AXIS, float(band[0]), float(band[1]))
        if delay_seconds is None or delay_meters is None:
            text = "--"
        else:
            text = (
                f"{float(delay_seconds) * 1000.0:.3f} ms\n{float(delay_meters):.3f} m"
            )
        dpg.set_value(self.DELAY_TEXT, text)

    def _set_band(self, sender, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])
        dpg.set_value(
            self.DELAY_FIT,
            [*self.module.measurement.module_state["delay_fit_band"], 0, 0],
        )

    def _set_duration(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("duration", value))

    def _set_smoothing(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("smoothing_octaves", value))

    def _set_points(self, sender, value: int, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("points", value))

    def _set_delay_fit(self, sender, value: list[int], user_data=None) -> None:
        fit = self.module.set_setting("delay_fit_band", (value[0], value[1]))
        dpg.set_value(sender, [*fit, 0, 0])

    def _set_delay_correction(self, sender, value: float, user_data=None) -> None:
        correction = self.module.set_setting("delay_correction_meters", value)
        dpg.set_value(sender, correction)
