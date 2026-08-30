from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg
import numpy as np

from audioanalysis import SmoothingWindow

if TYPE_CHECKING:
    from spectrum_app.modules.spectrum.module import SpectrumModule


class SpectrumView:
    ROOT = "module::spectrum::controls"
    MULTIPLE = "module::spectrum::multiple"
    MULTIPLE_CONTROLS = "module::spectrum::multiple_controls"
    COUNT = "module::spectrum::multiple_count"
    AUTO = "module::spectrum::multiple_auto"
    WEIGHTING_GROUP = "module::spectrum::weighting_group"
    REPEAT_DIALOG = "module::spectrum::repeat_dialog"
    REPEAT_TEXT = "module::spectrum::repeat_text"
    REPEAT_DIALOG_WIDTH = 560
    REPEAT_DIALOG_HEIGHT = 160
    LEVEL_PLOT = "module::spectrum::level_plot"
    LEVEL_X_AXIS = "module::spectrum::level_x_axis"
    LEVEL_Y_AXIS = "module::spectrum::level_y_axis"
    LEVEL_SERIES_1 = "module::spectrum::level_series_1"
    LEVEL_SERIES_2 = "module::spectrum::level_series_2"

    def __init__(self, module: "SpectrumModule") -> None:
        self.module = module

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
            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Generator",
                default_open=True,
            ):
                dpg.add_text("Band, Hz")
                dpg.add_input_intx(
                    size=2,
                    default_value=list(state["band"]),
                    width=-1,
                    callback=self._set_band,
                )
                dpg.add_text("Length, s")
                dpg.add_input_float(
                    default_value=state["duration"],
                    width=-1,
                    step=0,
                    callback=self._set_duration,
                )
                dpg.add_checkbox(
                    label="Multiple",
                    tag=self.MULTIPLE,
                    default_value=state["multiple"],
                    callback=self._set_multiple,
                )
                with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                    tag=self.MULTIPLE_CONTROLS,
                    show=state["multiple"],
                ):
                    dpg.add_text("Count")
                    dpg.add_input_int(
                        tag=self.COUNT,
                        default_value=state["count"],
                        min_value=2,
                        max_value=100,
                        min_clamped=True,
                        max_clamped=True,
                        step=0,
                        width=-1,
                        callback=self._set_count,
                    )
                    dpg.add_checkbox(
                        label="Auto",
                        tag=self.AUTO,
                        default_value=state["auto"],
                        callback=self._set_auto,
                    )

            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Analyzer",
                default_open=True,
            ):
                dpg.add_text("Reference")
                dpg.add_combo(
                    ["none", "channel b", "generator"],
                    default_value=state["reference"],
                    width=-1,
                    callback=self._set_reference,
                )
                with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                    tag=self.WEIGHTING_GROUP,
                    show=state["reference"] == "none",
                ):
                    dpg.add_text("Weighting")
                    dpg.add_combo(
                        ["none", "pink"],
                        default_value=state["weighting"],
                        width=-1,
                        callback=self._set_weighting,
                    )
            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Smoothing",
                default_open=True,
            ):
                dpg.add_text("Window width, octaves")
                dpg.add_input_float(
                    default_value=state["window_width"],
                    width=-1,
                    step=0.1,
                    callback=self._set_window_width,
                )
                dpg.add_text("Points number")
                dpg.add_input_int(
                    default_value=state["points"],
                    width=-1,
                    step=0,
                    callback=self._set_points,
                )
                dpg.add_text("Window function")
                dpg.add_combo(
                    SmoothingWindow.list(),
                    default_value=state["window"],
                    width=-1,
                    callback=self._set_window,
                )

        with dpg.plot(  # pyright: ignore[reportGeneralTypeIssues]
            parent=bottom_parent,
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
                tag=self.LEVEL_SERIES_1,
                label="A",
                parent=self.LEVEL_Y_AXIS,
            )
            dpg.add_line_series(
                [],
                [],
                tag=self.LEVEL_SERIES_2,
                label="B",
                parent=self.LEVEL_Y_AXIS,
            )
        dpg.set_axis_limits(self.LEVEL_Y_AXIS, 0.0, 1.0)
        self.update_levels(
            np.asarray(state["level_time"], dtype=np.float64),
            np.asarray(state["level_values"], dtype=np.float64),
            duration=float(state["duration"]),
        )

        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Spectrum multiple measurement",
            tag=self.REPEAT_DIALOG,
            width=self.REPEAT_DIALOG_WIDTH,
            height=self.REPEAT_DIALOG_HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.module.break_multiple_measurement,
        ):
            dpg.add_text(
                "Move the microphone before the next measurement",
                tag=self.REPEAT_TEXT,
                wrap=self.REPEAT_DIALOG_WIDTH - 30,
            )
            dpg.add_button(
                label="Continue",
                width=-1,
                callback=self.module.continue_multiple_measurement,
            )
            dpg.add_button(
                label="Break",
                width=-1,
                callback=self.module.break_multiple_measurement,
            )

    def destroy(self) -> None:
        for item in (self.ROOT, self.LEVEL_PLOT, self.REPEAT_DIALOG):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)

    def set_enabled(self, enabled: bool) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.configure_item(self.ROOT, enabled=enabled)

    def show_repeat_dialog(self, completed: int, total: int) -> None:
        dpg.set_value(
            self.REPEAT_TEXT,
            "Move the microphone before the next measurement\n"
            f"Completed: {completed} of {total}",
        )
        main_position = dpg.get_item_pos(self.module.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.module.app.main_window.tag)
        if main_size == [100, 100]:
            main_size = [
                dpg.get_viewport_client_width(),
                dpg.get_viewport_client_height(),
            ]
        position = [
            main_position[0] + (main_size[0] - self.REPEAT_DIALOG_WIDTH) / 2,
            main_position[1] + (main_size[1] - self.REPEAT_DIALOG_HEIGHT) / 2,
        ]
        dpg.set_item_pos(self.REPEAT_DIALOG, position)
        dpg.configure_item(self.REPEAT_DIALOG, show=True)

    def hide_repeat_dialog(self) -> None:
        if dpg.does_item_exist(self.REPEAT_DIALOG):
            dpg.configure_item(self.REPEAT_DIALOG, show=False)

    def update_levels(
        self,
        times: np.ndarray,
        levels: np.ndarray,
        *,
        duration: float,
    ) -> None:
        channel_1 = (
            levels[:, 0]
            if levels.ndim == 2 and levels.shape[1] >= 1
            else []
        )
        channel_2 = (
            levels[:, 1]
            if levels.ndim == 2 and levels.shape[1] >= 2
            else []
        )
        dpg.set_value(
            self.LEVEL_SERIES_1,
            [times[: len(channel_1)].tolist(), np.asarray(channel_1).tolist()],
        )
        dpg.set_value(
            self.LEVEL_SERIES_2,
            [times[: len(channel_2)].tolist(), np.asarray(channel_2).tolist()],
        )
        maximum_time = max(
            duration + 1.2,
            float(times[-1]) if times.size else 0.0,
        )
        dpg.set_axis_limits(self.LEVEL_X_AXIS, 0.0, maximum_time)

    def _set_band(self, sender: int | str, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_duration(self, sender: int | str, value: float, user_data=None) -> None:
        duration = self.module.set_setting("duration", value)
        dpg.set_value(sender, duration)

    def _set_multiple(self, sender, value: bool, user_data=None) -> None:
        multiple = self.module.set_setting("multiple", value)
        dpg.set_value(sender, multiple)
        dpg.configure_item(self.MULTIPLE_CONTROLS, show=multiple)

    def _set_count(self, sender, value: int, user_data=None) -> None:
        count = self.module.set_setting("count", value)
        dpg.set_value(sender, count)

    def _set_auto(self, sender, value: bool, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("auto", value))

    def _set_reference(self, sender, value: str, user_data=None) -> None:
        reference = self.module.set_setting("reference", value)
        dpg.set_value(sender, reference)
        dpg.configure_item(self.WEIGHTING_GROUP, show=reference == "none")

    def _set_weighting(self, sender, value: str, user_data=None) -> None:
        self.module.set_setting("weighting", value)

    def _set_window_width(
        self, sender: int | str, value: float, user_data=None
    ) -> None:
        width = self.module.set_setting("window_width", value)
        dpg.set_value(sender, width)

    def _set_points(self, sender: int | str, value: int, user_data=None) -> None:
        points = self.module.set_setting("points", value)
        dpg.set_value(sender, points)

    def _set_window(self, sender, value: str, user_data=None) -> None:
        self.module.set_setting("window", value)
