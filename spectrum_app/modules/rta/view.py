from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.core.model import PlotType
from spectrum_app.gui.controls import LevelMeter, add_level_meter
from spectrum_app.modules.rta.types import (
    FILTERED_IIR_GENERATOR,
    RTANoiseGeneratorType,
)

if TYPE_CHECKING:
    from spectrum_app.modules.rta.module import RTAModule


class RTAView:
    ROOT = "module::rta::controls"
    BOTTOM = "module::rta::bottom"
    METER = "module::rta::level_meter"
    NOISE = "module::rta::noise"
    BAND = "module::rta::band"
    LEVEL = "module::rta::level"
    LEVEL_GROUP = "module::rta::level_group"
    WINDOW_WIDTH = "module::rta::window_width"
    WINDOW_HOP = "module::rta::window_hop"
    POINTS = "module::rta::points"
    POINTS_HANDLERS = "module::rta::points::handlers"
    SMOOTHING_GROUP = "module::rta::smoothing_group"
    SMOOTHING = "module::rta::smoothing"
    GENERATOR_GROUP = "module::rta::generator_group"
    FFT_GROUP = "module::rta::fft_group"

    def __init__(self, module: "RTAModule") -> None:
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
            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Generator",
                default_open=True,
            ):
                with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                    tag=self.GENERATOR_GROUP,
                ):
                    dpg.add_checkbox(
                        label="Noise",
                        tag=self.NOISE,
                        default_value=state["noise"],
                        callback=self._set_noise,
                    )
                    dpg.add_text("Freq band, Hz")
                    dpg.add_input_intx(
                        size=2,
                        tag=self.BAND,
                        default_value=list(state["band"]),
                        width=-1,
                        callback=self._set_band,
                    )
                    with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                        tag=self.LEVEL_GROUP,
                    ):
                        dpg.add_text("Level, dB")
                        dpg.add_slider_float(
                            tag=self.LEVEL,
                            default_value=state["level_db"],
                            min_value=-10.0,
                            max_value=10.0,
                            clamped=True,
                            format="%.1f dB",
                            width=-1,
                            callback=self._set_level,
                        )

            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="FFT",
                default_open=True,
            ):
                with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                    tag=self.FFT_GROUP,
                ):
                    dpg.add_text("Window width, s")
                    dpg.add_input_float(
                        tag=self.WINDOW_WIDTH,
                        default_value=state["window_width"],
                        min_value=0.01,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        callback=self._set_window_width,
                    )
                    dpg.add_text("Hop size, s")
                    dpg.add_input_float(
                        tag=self.WINDOW_HOP,
                        default_value=state["window_hop"],
                        min_value=0.001,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        callback=self._set_window_hop,
                    )

            with dpg.collapsing_header(  # pyright: ignore[reportGeneralTypeIssues]
                label="Smoothing",
                default_open=True,
            ):
                dpg.add_text("Point count")
                dpg.add_input_int(
                    tag=self.POINTS,
                    default_value=state["points"],
                    min_value=24,
                    max_value=2048,
                    min_clamped=True,
                    max_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._commit_points,
                    on_enter=True,
                )
                with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
                    tag=self.SMOOTHING_GROUP,
                ):
                    dpg.add_text("Smoothing, oct")
                    dpg.add_input_float(
                        tag=self.SMOOTHING,
                        default_value=state["smoothing_octaves"],
                        min_value=0.01,
                        min_clamped=True,
                        step=0,
                        width=-1,
                        callback=self._set_smoothing,
                    )

        with dpg.item_handler_registry(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.POINTS_HANDLERS,
        ):
            dpg.add_item_deactivated_after_edit_handler(
                callback=self._commit_points,
            )
        dpg.bind_item_handler_registry(self.POINTS, self.POINTS_HANDLERS)

        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=bottom_parent,
            tag=self.BOTTOM,
            horizontal=True,
        ):
            self.level_meter = add_level_meter(
                self.BOTTOM,
                self.METER,
                bottom_parent,
                labels=("A", "B"),
                height_offset=-20,
            )
        self.update_smoothing_visibility(int(state["points"]))
        self.update_generator_visibility(self.module.settings.generator)
        self.update_levels((0.0, 0.0))

    def destroy(self) -> None:
        for item in (self.ROOT, self.BOTTOM, self.POINTS_HANDLERS):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)
        self.level_meter = None

    def update(self) -> None:
        if self.level_meter is not None:
            self.level_meter.resize()

    def set_enabled(self, enabled: bool) -> None:
        for item in (self.GENERATOR_GROUP, self.FFT_GROUP):
            if dpg.does_item_exist(item):
                dpg.configure_item(item, enabled=enabled)

    def update_levels(self, levels: tuple[float, float]) -> None:
        if self.level_meter is not None:
            self.level_meter.set_levels(*levels)

    def update_smoothing_visibility(self, points: int) -> None:
        if dpg.does_item_exist(self.SMOOTHING_GROUP):
            dpg.configure_item(
                self.SMOOTHING_GROUP,
                show=self.module._effective_plot_type(points) == PlotType.LINE,
            )

    def update_generator_visibility(
        self,
        generator: RTANoiseGeneratorType,
    ) -> None:
        filtered = generator == FILTERED_IIR_GENERATOR
        if dpg.does_item_exist(self.LEVEL_GROUP):
            dpg.configure_item(self.LEVEL_GROUP, show=filtered)
        if not filtered and dpg.does_item_exist(self.LEVEL):
            dpg.set_value(self.LEVEL, 0.0)

    def _set_noise(self, sender, value: bool, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("noise", value))

    def _set_band(self, sender, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_level(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("level_db", value))

    def _set_window_width(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("window_width", value))

    def _set_window_hop(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(sender, self.module.set_setting("window_hop", value))

    def _commit_points(self, sender=None, app_data=None, user_data=None) -> None:
        points = self.module.set_setting("points", dpg.get_value(self.POINTS))
        dpg.set_value(self.POINTS, points)
        self.update_smoothing_visibility(points)

    def _set_smoothing(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(
            sender,
            self.module.set_setting("smoothing_octaves", value),
        )
