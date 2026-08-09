from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.gui.controls import LevelMeter, add_level_meter

if TYPE_CHECKING:
    from spectrum_app.modules.rta.module import RTAModule


class RTAView:
    ROOT = "module::rta::controls"
    BOTTOM = "module::rta::bottom"
    METER = "module::rta::level_meter"
    NOISE = "module::rta::noise"
    BAND = "module::rta::band"
    LEVEL = "module::rta::level"
    WINDOW_WIDTH = "module::rta::window_width"
    WINDOW_HOP = "module::rta::window_hop"
    POINTS = "module::rta::points"
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
                    min_value=2,
                    min_clamped=True,
                    step=0,
                    width=-1,
                    callback=self._set_points,
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
        self._set_smoothing_visibility(int(state["points"]))
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
        for item in (self.GENERATOR_GROUP, self.FFT_GROUP):
            if dpg.does_item_exist(item):
                dpg.configure_item(item, enabled=enabled)

    def update_levels(self, levels: tuple[float, float]) -> None:
        if self.level_meter is not None:
            self.level_meter.set_levels(*levels)

    def _set_smoothing_visibility(self, points: int) -> None:
        if dpg.does_item_exist(self.SMOOTHING_GROUP):
            dpg.configure_item(self.SMOOTHING_GROUP, show=points >= 100)

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

    def _set_points(self, sender, value: int, user_data=None) -> None:
        points = self.module.set_setting("points", value)
        dpg.set_value(sender, points)
        self._set_smoothing_visibility(points)

    def _set_smoothing(self, sender, value: float, user_data=None) -> None:
        dpg.set_value(
            sender,
            self.module.set_setting("smoothing_octaves", value),
        )
