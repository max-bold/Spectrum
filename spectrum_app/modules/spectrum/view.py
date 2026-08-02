from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from audioanalysis import SmoothingWindow

if TYPE_CHECKING:
    from spectrum_app.modules.spectrum.module import SpectrumModule


class SpectrumView:
    ROOT = "module::spectrum::controls"

    def __init__(self, module: "SpectrumModule") -> None:
        self.module = module

    def build(self, parent: int | str, state: dict[str, Any]) -> None:
        with dpg.group(  # pyright: ignore[reportGeneralTypeIssues]
            parent=parent,
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

    def destroy(self) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.delete_item(self.ROOT)

    def set_enabled(self, enabled: bool) -> None:
        if dpg.does_item_exist(self.ROOT):
            dpg.configure_item(self.ROOT, enabled=enabled)

    def _set_band(self, sender: int | str, value: list[int], user_data=None) -> None:
        band = self.module.set_setting("band", (value[0], value[1]))
        dpg.set_value(sender, [*band, 0, 0])

    def _set_duration(self, sender: int | str, value: float, user_data=None) -> None:
        duration = self.module.set_setting("duration", value)
        dpg.set_value(sender, duration)

    def _set_reference(self, sender, value: str, user_data=None) -> None:
        self.module.set_setting("reference", value)

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
