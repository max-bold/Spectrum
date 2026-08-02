from typing import TYPE_CHECKING, cast

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AxisScale, PhaseUnit

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class SettingsWindow:
    WIDTH = 667
    HEIGHT = 400

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.tag = "app::settings_window"
        self.frequency_range = "app::settings::frequency_range"
        self.impedance_scale = "app::settings::impedance_scale"
        self.thd_scale = "app::settings::thd_scale"
        self.phase_unit = "app::settings::phase_unit"

    def build(self) -> None:
        settings = self.app.settings
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Settings",
            tag=self.tag,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            with dpg.tab_bar():  # pyright: ignore[reportGeneralTypeIssues]
                with dpg.tab(  # pyright: ignore[reportGeneralTypeIssues]
                    label="Plot",
                ):
                    dpg.add_text("X axis range, Hz")
                    dpg.add_input_intx(
                        tag=self.frequency_range,
                        size=2,
                        default_value=[
                            int(settings.frequency_range[0]),
                            int(settings.frequency_range[1]),
                        ],
                        callback=self._set_frequency_range,
                    )

                    dpg.add_separator()
                    dpg.add_text("Impedance scale")
                    dpg.add_radio_button(
                        ("linear", "log"),
                        tag=self.impedance_scale,
                        default_value=settings.impedance_scale,
                        horizontal=True,
                        callback=self._set_impedance_scale,
                    )

                    dpg.add_separator()
                    dpg.add_text("THD scale")
                    dpg.add_radio_button(
                        ("linear", "log"),
                        tag=self.thd_scale,
                        default_value=settings.thd_scale,
                        horizontal=True,
                        callback=self._set_thd_scale,
                    )

                    dpg.add_separator()
                    dpg.add_text("Phase display")
                    dpg.add_radio_button(
                        ("deg", "deg/dec"),
                        tag=self.phase_unit,
                        default_value=settings.phase_unit,
                        horizontal=True,
                        callback=self._set_phase_unit,
                    )

    def show(self, sender=None, app_data=None, user_data=None) -> None:
        self._sync_controls()
        main_position = dpg.get_item_pos(self.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.app.main_window.tag)
        if main_size == [100, 100]:
            main_size = [
                dpg.get_viewport_client_width(),
                dpg.get_viewport_client_height(),
            ]
        position = [
            main_position[0] + (main_size[0] - self.WIDTH) / 2,
            main_position[1] + (main_size[1] - self.HEIGHT) / 2,
        ]
        dpg.set_item_pos(self.tag, position)
        dpg.configure_item(self.tag, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.tag, show=False)

    def _sync_controls(self) -> None:
        settings = self.app.settings
        dpg.set_value(
            self.frequency_range,
            [
                int(settings.frequency_range[0]),
                int(settings.frequency_range[1]),
                0,
                0,
            ],
        )
        dpg.set_value(self.impedance_scale, settings.impedance_scale)
        dpg.set_value(self.thd_scale, settings.thd_scale)
        dpg.set_value(self.phase_unit, settings.phase_unit)

    def _set_frequency_range(
        self, sender: int | str, value: list[int], user_data=None
    ) -> None:
        try:
            self.app.settings.frequency_range = (float(value[0]), float(value[1]))
        except ValueError:
            low, high = self.app.settings.frequency_range
            dpg.set_value(sender, [int(low), int(high), 0, 0])

    def _set_impedance_scale(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.impedance_scale = cast(AxisScale, value)

    def _set_thd_scale(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.thd_scale = cast(AxisScale, value)

    def _set_phase_unit(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.phase_unit = cast(PhaseUnit, value)
