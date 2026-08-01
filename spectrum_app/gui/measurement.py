from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from spectrum_app.core.model import Measurement

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class MeasurementPanel:
    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.run_button = "app::measurement::run"
        self.module_combo = "app::measurement::module"
        self.module_gui_host = "app::module_gui_host"
        self.green_button_theme = "app::measurement::green_button_theme"
        self.red_button_theme = "app::measurement::red_button_theme"
        self._shown_measurement_id: str | None = None
        self._shown_measuring: bool | None = None

    def build(self) -> None:
        self._build_themes()
        dpg.add_button(
            label="OFF",
            tag=self.run_button,
            width=-1,
            height=50,
            callback=self._toggle_measurement,
        )
        dpg.add_combo(
            [self.app.DEFAULT_MODULE_ID],
            tag=self.module_combo,
            width=-1,
            callback=self._set_module,
        )
        with dpg.child_window(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.module_gui_host,
            height=-1,
            border=False,
        ):
            pass
        self.update(force=True)

    def update(self, force: bool = False) -> None:
        measurement = self._active_measurement()
        measurement_id = measurement.id if measurement else None
        measuring = self.app.app_state.measuring

        if force or measurement_id != self._shown_measurement_id:
            dpg.set_value(
                self.module_combo,
                measurement.module_id if measurement else "",
            )
            self._shown_measurement_id = measurement_id

        if force or measuring != self._shown_measuring:
            dpg.set_item_label(self.run_button, "ON" if measuring else "OFF")
            dpg.bind_item_theme(
                self.run_button,
                self.red_button_theme if measuring else self.green_button_theme,
            )
            self._shown_measuring = measuring

        dpg.configure_item(
            self.run_button,
            enabled=measurement is not None,
        )
        dpg.configure_item(
            self.module_combo,
            enabled=measurement is not None and not measuring,
        )

    def _build_themes(self) -> None:
        with dpg.theme(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.green_button_theme,
        ):
            with dpg.theme_component(  # pyright: ignore[reportGeneralTypeIssues]
                dpg.mvButton,
            ):
                self._add_button_colors((0, 200, 0), (0, 220, 0), (0, 160, 0))

        with dpg.theme(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.red_button_theme,
        ):
            with dpg.theme_component(  # pyright: ignore[reportGeneralTypeIssues]
                dpg.mvButton,
            ):
                self._add_button_colors((200, 0, 0), (220, 0, 0), (160, 0, 0))

    @staticmethod
    def _add_button_colors(
        normal: tuple[int, int, int],
        hovered: tuple[int, int, int],
        active: tuple[int, int, int],
    ) -> None:
        dpg.add_theme_color(
            dpg.mvThemeCol_Button,
            (*normal, 255),
            category=dpg.mvThemeCat_Core,
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonHovered,
            (*hovered, 255),
            category=dpg.mvThemeCat_Core,
        )
        dpg.add_theme_color(
            dpg.mvThemeCol_ButtonActive,
            (*active, 255),
            category=dpg.mvThemeCat_Core,
        )

    def _toggle_measurement(self, sender=None, app_data=None, user_data=None) -> None:
        self.app.app_state.measuring = not self.app.app_state.measuring
        self.update()

    def _set_module(self, sender: int | str, module_id: str, user_data=None) -> None:
        measurement = self._active_measurement()
        if measurement is not None:
            measurement.module_id = module_id

    def _active_measurement(self) -> Measurement | None:
        active_id = self.app.app_state.active_measurement_id
        for measurement in self.app.app_state.measurements:
            if measurement.id == active_id:
                return measurement
        return None
