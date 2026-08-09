from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from spectrum_app.core.model import Measurement
from spectrum_app.modules.base import BaseModule

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class MeasurementPanel:
    CONFIRM_WIDTH = 440
    CONFIRM_HEIGHT = 150

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.run_button = "app::measurement::run"
        self.module_combo = "app::measurement::module"
        self.module_gui_host = "app::module_gui_host"
        self.green_button_theme = "app::measurement::green_button_theme"
        self.red_button_theme = "app::measurement::red_button_theme"
        self.module_change_dialog = "app::measurement::module_change_dialog"
        self.module_change_text = "app::measurement::module_change_text"
        self._shown_measurement_id: str | None = None
        self._shown_module_id: str | None = None
        self._shown_measuring: bool | None = None
        self._shown_button_label: str | None = None
        self._module_ids_by_name = {
            module.name: module.id for module in self.app.module_manager.modules
        }
        self._modules_ready = False
        self._active_module: BaseModule | None = None
        self._active_module_measurement_id: str | None = None
        self._pending_module_change: tuple[str, str] | None = None

    def build(self) -> None:
        self._build_themes()
        dpg.add_button(
            label="MEASURE",
            tag=self.run_button,
            width=-1,
            height=50,
            callback=self._toggle_measurement,
        )
        dpg.add_combo(
            list(self._module_ids_by_name),
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

    def build_dialogs(self) -> None:
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Change measurement module",
            tag=self.module_change_dialog,
            width=self.CONFIRM_WIDTH,
            height=self.CONFIRM_HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self._cancel_module_change,
        ):
            dpg.add_text(
                "Changing the module will permanently delete the existing "
                "measurement data. Continue?",
                tag=self.module_change_text,
                wrap=self.CONFIRM_WIDTH - 30,
            )
            with dpg.group(horizontal=True):  # pyright: ignore[reportGeneralTypeIssues]
                dpg.add_button(
                    label="Change module",
                    callback=self._confirm_module_change,
                    width=200,
                )
                dpg.add_button(
                    label="Cancel",
                    callback=self._cancel_module_change,
                    width=200,
                )

    def modules_initialized(self) -> None:
        self._modules_ready = True
        self.update(force=True)

    def shutdown(self) -> None:
        self._deactivate_module()
        self._modules_ready = False

    def deactivate(self) -> None:
        self._deactivate_module()

    def update(self, force: bool = False) -> None:
        measurement = self._active_measurement()
        if self._modules_ready:
            self._sync_active_module(measurement)
            if self._active_module is not None:
                self._active_module.update()
        measurement_id = measurement.id if measurement else None
        measuring = self.app.app_state.measuring

        module_id = measurement.module_id if measurement else None
        if (
            force
            or measurement_id != self._shown_measurement_id
            or module_id != self._shown_module_id
        ):
            dpg.set_value(
                self.module_combo,
                self._module_name(module_id) if module_id else "",
            )
            self._shown_measurement_id = measurement_id
            self._shown_module_id = module_id

        button_label = (
            "STOP"
            if measuring
            else (
                self._active_module.measurement_button_label
                if self._active_module is not None
                else "MEASURE"
            )
        )
        if force or button_label != self._shown_button_label:
            dpg.set_item_label(self.run_button, button_label)
            self._shown_button_label = button_label

        if force or measuring != self._shown_measuring:
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
        if self._active_module is None:
            return
        if self.app.app_state.measuring:
            self._active_module.stop_measurement()
        else:
            self._active_module.start_measurement()
        self.update()

    def _set_module(self, sender: int | str, module_name: str, user_data=None) -> None:
        measurement = self._active_measurement()
        if measurement is not None:
            module_id = self._module_ids_by_name[module_name]
            if measurement.module_id == module_id:
                return
            if self._has_measurement_data(measurement):
                self._pending_module_change = measurement.id, module_id
                dpg.set_value(
                    self.module_combo,
                    self._module_name(measurement.module_id),
                )
                self._center_module_change_dialog()
                dpg.configure_item(self.module_change_dialog, show=True)
                return
            self._apply_module_change(measurement, module_id)

    def _confirm_module_change(
        self, sender=None, app_data=None, user_data=None
    ) -> None:
        pending = self._pending_module_change
        self._pending_module_change = None
        dpg.configure_item(self.module_change_dialog, show=False)
        if pending is None:
            return
        measurement_id, module_id = pending
        measurement = next(
            (
                item
                for item in self.app.app_state.measurements
                if item.id == measurement_id
            ),
            None,
        )
        if measurement is not None:
            self._apply_module_change(measurement, module_id)

    def _cancel_module_change(self, sender=None, app_data=None, user_data=None) -> None:
        self._pending_module_change = None
        dpg.configure_item(self.module_change_dialog, show=False)
        measurement = self._active_measurement()
        if measurement is not None:
            dpg.set_value(
                self.module_combo,
                self._module_name(measurement.module_id),
            )

    def _apply_module_change(
        self,
        measurement: Measurement,
        module_id: str,
    ) -> None:
        removed_graph_ids = {graph.id for graph in measurement.graphs}
        self.app.app_state.visible_graph_ids = [
            graph_id
            for graph_id in self.app.app_state.visible_graph_ids
            if graph_id not in removed_graph_ids
        ]
        measurement.module_id = module_id
        measurement.module_state.clear()
        measurement.graphs.clear()
        self.app.app_state.graph_data_changed = True
        self.update(force=True)

    @staticmethod
    def _has_measurement_data(measurement: Measurement) -> bool:
        if measurement.graphs:
            return True
        data_markers = ("recording", "generator", "result", "calibration")
        return any(
            value is not None and any(marker in key for marker in data_markers)
            for key, value in measurement.module_state.items()
        )

    def _center_module_change_dialog(self) -> None:
        main_position = dpg.get_item_pos(self.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.app.main_window.tag)
        dpg.set_item_pos(
            self.module_change_dialog,
            [
                main_position[0] + (main_size[0] - self.CONFIRM_WIDTH) / 2,
                main_position[1] + (main_size[1] - self.CONFIRM_HEIGHT) / 2,
            ],
        )

    def _module_name(self, module_id: str) -> str:
        try:
            return self.app.module_manager.module(module_id).name
        except ValueError:
            return module_id

    def _sync_active_module(self, measurement: Measurement | None) -> None:
        module = (
            self.app.module_manager.module(measurement.module_id)
            if measurement is not None
            else None
        )
        if module is self._active_module and self._active_module_measurement_id == (
            measurement.id if measurement is not None else None
        ):
            return

        self._deactivate_module()
        if module is not None and measurement is not None:
            module.activate(measurement)
            self._active_module = module
            self._active_module_measurement_id = measurement.id

    def _deactivate_module(self) -> None:
        if self._active_module is None:
            return
        try:
            if self.app.app_state.measuring:
                self._active_module.stop_measurement()
            self._active_module.deactivate()
        finally:
            self._active_module = None
            self._active_module_measurement_id = None

    def _active_measurement(self) -> Measurement | None:
        active_id = self.app.app_state.active_measurement_id
        for measurement in self.app.app_state.measurements:
            if measurement.id == active_id:
                return measurement
        return None
