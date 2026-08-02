from pathlib import Path
from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

from spectrum_app.core.model import Measurement

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class AppStatePanel:
    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.measurements_table = "app::measurements_table"
        self.add_measurement_button = "app::add_measurement"
        self.texture_registry = "app::app_state::texture_registry"
        self.delete_icon = "app::app_state::delete_icon"
        self._built = False
        self._shown_measurement_ids: list[str] = []

    def build(self) -> None:
        self._load_icons()
        self._shown_measurement_ids.clear()

        with dpg.table(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.measurements_table,
            header_row=False,
        ):
            dpg.add_table_column(width_fixed=True)
            dpg.add_table_column(width_fixed=True)
            dpg.add_table_column()
            dpg.add_table_column(width_fixed=True)

            for measurement in self.app.app_state.measurements:
                self._add_measurement_row(measurement)

        dpg.add_button(
            label="+",
            tag=self.add_measurement_button,
            width=-1,
            callback=self._create_measurement,
        )
        self._built = True

    def rebuild(self) -> None:
        if not self._built:
            return
        for measurement_id in self._shown_measurement_ids:
            dpg.delete_item(self._row_tag(measurement_id))
        self._shown_measurement_ids.clear()
        for measurement in self.app.app_state.measurements:
            self._add_measurement_row(measurement)
        self._sync_active_measurement_checkboxes()

    def _add_measurement_row(self, measurement: Measurement) -> None:
        with dpg.table_row(  # pyright: ignore[reportGeneralTypeIssues]
            parent=self.measurements_table,
            tag=self._row_tag(measurement.id),
        ):
            dpg.add_checkbox(
                tag=self._active_tag(measurement.id),
                default_value=measurement.id
                == self.app.app_state.active_measurement_id,
                callback=self._set_active_measurement,
                user_data=measurement.id,
            )
            dpg.add_checkbox(
                tag=self._visible_tag(measurement.id),
                default_value=self._measurement_is_visible(measurement),
                callback=self._set_measurement_visible,
                user_data=measurement.id,
            )
            dpg.add_input_text(
                default_value=measurement.name,
                width=-1,
                callback=self._set_measurement_name,
                user_data=measurement.id,
            )
            dpg.add_image_button(
                self.delete_icon,
                tag=self._delete_tag(measurement.id),
                width=16,
                height=16,
                background_color=(0, 0, 0, 0),
                callback=self._delete_measurement,
                user_data=measurement.id,
            )
        self._shown_measurement_ids.append(measurement.id)

    def _load_icons(self) -> None:
        if dpg.does_item_exist(self.delete_icon):
            return

        icon_path = Path(__file__).parent / "icons" / "icons8-delete-16-white.png"
        width, height, _, data = dpg.load_image(str(icon_path))
        with dpg.texture_registry(  # pyright: ignore[reportGeneralTypeIssues]
            tag=self.texture_registry,
        ):
            dpg.add_static_texture(
                width,
                height,
                data,
                tag=self.delete_icon,
            )

    def _create_measurement(self, sender=None, app_data=None, user_data=None) -> None:
        measurement = self.app.create_measurement()
        self._add_measurement_row(measurement)
        self._sync_active_measurement_checkboxes()

    def _set_active_measurement(
        self, sender: int | str, checked: bool, measurement_id: str
    ) -> None:
        if checked:
            self.app.app_state.active_measurement_id = measurement_id
        self._sync_active_measurement_checkboxes()

    def _delete_measurement(
        self, sender: int | str, app_data, measurement_id: str
    ) -> None:
        self.app.delete_measurement(measurement_id)
        dpg.delete_item(self._row_tag(measurement_id))
        self._shown_measurement_ids.remove(measurement_id)
        self._sync_active_measurement_checkboxes()

    def _set_measurement_visible(
        self, sender: int | str, visible: bool, measurement_id: str
    ) -> None:
        measurement = self._find_measurement(measurement_id)
        graph_ids = {graph.id for graph in measurement.graphs}

        if visible:
            for graph_id in graph_ids:
                if graph_id not in self.app.app_state.visible_graph_ids:
                    self.app.app_state.visible_graph_ids.append(graph_id)
        else:
            self.app.app_state.visible_graph_ids = [
                graph_id
                for graph_id in self.app.app_state.visible_graph_ids
                if graph_id not in graph_ids
            ]

        self.app.app_state.graph_data_changed = True
        dpg.set_value(sender, self._measurement_is_visible(measurement))

    def _set_measurement_name(
        self, sender: int | str, name: str, measurement_id: str
    ) -> None:
        self._find_measurement(measurement_id).name = name
        self.app.app_state.graph_data_changed = True

    def _sync_active_measurement_checkboxes(self) -> None:
        active_id = self.app.app_state.active_measurement_id
        for measurement in self.app.app_state.measurements:
            dpg.set_value(
                self._active_tag(measurement.id),
                measurement.id == active_id,
            )

    def _measurement_is_visible(self, measurement: Measurement) -> bool:
        graph_ids = [graph.id for graph in measurement.graphs]
        return bool(graph_ids) and all(
            graph_id in self.app.app_state.visible_graph_ids
            for graph_id in graph_ids
        )

    def _find_measurement(self, measurement_id: str) -> Measurement:
        for measurement in self.app.app_state.measurements:
            if measurement.id == measurement_id:
                return measurement
        raise ValueError(f"Unknown measurement: {measurement_id}")

    @staticmethod
    def _active_tag(measurement_id: str) -> str:
        return f"app::measurement::{measurement_id}::active"

    @staticmethod
    def _visible_tag(measurement_id: str) -> str:
        return f"app::measurement::{measurement_id}::visible"

    @staticmethod
    def _row_tag(measurement_id: str) -> str:
        return f"app::measurement::{measurement_id}::row"

    @staticmethod
    def _delete_tag(measurement_id: str) -> str:
        return f"app::measurement::{measurement_id}::delete"
