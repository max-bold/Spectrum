from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import re
from threading import Thread
from typing import TYPE_CHECKING, Any, Literal

import dearpygui.dearpygui as dpg

from spectrum_app.core.measurement_io import (
    MEASUREMENT_EXTENSION,
    MeasurementIOError,
    load_measurement,
    save_measurement,
)
from spectrum_app.core.model import Measurement

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


Operation = Literal["import", "export"]
Result = tuple[Operation, Path, Measurement | None, str | None]


class MeasurementDialogs:
    WIDTH = 700
    HEIGHT = 400

    def __init__(self, app: SpectrumApplication) -> None:
        self.app = app
        self.import_dialog = "app::measurement_io::import_dialog"
        self.export_dialog = "app::measurement_io::export_dialog"
        self._results: Queue[Result] = Queue()
        self._operation_active = False

    def build(self, import_menu: int | str, export_menu: int | str) -> None:
        dpg.add_menu_item(
            label="Measurement",
            parent=import_menu,
            callback=self.show_import_dialog,
        )
        dpg.add_menu_item(
            label="Measurement",
            parent=export_menu,
            callback=self.show_export_dialog,
        )
        dpg.add_file_dialog(
            tag=self.import_dialog,
            show=False,
            modal=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            callback=self.import_measurement,
        )
        dpg.add_file_extension(MEASUREMENT_EXTENSION, parent=self.import_dialog)
        dpg.add_file_dialog(
            tag=self.export_dialog,
            show=False,
            modal=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            default_filename=f"measurement{MEASUREMENT_EXTENSION}",
            callback=self.export_measurement,
        )
        dpg.add_file_extension(MEASUREMENT_EXTENSION, parent=self.export_dialog)

    def show_import_dialog(self, sender=None, app_data=None, user_data=None) -> None:
        if not self._can_start("import"):
            return
        dpg.show_item(self.import_dialog)

    def show_export_dialog(self, sender=None, app_data=None, user_data=None) -> None:
        if not self._can_start("export"):
            return
        measurement = self.app.active_measurement
        if measurement is None:
            self.app.main_window.set_status_text("No active measurement to export")
            return
        dpg.configure_item(
            self.export_dialog,
            default_filename=self._filename(measurement.name),
        )
        dpg.show_item(self.export_dialog)

    def import_measurement(
        self,
        sender: int | str,
        app_data: dict[str, Any],
        user_data=None,
    ) -> None:
        path = self._path_from_dialog(app_data)
        if path is None or not self._can_start("import"):
            return
        self._hide_dialog(sender)
        self._operation_active = True
        self.app.main_window.set_status_text(f"Importing measurement: {path}")
        Thread(
            target=self._import_worker,
            args=(path,),
            name="measurement-import",
            daemon=True,
        ).start()

    def export_measurement(
        self,
        sender: int | str,
        app_data: dict[str, Any],
        user_data=None,
    ) -> None:
        path = self._path_from_dialog(app_data)
        measurement = self.app.active_measurement
        if path is None or measurement is None or not self._can_start("export"):
            return
        self._hide_dialog(sender)
        self._operation_active = True
        self.app.main_window.set_status_text(f"Exporting measurement: {path}")
        Thread(
            target=self._export_worker,
            args=(measurement, path),
            name="measurement-export",
            daemon=True,
        ).start()

    def update(self) -> None:
        try:
            operation, path, measurement, error = self._results.get_nowait()
        except Empty:
            return
        self._operation_active = False
        if error is not None:
            self.app.main_window.set_status_text(
                f"Measurement {operation} error: {error}"
            )
            return
        if operation == "import":
            if measurement is None:
                self.app.main_window.set_status_text(
                    "Measurement import error: empty measurement"
                )
                return
            try:
                self.app.add_imported_measurement(measurement)
            except MeasurementIOError as import_error:
                self.app.main_window.set_status_text(str(import_error))
                return
            self.app.main_window.set_status_text(f"Measurement imported: {path}")
            return
        self.app.main_window.set_status_text(f"Measurement exported: {path}")

    def _import_worker(self, path: Path) -> None:
        try:
            measurement = load_measurement(path)
            error = None
        except MeasurementIOError as exception:
            measurement = None
            error = str(exception)
        self._results.put(("import", path, measurement, error))

    def _export_worker(self, measurement: Measurement, path: Path) -> None:
        try:
            saved_path = save_measurement(measurement, path)
            error = None
        except MeasurementIOError as exception:
            saved_path = path
            error = str(exception)
        self._results.put(("export", saved_path, None, error))

    def _can_start(self, operation: Operation) -> bool:
        if self._operation_active:
            self.app.main_window.set_status_text(
                "Another measurement file operation is already running"
            )
            return False
        if self.app.app_state.measuring:
            self.app.main_window.set_status_text(
                f"Stop the current measurement before {operation}"
            )
            return False
        return True

    @staticmethod
    def _filename(name: str) -> str:
        stem = re.sub(r'[<>:"/\\|?*]+', "_", name).strip(" .")
        stem = stem or "measurement"
        if stem.lower().endswith(MEASUREMENT_EXTENSION):
            return stem
        return f"{stem}{MEASUREMENT_EXTENSION}"

    @staticmethod
    def _path_from_dialog(app_data: dict[str, Any]) -> Path | None:
        value = app_data.get("file_path_name")
        return Path(value) if value else None

    @staticmethod
    def _hide_dialog(dialog: int | str) -> None:
        if dpg.does_item_exist(dialog):
            dpg.hide_item(dialog)
