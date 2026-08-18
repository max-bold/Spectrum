from pathlib import Path
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.core.project import PROJECT_EXTENSION, ProjectError

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class ProjectDialogs:
    WIDTH = 700
    HEIGHT = 400

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.open_dialog = "app::project::open_dialog"
        self.save_dialog = "app::project::save_dialog"

    def build(self, file_menu: int | str) -> None:
        dpg.add_menu_item(
            label="Open",
            parent=file_menu,
            callback=self.show_open_dialog,
        )
        dpg.add_menu_item(
            label="Save",
            parent=file_menu,
            callback=self.save,
        )
        dpg.add_menu_item(
            label="Save As...",
            parent=file_menu,
            callback=self.show_save_dialog,
        )

        dpg.add_file_dialog(
            tag=self.open_dialog,
            show=False,
            modal=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            callback=self.open,
        )
        dpg.add_file_extension(PROJECT_EXTENSION, parent=self.open_dialog)
        dpg.add_file_dialog(
            tag=self.save_dialog,
            show=False,
            modal=True,
            width=self.WIDTH,
            height=self.HEIGHT,
            default_filename=f"project{PROJECT_EXTENSION}",
            callback=self.save_as,
        )
        dpg.add_file_extension(PROJECT_EXTENSION, parent=self.save_dialog)

    def show_open_dialog(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.show_item(self.open_dialog)

    def show_save_dialog(self, sender=None, app_data=None, user_data=None) -> None:
        project_path = self.app.app_state.project_path
        if project_path is not None:
            dpg.configure_item(
                self.save_dialog,
                default_path=str(project_path.parent),
                default_filename=project_path.name,
            )
        dpg.show_item(self.save_dialog)

    def save(self, sender=None, app_data=None, user_data=None) -> None:
        if self.app.app_state.project_path is None:
            self.show_save_dialog()
            return
        self._save_to(self.app.app_state.project_path)

    def save_as(self, sender: int | str, app_data: dict[str, Any], user_data=None) -> None:
        path = self._path_from_dialog(app_data)
        if path is not None:
            self._hide_dialog(sender)
            self._save_to(path)

    def open(self, sender: int | str, app_data: dict[str, Any], user_data=None) -> None:
        path = self._path_from_dialog(app_data)
        if path is None:
            return
        self._hide_dialog(sender)
        try:
            self.app.load_project(path)
        except ProjectError as error:
            self.app.main_window.set_status_text(str(error))
            return
        self.app.main_window.set_status_text(f"Opened: {path}")

    def _save_to(self, path: Path) -> None:
        try:
            saved_path = self.app.save_project(path)
        except ProjectError as error:
            self.app.main_window.set_status_text(str(error))
            return
        self.app.main_window.set_status_text(f"Saved: {saved_path}")

    @staticmethod
    def _path_from_dialog(app_data: dict[str, Any]) -> Path | None:
        value = app_data.get("file_path_name")
        return Path(value) if value else None

    @staticmethod
    def _hide_dialog(dialog: int | str) -> None:
        if dpg.does_item_exist(dialog):
            dpg.hide_item(dialog)
