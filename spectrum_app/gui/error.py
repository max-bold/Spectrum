from typing import TYPE_CHECKING

import dearpygui.dearpygui as dpg

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class ErrorDialog:
    """One reusable application-modal dialog for actionable error details."""

    WIDTH = 700
    HEIGHT = 340
    TAG = "app::error_dialog"
    MESSAGE = "app::error_dialog::message"

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app

    def build(self) -> None:
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Error",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            dpg.add_text("", tag=self.MESSAGE, wrap=self.WIDTH - 30)
            dpg.add_button(label="OK", width=-1, callback=self.hide)

    def show(self, title: str, message: str) -> None:
        if not dpg.does_item_exist(self.TAG):
            return
        dpg.set_value(self.MESSAGE, message)
        dpg.configure_item(self.TAG, label=title)
        self._center()
        dpg.configure_item(self.TAG, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        if dpg.does_item_exist(self.TAG):
            dpg.configure_item(self.TAG, show=False)

    def _center(self) -> None:
        main = self.app.main_window.tag
        main_position = dpg.get_item_pos(main)
        main_size = dpg.get_item_rect_size(main)
        if main_size == [100, 100]:
            main_size = [
                dpg.get_viewport_client_width(),
                dpg.get_viewport_client_height(),
            ]
        dpg.set_item_pos(
            self.TAG,
            [
                main_position[0] + (main_size[0] - self.WIDTH) / 2,
                main_position[1] + (main_size[1] - self.HEIGHT) / 2,
            ],
        )
