from typing import Any

import dearpygui.dearpygui as dpg


class DearPyGuiRuntime:
    """Owns the Dear PyGui context, viewport, callbacks, and render loop API."""

    def __init__(self, backend: Any = dpg) -> None:
        self.backend = backend
        self._context_created = False

    @property
    def running(self) -> bool:
        return self._context_created and self.backend.is_dearpygui_running()

    def create_context(self) -> None:
        if self._context_created:
            raise RuntimeError("Dear PyGui context is already created")

        self.backend.create_context()
        self._context_created = True
        self.backend.configure_app(manual_callback_management=True)

    def show_viewport(
        self,
        *,
        title: str,
        width: int,
        height: int,
        primary_window: int | str,
    ) -> None:
        if not self._context_created:
            raise RuntimeError("Dear PyGui context is not created")

        self.backend.create_viewport(title=title, width=width, height=height)
        self.backend.setup_dearpygui()
        self.backend.show_viewport()
        self.backend.set_primary_window(primary_window, True)

    def process_callbacks(self) -> None:
        jobs = self.backend.get_callback_queue()
        self.backend.run_callbacks(jobs)

    def render_frame(self) -> None:
        self.backend.render_dearpygui_frame()

    def destroy_context(self) -> None:
        if not self._context_created:
            return

        try:
            self.backend.destroy_context()
        finally:
            self._context_created = False
