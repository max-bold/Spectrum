import dearpygui.dearpygui as dpg

from spectrum_app.gui import build_ui
from spectrum_app.callbacks import bind_input_commit_handlers
from spectrum_app.state import create_app_state
from spectrum_app.ui.sync import sync_ui


def main() -> None:
    dpg.create_context()
    state = create_app_state()
    refs = build_ui(state)
    bind_input_commit_handlers(state, refs)
    state.start_services()

    dpg.create_viewport(title="BM Spectrum", width=1024, height=768)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Primary Window", True)

    try:
        while dpg.is_dearpygui_running():
            sync_ui(state, refs)
            dpg.render_dearpygui_frame()
    finally:
        state.stop_services()
        dpg.destroy_context()


if __name__ == "__main__":
    main()
