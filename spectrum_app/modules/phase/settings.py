from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication

SettingsChanged = Callable[[str], None]


class PhaseSettings:
    MODULE_ID = "phase"
    DEFAULT_PRE_SILENCE = 0.5
    DEFAULT_POST_SILENCE = 0.5
    DEFAULT_FADE = 0.5

    def __init__(
        self,
        app_settings: AppSettings,
        on_change: SettingsChanged | None = None,
    ) -> None:
        self._app_settings = app_settings
        self._on_change = on_change

    @property
    def pre_silence(self) -> float:
        return self._get("pre_silence", self.DEFAULT_PRE_SILENCE)

    @pre_silence.setter
    def pre_silence(self, value: float) -> None:
        self._set("pre_silence", value)

    @property
    def post_silence(self) -> float:
        return self._get("post_silence", self.DEFAULT_POST_SILENCE)

    @post_silence.setter
    def post_silence(self, value: float) -> None:
        self._set("post_silence", value)

    @property
    def fade(self) -> float:
        return self._get("fade", self.DEFAULT_FADE)

    @fade.setter
    def fade(self, value: float) -> None:
        self._set("fade", value)

    def _get(self, key: str, default: float) -> float:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        try:
            return self._normalize(value)
        except (TypeError, ValueError):
            return default

    def _set(self, key: str, value: Any) -> None:
        normalized = self._normalize(value)
        if self._app_settings.module_setting(self.MODULE_ID, key) == normalized:
            return
        self._app_settings.set_module_setting(self.MODULE_ID, key, normalized)
        if self._on_change is not None:
            self._on_change(key)

    @staticmethod
    def _normalize(value: Any) -> float:
        return min(10.0, max(0.0, float(value)))


class PhaseSettingsWindow:
    WIDTH = 480
    HEIGHT = 250
    TAG = "module::phase::settings"
    MENU_ITEM = "module::phase::settings_menu_item"

    def __init__(self, app: "SpectrumApplication", settings: PhaseSettings) -> None:
        self.app = app
        self.settings = settings
        self.pre_silence = "module::phase::settings::pre_silence"
        self.post_silence = "module::phase::settings::post_silence"
        self.fade = "module::phase::settings::fade"

    def build(self) -> None:
        dpg.add_menu_item(
            label="Phase",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Phase settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            self._add_float(
                "Pre silence, s",
                self.pre_silence,
                self.settings.pre_silence,
                self._set_pre_silence,
            )
            self._add_float(
                "Post silence, s",
                self.post_silence,
                self.settings.post_silence,
                self._set_post_silence,
            )
            self._add_float(
                "Fade in/out, s",
                self.fade,
                self.settings.fade,
                self._set_fade,
            )

    def destroy(self) -> None:
        for item in (self.TAG, self.MENU_ITEM):
            if dpg.does_item_exist(item):
                dpg.delete_item(item)

    def show(self, sender=None, app_data=None, user_data=None) -> None:
        self._sync_controls()
        main_position = dpg.get_item_pos(self.app.main_window.tag)
        main_size = dpg.get_item_rect_size(self.app.main_window.tag)
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
        dpg.configure_item(self.TAG, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.TAG, show=False)

    def _add_float(self, label: str, tag: str, value: float, callback) -> None:
        dpg.add_text(label)
        dpg.add_input_float(
            tag=tag,
            default_value=value,
            min_value=0.0,
            min_clamped=True,
            step=0,
            width=-1,
            callback=callback,
        )

    def _sync_controls(self) -> None:
        dpg.set_value(self.pre_silence, self.settings.pre_silence)
        dpg.set_value(self.post_silence, self.settings.post_silence)
        dpg.set_value(self.fade, self.settings.fade)

    def _change_allowed(self) -> bool:
        if not self.app.app_state.measuring:
            return True
        self.app.main_window.set_status_text(
            "Stop the active measurement before changing Phase settings"
        )
        self._sync_controls()
        return False

    def _set_pre_silence(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.pre_silence = value
            dpg.set_value(sender, self.settings.pre_silence)

    def _set_post_silence(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.post_silence = value
            dpg.set_value(sender, self.settings.post_silence)

    def _set_fade(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.fade = value
            dpg.set_value(sender, self.settings.fade)
