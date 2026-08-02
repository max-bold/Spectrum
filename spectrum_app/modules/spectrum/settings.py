from typing import TYPE_CHECKING, Any, Literal, cast

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


GeneratorMode = Literal["log chirp", "pink noise"]


class SpectrumSettings:
    MODULE_ID = "spectrum"
    DEFAULT_GENERATOR_MODE: GeneratorMode = "log chirp"
    DEFAULT_WELCH_SAMPLES = 8192
    DEFAULT_ONLINE_WELCH = True

    def __init__(self, app_settings: AppSettings) -> None:
        self._app_settings = app_settings

    @property
    def generator_mode(self) -> GeneratorMode:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "generator_mode",
            self.DEFAULT_GENERATOR_MODE,
        )
        if value not in ("log chirp", "pink noise"):
            return self.DEFAULT_GENERATOR_MODE
        return cast(GeneratorMode, value)

    @generator_mode.setter
    def generator_mode(self, value: GeneratorMode) -> None:
        if value not in ("log chirp", "pink noise"):
            raise ValueError(f"Unknown generator mode: {value}")
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "generator_mode",
            value,
        )

    @property
    def welch_samples(self) -> int:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "welch_samples",
            self.DEFAULT_WELCH_SAMPLES,
        )
        try:
            return self._normalize_welch_samples(value)
        except (TypeError, ValueError):
            return self.DEFAULT_WELCH_SAMPLES

    @welch_samples.setter
    def welch_samples(self, value: int) -> None:
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "welch_samples",
            self._normalize_welch_samples(value),
        )

    @property
    def online_welch(self) -> bool:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "online_welch",
            self.DEFAULT_ONLINE_WELCH,
        )
        return value if isinstance(value, bool) else self.DEFAULT_ONLINE_WELCH

    @online_welch.setter
    def online_welch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("Online Welch setting must be boolean")
        self._app_settings.set_module_setting(
            self.MODULE_ID,
            "online_welch",
            value,
        )

    @staticmethod
    def _normalize_welch_samples(value: Any) -> int:
        samples = max(16, int(value))
        lower = 1 << (samples.bit_length() - 1)
        upper = lower << 1
        return lower if samples - lower <= upper - samples else upper


class SpectrumSettingsWindow:
    WIDTH = 480
    HEIGHT = 220
    TAG = "module::spectrum::settings"
    MENU_ITEM = "module::spectrum::settings_menu_item"

    def __init__(
        self,
        app: "SpectrumApplication",
        settings: SpectrumSettings,
    ) -> None:
        self.app = app
        self.settings = settings
        self.generator_mode = "module::spectrum::settings::generator_mode"
        self.welch_samples = "module::spectrum::settings::welch_samples"
        self.online_welch = "module::spectrum::settings::online_welch"

    def build(self) -> None:
        dpg.add_menu_item(
            label="Spectrum",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Spectrum settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            dpg.add_text("Generator")
            dpg.add_combo(
                ["log chirp", "pink noise"],
                tag=self.generator_mode,
                default_value=self.settings.generator_mode,
                width=-1,
                callback=self._set_generator_mode,
            )
            dpg.add_separator()
            dpg.add_text("Welch bucket size, samples")
            dpg.add_input_int(
                tag=self.welch_samples,
                default_value=self.settings.welch_samples,
                width=-1,
                step=0,
                callback=self._set_welch_samples,
            )
            dpg.add_checkbox(
                label="Online Welch",
                tag=self.online_welch,
                default_value=self.settings.online_welch,
                callback=self._set_online_welch,
            )

    def destroy(self) -> None:
        if dpg.does_item_exist(self.TAG):
            dpg.delete_item(self.TAG)
        if dpg.does_item_exist(self.MENU_ITEM):
            dpg.delete_item(self.MENU_ITEM)

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
        dpg.set_item_pos(self.TAG, position)
        dpg.configure_item(self.TAG, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.TAG, show=False)

    def _sync_controls(self) -> None:
        dpg.set_value(self.generator_mode, self.settings.generator_mode)
        dpg.set_value(self.welch_samples, self.settings.welch_samples)
        dpg.set_value(self.online_welch, self.settings.online_welch)

    def _set_generator_mode(
        self,
        sender: int | str,
        value: str,
        user_data=None,
    ) -> None:
        self.settings.generator_mode = cast(GeneratorMode, value)

    def _set_welch_samples(
        self,
        sender: int | str,
        value: int,
        user_data=None,
    ) -> None:
        self.settings.welch_samples = value
        dpg.set_value(sender, self.settings.welch_samples)

    def _set_online_welch(
        self,
        sender: int | str,
        value: bool,
        user_data=None,
    ) -> None:
        self.settings.online_welch = value
