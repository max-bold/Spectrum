from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dearpygui.dearpygui as dpg

from spectrum_app.core.settings import AppSettings

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


SettingsChanged = Callable[[str], None]


class THDSettings:
    MODULE_ID = "thd"
    DEFAULT_SEGMENT_SECONDS = 1.0
    DEFAULT_OVERLAP_PERCENT = 90.0
    DEFAULT_SWEEP_BAND_EXPANSION = 1.5
    DEFAULT_MASK_EXPANSION = 2.0
    DEFAULT_POINTS = 1_200

    def __init__(
        self,
        app_settings: AppSettings,
        on_change: SettingsChanged | None = None,
    ) -> None:
        self._app_settings = app_settings
        self._on_change = on_change

    @property
    def segment_seconds(self) -> float:
        return self._float_setting(
            "segment_seconds",
            self.DEFAULT_SEGMENT_SECONDS,
            minimum=0.05,
            maximum=10.0,
        )

    @segment_seconds.setter
    def segment_seconds(self, value: float) -> None:
        self._set(
            "segment_seconds",
            self._normalize_float(value, 0.05, 10.0),
        )

    @property
    def overlap_percent(self) -> float:
        return self._float_setting(
            "overlap_percent",
            self.DEFAULT_OVERLAP_PERCENT,
            minimum=0.0,
            maximum=99.0,
        )

    @overlap_percent.setter
    def overlap_percent(self, value: float) -> None:
        self._set(
            "overlap_percent",
            self._normalize_float(value, 0.0, 99.0),
        )

    @property
    def sweep_band_expansion(self) -> float:
        return self._float_setting(
            "sweep_band_expansion",
            self.DEFAULT_SWEEP_BAND_EXPANSION,
            minimum=1.01,
            maximum=4.0,
        )

    @sweep_band_expansion.setter
    def sweep_band_expansion(self, value: float) -> None:
        self._set(
            "sweep_band_expansion",
            self._normalize_float(value, 1.01, 4.0),
        )

    @property
    def mask_expansion(self) -> float:
        return self._float_setting(
            "mask_expansion",
            self.DEFAULT_MASK_EXPANSION,
            minimum=0.1,
            maximum=10.0,
        )

    @mask_expansion.setter
    def mask_expansion(self, value: float) -> None:
        self._set(
            "mask_expansion",
            self._normalize_float(value, 0.1, 10.0),
        )

    @property
    def points(self) -> int:
        value = self._app_settings.module_setting(
            self.MODULE_ID,
            "points",
            self.DEFAULT_POINTS,
        )
        try:
            return min(10_000, max(32, int(value)))
        except (TypeError, ValueError):
            return self.DEFAULT_POINTS

    @points.setter
    def points(self, value: int) -> None:
        self._set("points", min(10_000, max(32, int(value))))

    def _float_setting(
        self,
        key: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        value = self._app_settings.module_setting(self.MODULE_ID, key, default)
        try:
            return self._normalize_float(value, minimum, maximum)
        except (TypeError, ValueError):
            return default

    def _set(self, key: str, value: int | float) -> None:
        if self._app_settings.module_setting(self.MODULE_ID, key) == value:
            return
        self._app_settings.set_module_setting(self.MODULE_ID, key, value)
        if self._on_change is not None:
            self._on_change(key)

    @staticmethod
    def _normalize_float(value: Any, minimum: float, maximum: float) -> float:
        return min(maximum, max(minimum, float(value)))


class THDSettingsWindow:
    WIDTH = 480
    HEIGHT = 360
    TAG = "module::thd::settings"
    MENU_ITEM = "module::thd::settings_menu_item"

    def __init__(
        self,
        app: "SpectrumApplication",
        settings: THDSettings,
    ) -> None:
        self.app = app
        self.settings = settings
        self.segment_seconds = "module::thd::settings::segment_seconds"
        self.overlap_percent = "module::thd::settings::overlap_percent"
        self.sweep_band_expansion = "module::thd::settings::sweep_band_expansion"
        self.mask_expansion = "module::thd::settings::mask_expansion"
        self.points = "module::thd::settings::points"

    def build(self) -> None:
        dpg.add_menu_item(
            label="THD",
            tag=self.MENU_ITEM,
            parent=self.app.main_window.settings_menu,
            callback=self.show,
        )
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="THD settings",
            tag=self.TAG,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            dpg.add_text("STFT window, s")
            dpg.add_input_float(
                tag=self.segment_seconds,
                default_value=self.settings.segment_seconds,
                width=-1,
                step=0.05,
                callback=self._set_segment_seconds,
            )
            dpg.add_text("STFT overlap, %")
            dpg.add_input_float(
                tag=self.overlap_percent,
                default_value=self.settings.overlap_percent,
                width=-1,
                step=1.0,
                callback=self._set_overlap_percent,
            )
            dpg.add_separator()
            dpg.add_text("Sweep band expansion")
            dpg.add_input_float(
                tag=self.sweep_band_expansion,
                default_value=self.settings.sweep_band_expansion,
                width=-1,
                step=0.1,
                callback=self._set_sweep_band_expansion,
            )
            dpg.add_text("Fundamental mask expansion")
            dpg.add_input_float(
                tag=self.mask_expansion,
                default_value=self.settings.mask_expansion,
                width=-1,
                step=0.1,
                callback=self._set_mask_expansion,
            )
            dpg.add_text("Result points")
            dpg.add_input_int(
                tag=self.points,
                default_value=self.settings.points,
                width=-1,
                step=0,
                callback=self._set_points,
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

    def _sync_controls(self) -> None:
        dpg.set_value(self.segment_seconds, self.settings.segment_seconds)
        dpg.set_value(self.overlap_percent, self.settings.overlap_percent)
        dpg.set_value(
            self.sweep_band_expansion,
            self.settings.sweep_band_expansion,
        )
        dpg.set_value(self.mask_expansion, self.settings.mask_expansion)
        dpg.set_value(self.points, self.settings.points)

    def _change_allowed(self) -> bool:
        if not self.app.app_state.measuring:
            return True
        self.app.main_window.set_status_text(
            "Stop the active measurement before changing THD settings"
        )
        self._sync_controls()
        return False

    def _set_segment_seconds(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.segment_seconds = value
            dpg.set_value(sender, self.settings.segment_seconds)

    def _set_overlap_percent(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.overlap_percent = value
            dpg.set_value(sender, self.settings.overlap_percent)

    def _set_sweep_band_expansion(
        self,
        sender,
        value: float,
        user_data=None,
    ) -> None:
        if self._change_allowed():
            self.settings.sweep_band_expansion = value
            dpg.set_value(sender, self.settings.sweep_band_expansion)

    def _set_mask_expansion(self, sender, value: float, user_data=None) -> None:
        if self._change_allowed():
            self.settings.mask_expansion = value
            dpg.set_value(sender, self.settings.mask_expansion)

    def _set_points(self, sender, value: int, user_data=None) -> None:
        if self._change_allowed():
            self.settings.points = value
            dpg.set_value(sender, self.settings.points)
