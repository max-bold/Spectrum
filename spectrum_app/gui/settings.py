from typing import TYPE_CHECKING, cast

import dearpygui.dearpygui as dpg

from spectrum_app.core.audio import AudioDirection
from spectrum_app.core.settings import AxisScale, PhaseUnit

if TYPE_CHECKING:
    from spectrum_app.application import SpectrumApplication


class SettingsWindow:
    WIDTH = 667
    HEIGHT = 400

    def __init__(self, app: "SpectrumApplication") -> None:
        self.app = app
        self.tag = "app::settings_window"
        self.frequency_range = "app::settings::frequency_range"
        self.impedance_scale = "app::settings::impedance_scale"
        self.thd_scale = "app::settings::thd_scale"
        self.phase_unit = "app::settings::phase_unit"
        self.input_device = "app::settings::input_device"
        self.output_device = "app::settings::output_device"
        self.input_block_size = "app::settings::input_block_size"
        self.output_block_size = "app::settings::output_block_size"
        self._input_device_ids: dict[str, str] = {}
        self._output_device_ids: dict[str, str] = {}

    def build(self) -> None:
        settings = self.app.settings
        with dpg.window(  # pyright: ignore[reportGeneralTypeIssues]
            label="Application settings",
            tag=self.tag,
            width=self.WIDTH,
            height=self.HEIGHT,
            show=False,
            modal=True,
            no_resize=True,
            no_collapse=True,
            on_close=self.hide,
        ):
            with dpg.tab_bar():  # pyright: ignore[reportGeneralTypeIssues]
                with dpg.tab(  # pyright: ignore[reportGeneralTypeIssues]
                    label="Plot",
                ):
                    dpg.add_text("X axis range, Hz")
                    dpg.add_input_intx(
                        tag=self.frequency_range,
                        size=2,
                        default_value=[
                            int(settings.frequency_range[0]),
                            int(settings.frequency_range[1]),
                        ],
                        callback=self._set_frequency_range,
                    )

                    dpg.add_separator()
                    dpg.add_text("Impedance scale")
                    dpg.add_radio_button(
                        ("linear", "log"),
                        tag=self.impedance_scale,
                        default_value=settings.impedance_scale,
                        horizontal=True,
                        callback=self._set_impedance_scale,
                    )

                    dpg.add_separator()
                    dpg.add_text("THD scale")
                    dpg.add_radio_button(
                        ("linear", "log"),
                        tag=self.thd_scale,
                        default_value=settings.thd_scale,
                        horizontal=True,
                        callback=self._set_thd_scale,
                    )

                    dpg.add_separator()
                    dpg.add_text("Phase display")
                    dpg.add_radio_button(
                        ("deg", "deg/dec"),
                        tag=self.phase_unit,
                        default_value=settings.phase_unit,
                        horizontal=True,
                        callback=self._set_phase_unit,
                    )

                with dpg.tab(  # pyright: ignore[reportGeneralTypeIssues]
                    label="AudioIO",
                ):
                    input_items, input_value = self._device_items("input")
                    dpg.add_text("Input device")
                    dpg.add_combo(
                        input_items,
                        tag=self.input_device,
                        default_value=input_value,
                        width=-1,
                        callback=self._set_input_device,
                    )
                    dpg.add_text("Recommended input block size")
                    dpg.add_input_int(
                        tag=self.input_block_size,
                        default_value=settings.input_block_size,
                        min_value=1,
                        min_clamped=True,
                        callback=self._set_input_block_size,
                    )

                    dpg.add_separator()
                    output_items, output_value = self._device_items("output")
                    dpg.add_text("Output device")
                    dpg.add_combo(
                        output_items,
                        tag=self.output_device,
                        default_value=output_value,
                        width=-1,
                        callback=self._set_output_device,
                    )
                    dpg.add_text("Recommended output block size")
                    dpg.add_input_int(
                        tag=self.output_block_size,
                        default_value=settings.output_block_size,
                        min_value=1,
                        min_clamped=True,
                        callback=self._set_output_block_size,
                    )

    def update(self) -> None:
        if self.app._audio_service.consume_devices_changed():
            self._sync_audio_devices()

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
        dpg.set_item_pos(self.tag, position)
        dpg.configure_item(self.tag, show=True)

    def hide(self, sender=None, app_data=None, user_data=None) -> None:
        dpg.configure_item(self.tag, show=False)

    def _sync_controls(self) -> None:
        settings = self.app.settings
        dpg.set_value(
            self.frequency_range,
            [
                int(settings.frequency_range[0]),
                int(settings.frequency_range[1]),
                0,
                0,
            ],
        )
        dpg.set_value(self.impedance_scale, settings.impedance_scale)
        dpg.set_value(self.thd_scale, settings.thd_scale)
        dpg.set_value(self.phase_unit, settings.phase_unit)
        self._sync_audio_devices()
        dpg.set_value(self.input_block_size, settings.input_block_size)
        dpg.set_value(self.output_block_size, settings.output_block_size)

    def _sync_audio_devices(self) -> None:
        input_items, input_value = self._device_items("input")
        output_items, output_value = self._device_items("output")
        dpg.configure_item(self.input_device, items=input_items)
        dpg.configure_item(self.output_device, items=output_items)
        dpg.set_value(self.input_device, input_value)
        dpg.set_value(self.output_device, output_value)

    def _device_items(
        self, direction: AudioDirection
    ) -> tuple[list[str], str]:
        if direction == "input":
            default_label = self.app._audio_service.DEFAULT_INPUT_LABEL
            devices = self.app._audio_service.input_devices
            selected_id = self.app.settings.input_device
            destination = self._input_device_ids
        else:
            default_label = self.app._audio_service.DEFAULT_OUTPUT_LABEL
            devices = self.app._audio_service.output_devices
            selected_id = self.app.settings.output_device
            destination = self._output_device_ids

        destination.clear()
        destination[default_label] = ""
        for device in devices:
            destination[device.label] = device.id

        selected_label = default_label
        if selected_id:
            selected = next(
                (device for device in devices if device.id == selected_id),
                None,
            )
            if selected is not None:
                selected_label = selected.label
            else:
                selected_label = self._unavailable_device_label(selected_id)
                destination[selected_label] = selected_id
        return list(destination), selected_label

    @staticmethod
    def _unavailable_device_label(device_id: str) -> str:
        parts = device_id.split("\x1f")
        if len(parts) >= 2:
            return f"Unavailable: {parts[1]} ({parts[0]})"
        return f"Unavailable: {device_id}"

    def _set_frequency_range(
        self, sender: int | str, value: list[int], user_data=None
    ) -> None:
        try:
            self.app.settings.frequency_range = (float(value[0]), float(value[1]))
        except ValueError:
            low, high = self.app.settings.frequency_range
            dpg.set_value(sender, [int(low), int(high), 0, 0])

    def _set_impedance_scale(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.impedance_scale = cast(AxisScale, value)

    def _set_thd_scale(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.thd_scale = cast(AxisScale, value)

    def _set_phase_unit(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.phase_unit = cast(PhaseUnit, value)

    def _set_input_device(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.input_device = self._input_device_ids[value]

    def _set_output_device(
        self, sender: int | str, value: str, user_data=None
    ) -> None:
        self.app.settings.output_device = self._output_device_ids[value]

    def _set_input_block_size(
        self, sender: int | str, value: int, user_data=None
    ) -> None:
        try:
            self.app.settings.input_block_size = value
        except ValueError:
            dpg.set_value(sender, self.app.settings.input_block_size)

    def _set_output_block_size(
        self, sender: int | str, value: int, user_data=None
    ) -> None:
        try:
            self.app.settings.output_block_size = value
        except ValueError:
            dpg.set_value(sender, self.app.settings.output_block_size)
