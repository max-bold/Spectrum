from collections.abc import Mapping
from dataclasses import dataclass
from threading import Condition, Event, RLock, Thread, current_thread
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from spectrum_app.core.settings import (
    AppSettings,
    InputRouting,
    OutputRouting,
)


AudioDirection = Literal["input", "output"]


class AudioError(RuntimeError):
    """An error reported by the application audio layer."""


@dataclass(frozen=True)
class AudioDevice:
    id: str
    index: int
    name: str
    host_api: str
    sample_rate: int
    input_channels: int
    output_channels: int

    def channels(self, direction: AudioDirection) -> int:
        if direction == "input":
            return self.input_channels
        return self.output_channels

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.host_api})"


class AudioService:
    """Owns PortAudio state, device discovery and all application streams."""

    DEFAULT_INPUT_LABEL = "Default input device"
    DEFAULT_OUTPUT_LABEL = "Default output device"
    UNSUPPORTED_HOST_APIS = frozenset({"Windows WDM-KS"})

    def __init__(
        self,
        settings: AppSettings,
        backend: Any = sd,
        update_interval: float = 1.0,
    ) -> None:
        self.settings = settings
        self._backend = backend
        self._update_interval = update_interval
        self._condition = Condition(RLock())
        self._operation_lock = RLock()
        self._portaudio_lock = RLock()
        self._stop_event = Event()
        self._wake_event = Event()
        self._updater: Thread | None = None
        self._refreshing = False
        self._active_sessions = 0
        self._streams: dict[AudioDirection, Any] = {}
        self._stream_routing: dict[
            AudioDirection,
            InputRouting | OutputRouting,
        ] = {}
        self._input_devices: tuple[AudioDevice, ...] = ()
        self._output_devices: tuple[AudioDevice, ...] = ()
        self._default_input_id = ""
        self._default_output_id = ""
        self.devices_changed = True
        self._last_error: str | None = None

    @property
    def input_devices(self) -> tuple[AudioDevice, ...]:
        with self._condition:
            return self._input_devices

    @property
    def output_devices(self) -> tuple[AudioDevice, ...]:
        with self._condition:
            return self._output_devices

    @property
    def selected_input_device(self) -> AudioDevice | None:
        return self._selected_device("input")

    @property
    def selected_output_device(self) -> AudioDevice | None:
        return self._selected_device("output")

    @property
    def input_routing(self) -> InputRouting:
        device = self.selected_input_device
        if device is None:
            return None, None
        defaults = self.default_input_routing(device.input_channels)
        normalized = tuple(
            channel
            if channel is None or channel < device.input_channels
            else defaults[index]
            for index, channel in enumerate(self.settings.input_routing)
        )
        return normalized[0], normalized[1]

    @property
    def output_routing(self) -> OutputRouting:
        device = self.selected_output_device
        if device is None:
            return ()
        configured = self.settings.output_routing
        return tuple(
            configured[index] if index < len(configured) else True
            for index in range(device.output_channels)
        )

    @staticmethod
    def default_input_routing(channels: int) -> InputRouting:
        if channels <= 0:
            return None, None
        return (0, 1) if channels > 1 else (0, 0)

    @staticmethod
    def default_output_routing(channels: int) -> OutputRouting:
        return (True,) * max(0, channels)

    def start(self) -> None:
        with self._condition:
            if self._updater is not None:
                return
            self._stop_event.clear()

        self._refresh_devices()
        updater = Thread(
            target=self._update_loop,
            name="audio-device-updater",
            daemon=True,
        )
        with self._condition:
            self._updater = updater
        updater.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        with self._condition:
            updater = self._updater
        if updater is not None and updater is not current_thread():
            updater.join()
        self.close_all()
        with self._condition:
            self._updater = None

    def request_device_refresh(self) -> None:
        self._wake_event.set()

    def consume_devices_changed(self) -> bool:
        with self._condition:
            changed = self.devices_changed
            self.devices_changed = False
            return changed

    def consume_error(self) -> str | None:
        with self._condition:
            error = self._last_error
            self._last_error = None
            return error

    def open_stream(self, direction: AudioDirection) -> bool:
        with self._operation_lock:
            if direction in self._streams:
                return True

            self._reserve_session()
            stream: Any | None = None
            try:
                device = self._selected_device(direction)
                if device is None:
                    raise AudioError(f"Selected {direction} device is unavailable")

                physical_channels = device.channels(direction)
                if physical_channels <= 0:
                    raise AudioError(f"Device does not support {direction}")

                routing: InputRouting | OutputRouting
                if direction == "input":
                    routing = self.input_routing
                    selected_channels = [
                        channel for channel in routing if channel is not None
                    ]
                else:
                    routing = self.output_routing
                    selected_channels = [
                        index
                        for index, enabled in enumerate(routing)
                        if enabled
                    ]
                native_channels = (
                    max(selected_channels) + 1 if selected_channels else 1
                )

                stream_type = (
                    self._backend.InputStream
                    if direction == "input"
                    else self._backend.OutputStream
                )
                with self._portaudio_lock:
                    native_stream: Any = stream_type(
                        device=device.index,
                        samplerate=device.sample_rate,
                        channels=native_channels,
                        dtype="float32",
                        blocksize=0,
                    )
                    stream = native_stream
                    native_stream.start()
                self._streams[direction] = stream
                if direction == "input":
                    self._stream_routing[direction] = cast(InputRouting, routing)
                else:
                    output_routing = cast(OutputRouting, routing)
                    self._stream_routing[direction] = output_routing[
                        :native_channels
                    ]
                return True
            except Exception as error:
                if stream is not None:
                    try:
                        self._close_native_stream(stream)
                    except Exception:
                        pass
                self._release_session()
                self._report_error(error)
                return False

    def close_stream(self, direction: AudioDirection) -> bool:
        with self._operation_lock:
            stream = self._streams.pop(direction, None)
            self._stream_routing.pop(direction, None)
            if stream is None:
                return True
            try:
                self._close_native_stream(stream)
            except Exception as error:
                self._report_error(error)
                return False
            finally:
                self._release_session()
            return True

    def close_all(self) -> bool:
        result = True
        for direction in ("input", "output"):
            result = self.close_stream(direction) and result
        return result

    def read(self, samples: int) -> NDArray[np.float32]:
        if samples <= 0:
            raise ValueError("Sample count must be positive")
        with self._operation_lock:
            stream = self._streams.get("input")
            routing = self._stream_routing.get("input")
        if stream is None:
            raise AudioError("Audio input is not open")
        try:
            data, overflowed = stream.read(samples)
            if overflowed:
                raise AudioError("Audio input overflow")
            physical = np.asarray(data, dtype=np.float32)
            if physical.ndim != 2:
                raise AudioError("Audio input returned an invalid array")
            logical = np.zeros((len(physical), 2), dtype=np.float32)
            if not isinstance(routing, tuple) or len(routing) != 2:
                raise AudioError("Audio input routing is unavailable")
            for logical_index, physical_index in enumerate(routing):
                if physical_index is not None:
                    logical[:, logical_index] = physical[:, physical_index]
            return logical
        except Exception as error:
            self.close_stream("input")
            self._report_error(error)
            raise AudioError(str(error)) from error

    def write(self, data: NDArray[Any]) -> None:
        with self._operation_lock:
            stream = self._streams.get("output")
            routing = self._stream_routing.get("output")
        if stream is None:
            raise AudioError("Audio output is not open")
        try:
            logical = np.asarray(data, dtype=np.float32)
            if logical.ndim != 1:
                raise AudioError(
                    "Audio output data must be a one-dimensional mono array"
                )
            if not isinstance(routing, tuple):
                raise AudioError("Audio output routing is unavailable")
            physical = np.zeros((len(logical), len(routing)), dtype=np.float32)
            for physical_index, enabled in enumerate(routing):
                if enabled:
                    physical[:, physical_index] = logical
            underflowed = stream.write(physical)
            if underflowed:
                raise AudioError("Audio output underflow")
        except Exception as error:
            self.close_stream("output")
            self._report_error(error)
            raise AudioError(str(error)) from error

    def _reserve_session(self) -> None:
        with self._condition:
            self._active_sessions += 1
            while self._refreshing:
                self._condition.wait()

    def _release_session(self) -> None:
        with self._condition:
            self._active_sessions -= 1
            if self._active_sessions == 0:
                self._wake_event.set()
            self._condition.notify_all()

    def _selected_device(self, direction: AudioDirection) -> AudioDevice | None:
        with self._condition:
            if direction == "input":
                devices = self._input_devices
                selected_id = self.settings.input_device or self._default_input_id
            else:
                devices = self._output_devices
                selected_id = self.settings.output_device or self._default_output_id
            return next((device for device in devices if device.id == selected_id), None)

    def _update_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(self._update_interval)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            self._refresh_devices()

    def _refresh_devices(self) -> None:
        with self._condition:
            if self._active_sessions or self._refreshing:
                return
            self._refreshing = True

        try:
            with self._portaudio_lock:
                terminate = getattr(self._backend, "_terminate", None)
                initialize = getattr(self._backend, "_initialize", None)
                if terminate is not None and initialize is not None:
                    terminate()
                    initialize()
                devices = self._backend.query_devices()
                host_apis = self._backend.query_hostapis()
                default_input_index = self._default_device_index("input")
                default_output_index = self._default_device_index("output")

            input_devices: list[AudioDevice] = []
            output_devices: list[AudioDevice] = []
            id_counts: dict[str, int] = {}
            for index, raw_device in enumerate(devices):
                device = self._make_device(index, raw_device, host_apis, id_counts)
                if device.host_api in self.UNSUPPORTED_HOST_APIS:
                    continue
                if device.input_channels > 0:
                    input_devices.append(device)
                if device.output_channels > 0:
                    output_devices.append(device)

            default_input_id = self._id_for_index(input_devices, default_input_index)
            default_output_id = self._id_for_index(output_devices, default_output_index)
            snapshots = (tuple(input_devices), tuple(output_devices))
            with self._condition:
                changed = snapshots != (self._input_devices, self._output_devices)
                changed |= default_input_id != self._default_input_id
                changed |= default_output_id != self._default_output_id
                self._input_devices, self._output_devices = snapshots
                self._default_input_id = default_input_id
                self._default_output_id = default_output_id
                self.devices_changed |= changed
        except Exception as error:
            self._report_error(error)
        finally:
            with self._condition:
                self._refreshing = False
                self._condition.notify_all()

    def _default_device_index(self, direction: AudioDirection) -> int:
        try:
            raw_device = self._backend.query_devices(None, direction)
            if isinstance(raw_device, Mapping) and "index" in raw_device:
                return int(raw_device["index"])
        except Exception:
            pass

        default_devices = getattr(getattr(self._backend, "default", None), "device", ())
        position = 0 if direction == "input" else 1
        try:
            return int(default_devices[position])
        except (IndexError, TypeError, ValueError):
            return -1

    @staticmethod
    def _make_device(
        index: int,
        raw_device: Mapping[str, Any],
        host_apis: Any,
        id_counts: dict[str, int],
    ) -> AudioDevice:
        host_api_index = int(raw_device.get("hostapi", -1))
        try:
            host_api = str(host_apis[host_api_index]["name"])
        except (IndexError, KeyError, TypeError):
            host_api = f"Host API {host_api_index}"
        name = str(raw_device.get("name", f"Device {index}"))
        base_id = f"{host_api}\x1f{name}"
        ordinal = id_counts.get(base_id, 0)
        id_counts[base_id] = ordinal + 1
        device_id = base_id if ordinal == 0 else f"{base_id}\x1f{ordinal}"
        return AudioDevice(
            id=device_id,
            index=int(raw_device.get("index", index)),
            name=name,
            host_api=host_api,
            sample_rate=int(round(float(raw_device.get("default_samplerate", 0)))),
            input_channels=int(raw_device.get("max_input_channels", 0)),
            output_channels=int(raw_device.get("max_output_channels", 0)),
        )

    @staticmethod
    def _id_for_index(devices: list[AudioDevice], index: int) -> str:
        device = next((item for item in devices if item.index == index), None)
        return device.id if device is not None else ""

    @staticmethod
    def _close_native_stream(stream: Any) -> None:
        try:
            stream.stop()
        finally:
            stream.close()

    def _report_error(self, error: Exception) -> None:
        with self._condition:
            self._last_error = str(error)


class AudioInput:
    def __init__(self, service: AudioService) -> None:
        self._service = service

    @property
    def sample_rate(self) -> int:
        device = self._service.selected_input_device
        return device.sample_rate if device is not None else 0

    @property
    def block_size(self) -> int:
        return self._service.settings.input_block_size

    def open(self) -> bool:
        return self._service.open_stream("input")

    def read(self, samples: int) -> NDArray[np.float32]:
        return self._service.read(samples)

    def close(self) -> bool:
        return self._service.close_stream("input")


class AudioOutput:
    def __init__(self, service: AudioService) -> None:
        self._service = service

    @property
    def sample_rate(self) -> int:
        device = self._service.selected_output_device
        return device.sample_rate if device is not None else 0

    @property
    def block_size(self) -> int:
        return self._service.settings.output_block_size

    def open(self) -> bool:
        return self._service.open_stream("output")

    def write(self, data: NDArray[Any]) -> None:
        self._service.write(data)

    def close(self) -> bool:
        return self._service.close_stream("output")
