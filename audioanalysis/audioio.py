from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Literal, Mapping, cast

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd

from .generators import pink_noise, pink_noise_zi
from .types import FrequencyBand


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    host_api: str
    input_channels: int
    output_channels: int
    default_sample_rate: float


def list_devices(kind: Literal["input", "output"] | None = None) -> list[AudioDevice]:
    """List PortAudio devices as reusable data objects."""
    hostapis = sd.query_hostapis()
    devices = sd.query_devices()
    hostapi_names = [str(item["name"]) for item in hostapis if isinstance(item, dict)]
    result: list[AudioDevice] = []
    for device in devices:
        if not isinstance(device, dict):
            continue
        host_api_index = int(device["hostapi"])
        if host_api_index >= len(hostapi_names):
            continue
        audio_device = AudioDevice(
            index=int(device["index"]),
            name=str(device["name"]),
            host_api=hostapi_names[host_api_index],
            input_channels=int(device["max_input_channels"]),
            output_channels=int(device["max_output_channels"]),
            default_sample_rate=float(device["default_samplerate"]),
        )
        if kind == "input" and audio_device.input_channels <= 0:
            continue
        if kind == "output" and audio_device.output_channels <= 0:
            continue
        result.append(audio_device)
    return result


def play_and_record(
    signal: NDArray[np.floating],
    *,
    sample_rate: int,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    channels: int = 2,
    blocking: bool = True,
) -> NDArray[np.float32]:
    """Play a signal and record synchronized input through sounddevice."""
    recording = sd.playrec(
        np.asarray(signal, dtype=np.float32),
        samplerate=sample_rate,
        channels=channels,
        device=(input_device, output_device),
        blocking=blocking,
    )
    return np.asarray(recording, dtype=np.float32)


class PinkNoiseThread(Thread):
    """Continuously write band-limited pink noise to a sounddevice output device.

    The object owns one worker thread. ``start()`` starts the worker on the
    first call and starts playback on every call, ``stop()`` stops the current
    playback pass, and ``close()`` terminates the worker. The pinking and
    band-pass filters keep their internal state within one playback pass, so
    the output is continuous and does not restart its filter transient on every
    write.

    Args:
        device: Output device index or name accepted by ``sounddevice``.
        band: Output frequency range in hertz.
        amplitude: Target absolute peak for each generated block.
        block_size: Number of frames generated and written per stream write.
        pad: Silence duration in seconds before start and after stop.
        fade: Fade-in and fade-out duration in seconds.
        daemon: Whether the worker thread should be daemonized.

    Note:
        ``sample_rate`` and ``channels`` are read from the selected output
        device defaults. The stream implementation and random generator are
        intentionally internal to the thread.

    Raises:
        ValueError: If the selected device defaults, ``amplitude``,
            ``block_size``, ``pad``, ``fade``, or ``band`` are invalid.
    """

    def __init__(
        self,
        *,
        device: int | str | None = None,
        band: FrequencyBand | tuple[float, float] = FrequencyBand(),
        amplitude: float = 0.9,
        block_size: int = 1024,
        pad: float = 0.2,
        fade: float = 0.5,
        daemon: bool = True,
    ) -> None:
        super().__init__(daemon=daemon)
        self._validate_explicit_inputs(
            block_size=block_size,
            pad=pad,
            fade=fade,
        )
        self.device = device
        self.band = _coerce_band(band)
        self.amplitude = float(amplitude)
        self.sample_rate = 0
        self.channels = 0
        self.block_size = int(block_size)
        self.pad = float(pad)
        self.fade = float(fade)
        self.rng = np.random.default_rng()
        self.exception: BaseException | None = None

        self._refresh_output_device_defaults()
        self._validate()
        self._start_event = Event()
        self._stop_event = Event()
        self._close_event = Event()
        self._worker_started = False
        self._zi = pink_noise_zi(self.sample_rate, self.band)
        self._fade_in_position = 0

    def start(self) -> None:
        """Start the worker thread if needed and begin pink-noise playback."""
        if self._close_event.is_set():
            raise RuntimeError("PinkNoiseThread is closed")
        self.raise_if_failed()
        self._stop_event.clear()
        self._start_event.set()
        if not self._worker_started:
            self._worker_started = True
            super().start()

    def stop(self) -> None:
        """Request playback stop.

        The worker remains alive and waits for the next ``start()`` call.
        """
        self._start_event.clear()
        self._stop_event.set()

    def close(self, timeout: float | None = None) -> None:
        """Stop playback, terminate the worker thread, and wait for it."""
        self._close_event.set()
        self.stop()
        self._start_event.set()
        if self.is_alive():
            self.join(timeout=timeout)

    def raise_if_failed(self) -> None:
        """Raise an exception captured from the worker thread, if any."""
        if self.exception is not None:
            raise self.exception

    def run(self) -> None:
        """Wait for playback requests and write pink-noise blocks."""
        try:
            while not self._close_event.is_set():
                self._start_event.wait()
                if self._close_event.is_set():
                    break
                self._run_playback()
        except BaseException as exc:
            self.exception = exc

    def _run_playback(self) -> None:
        self._refresh_output_device_defaults()
        self._validate()
        self._reset_generation()
        self._stop_event.clear()
        with sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self.device,
            channels=self.channels,
            dtype="float32",
        ) as stream:
            self._write_silence(stream, self.pad_samples)
            while not self._stop_event.is_set() and not self._close_event.is_set():
                stream.write(self._apply_fade_in(self._next_block(self.block_size)))
            self._write_stop_tail(stream)
            self._drain_stream(stream)

    @property
    def pad_samples(self) -> int:
        """Silence padding length in samples."""
        return int(round(self.pad * self.sample_rate))

    @property
    def fade_samples(self) -> int:
        """Fade length in samples."""
        return int(round(self.fade * self.sample_rate))

    def _next_block(self, frames: int) -> NDArray[np.float32]:
        signal, zi = pink_noise(
            frames,
            self.sample_rate,
            self.band,
            amplitude=self.amplitude,
            channels=self.channels,
            rng=self.rng,
            pad=0,
            fade=0,
            zi=self._zi,
        )
        self._zi = zi
        return signal.as_array(np.float32)

    def _reset_generation(self) -> None:
        self._zi = pink_noise_zi(self.sample_rate, self.band)
        self._fade_in_position = 0

    def _refresh_output_device_defaults(self) -> None:
        device_info = self._query_output_device()
        self.sample_rate = self._resolve_sample_rate(device_info)
        self.channels = self._resolve_channels(device_info)

    def _apply_fade_in(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        if self.fade_samples <= 1 or self._fade_in_position >= self.fade_samples:
            return block

        positions = np.arange(
            self._fade_in_position,
            self._fade_in_position + block.shape[0],
            dtype=np.float32,
        )
        gains = np.clip(positions / float(self.fade_samples - 1), 0.0, 1.0)
        self._fade_in_position += block.shape[0]
        return cast(NDArray[np.float32], block * gains[:, None])

    def _write_stop_tail(self, stream: Any) -> None:
        tail = self._stop_tail()
        if tail.shape[0] > 0:
            stream.write(tail)

    def _stop_tail(self) -> NDArray[np.float32]:
        parts: list[NDArray[np.float32]] = []
        if self.fade_samples > 0:
            fade_block = self._next_block(self.fade_samples)
            gains = np.linspace(1.0, 0.0, self.fade_samples, dtype=np.float32)
            parts.append(cast(NDArray[np.float32], fade_block * gains[:, None]))
        if self.pad_samples > 0:
            parts.append(np.zeros((self.pad_samples, self.channels), dtype=np.float32))
        if not parts:
            return np.empty((0, self.channels), dtype=np.float32)
        return cast(NDArray[np.float32], np.vstack(parts))

    def _write_silence(self, stream: Any, frames: int) -> None:
        if frames <= 0:
            return
        stream.write(np.zeros((frames, self.channels), dtype=np.float32))

    @staticmethod
    def _drain_stream(stream: Any) -> None:
        stop = getattr(stream, "stop", None)
        if callable(stop):
            stop()

    def _validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.amplitude < 0.0:
            raise ValueError("Amplitude must not be negative")
        if self.channels < 1:
            raise ValueError("Channel count must be positive")
        if self.block_size <= 0:
            raise ValueError("Block size must be positive")
        if self.pad < 0.0:
            raise ValueError("Pad duration must not be negative")
        if self.fade < 0.0:
            raise ValueError("Fade duration must not be negative")
        self.band.validate(nyquist=self.sample_rate / 2)

    @staticmethod
    def _validate_explicit_inputs(
        *,
        block_size: int,
        pad: float,
        fade: float,
    ) -> None:
        if int(block_size) <= 0:
            raise ValueError("Block size must be positive")
        if float(pad) < 0.0:
            raise ValueError("Pad duration must not be negative")
        if float(fade) < 0.0:
            raise ValueError("Fade duration must not be negative")

    def _query_output_device(self) -> Mapping[str, Any]:
        try:
            device_info = sd.query_devices(self.device, kind="output")
        except (sd.PortAudioError, ValueError) as exc:
            raise ValueError("No such device") from exc
        if not isinstance(device_info, dict):
            raise ValueError("Could not query output device defaults")
        return cast(Mapping[str, Any], device_info)

    @staticmethod
    def _resolve_sample_rate(
        device_info: Mapping[str, Any],
    ) -> int:
        return int(round(float(device_info["default_samplerate"])))

    @staticmethod
    def _resolve_channels(
        device_info: Mapping[str, Any],
    ) -> int:
        return int(device_info["max_output_channels"])


def _coerce_band(band: FrequencyBand | tuple[float, float]) -> FrequencyBand:
    if isinstance(band, FrequencyBand):
        return band
    return FrequencyBand(float(band[0]), float(band[1]))
