"""Signal generators.

Generator functions return mono ``ASignal`` instances ready for playback or
offline analysis. Callers can pass an explicit ``numpy.random.Generator`` when
deterministic noise is needed.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    chirp,
    sos2zpk,
    sosfilt,
    sosfilt_zi,
    sosfreqz,
    zpk2sos,
)

from .types import ASignal, FrequencyBand

PINKING_SOS = np.array(
    [
        [0.04992203, -0.00539063, 0.0, 1.0, -0.55594526, 0.0],
        [1.0, -1.81488818, 0.81786161, 1.0, -1.93901074, 0.93928204],
    ],
    dtype=np.float64,
)
PINKING_REFERENCE_SAMPLE_RATE = 44_100
PINKING_GAIN_REFERENCE_FREQUENCY = 1_000.0


def extend_log_sweep_band(
    band: FrequencyBand,
    duration: float,
    fade_in: float,
    fade_out: float,
) -> FrequencyBand:
    """Extend a logarithmic sweep so fades remain outside its working band."""
    band.validate()
    if duration <= 0.0:
        raise ValueError("Sweep duration must be positive")
    if fade_in < 0.0 or fade_out < 0.0:
        raise ValueError("Sweep fades must not be negative")
    frequency_ratio = band.high / band.low
    return FrequencyBand(
        band.low / frequency_ratio ** (fade_in / duration),
        band.high * frequency_ratio ** (fade_out / duration),
    )


def log_chirp(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand = FrequencyBand(),
    *,
    amplitude: float = 0.9,
) -> ASignal:
    """Generate a peak-normalized logarithmic chirp.

    The chirp sweeps exponentially from ``band.low`` to ``band.high`` over the
    requested number of samples. This is useful for frequency-response and
    impulse-response measurements because each octave receives comparable time
    coverage. Output is always mono. Use ``ASignal.fade()`` and
    ``ASignal.pad()`` when shaping is required.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz. The upper band edge must be below
            ``sample_rate / 2``.
        band: Frequency range in hertz.
        amplitude: Target absolute peak after normalization.

    Returns:
        An ``ASignal`` containing the generated chirp.

    Raises:
        ValueError: If ``samples`` is not positive, the frequency band is
            invalid, or the high band edge reaches Nyquist.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    band.validate(nyquist=sample_rate / 2)
    time = np.arange(samples, dtype=np.float64) / float(sample_rate)
    duration = samples / float(sample_rate)
    signal = chirp(
        time,
        band.low,
        duration,
        band.high,
        method="logarithmic",
        phi=-90,
    )
    return ASignal(signal, sample_rate).normalize(amplitude)


def white_noise(
    samples: int,
    sample_rate: int = 44_100,
    *,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
) -> ASignal:
    """Generate peak-normalized white noise.

    Samples are drawn from a uniform distribution over ``[-1.0, 1.0]`` and then
    normalized to the requested peak amplitude. Output is always mono. Pass a seeded
    ``numpy.random.Generator`` to make the output reproducible in tests or
    measurement scripts.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz.
        amplitude: Target absolute peak after normalization.
        rng: Optional random number generator. If omitted, a fresh default
            generator is created.

    Returns:
        An ``ASignal`` containing the generated white noise.

    Raises:
        ValueError: If ``samples`` is not positive.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    generator = rng or np.random.default_rng()
    signal = generator.uniform(-1.0, 1.0, samples)
    return ASignal(signal, sample_rate).normalize(amplitude)


def pink_noise(
    samples: int,
    sample_rate: int = 44_100,
    band: FrequencyBand | None = None,
    *,
    amplitude: float = 0.9,
    rng: np.random.Generator | None = None,
    zi: NDArray[np.float64] | None = None,
) -> tuple[ASignal, NDArray[np.float64]]:
    """Generate mono pink noise from amplitude-limited white noise.

    The function starts with uniform white noise bounded by ``amplitude``, then
    applies a sample-rate-adjusted pinking-filter cascade. When ``band`` is
    provided, a Butterworth band-pass filter is appended to the cascade. The
    filtered result is not normalized, so consecutive calls with filter state
    do not introduce gain changes at block boundaries.

    Args:
        samples: Number of samples to generate. Must be positive.
        sample_rate: Sampling rate in hertz. The upper band edge must be below
            ``sample_rate / 2``.
        band: Optional frequency range. No band-pass filter is applied when it
            is ``None``.
        amplitude: Absolute bound of the white-noise samples before filtering.
        rng: Optional random number generator. If omitted, a fresh default
            generator is created.
        zi: Optional SOS filter state for continuous generation. When omitted,
            the filter is initialized from ``scipy.signal.sosfilt_zi`` and the
            first white-noise sample.

    Returns:
        The generated signal and the final SOS state to pass into the next
        call.

    Raises:
        ValueError: If ``samples`` is not positive or the optional frequency
            band is invalid.
    """
    if samples <= 0:
        raise ValueError("Sample count must be positive")
    generator = rng or np.random.default_rng()
    white = generator.uniform(-amplitude, amplitude, samples)
    combined_sos = pinking_sos(sample_rate)
    if band is not None:
        band.validate(nyquist=sample_rate / 2)
        band_sos = cast(
            NDArray[np.float64],
            butter(
                4,
                band.as_tuple(),
                "bandpass",
                output="sos",
                fs=sample_rate,
            ),
        )
        combined_sos = cast(
            NDArray[np.float64],
            np.vstack((combined_sos, band_sos)),
        )
    if zi is None:
        zi = cast(NDArray[np.float64], sosfilt_zi(combined_sos) * white[0])

    pink, zf = sosfilt(combined_sos, white, zi=zi)
    return ASignal(pink, sample_rate), cast(NDArray[np.float64], zf)


_PINK_NOISE_THREAD_DISABLED = r'''
# Disabled until continuous pink-noise playback is redesigned and moved into
# an application module. This code is intentionally not part of audioanalysis.
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
        amplitude: Absolute bound of the white noise before pinking and
            band-pass filtering.
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
        """Request playback stop while keeping the worker available."""
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
            playback_started_at = time.monotonic()
            frames_written = 0
            self._write_silence(stream, self.pad_samples)
            frames_written += self.pad_samples
            while not self._stop_event.is_set() and not self._close_event.is_set():
                if not self._wait_for_buffer_room(playback_started_at, frames_written):
                    break
                block = self._apply_fade_in(self._next_block(self.block_size))
                stream.write(block)
                frames_written += block.shape[0]
            self._wait_for_buffer_room(
                playback_started_at,
                frames_written,
                stop_can_interrupt=False,
            )
            frames_written += self._write_stop_tail(stream)
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

    def _write_stop_tail(self, stream: Any) -> int:
        tail = self._stop_tail()
        if tail.shape[0] > 0:
            stream.write(tail)
        return tail.shape[0]

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

    def _wait_for_buffer_room(
        self,
        playback_started_at: float,
        frames_written: int,
        *,
        stop_can_interrupt: bool = True,
    ) -> bool:
        half_block = self.block_size / 2.0
        while not self._close_event.is_set():
            if stop_can_interrupt and self._stop_event.is_set():
                return False
            played_frames = (time.monotonic() - playback_started_at) * self.sample_rate
            queued_frames = frames_written - played_frames
            if queued_frames <= half_block:
                return True
            wait_seconds = (queued_frames - half_block) / self.sample_rate
            time.sleep(min(0.01, max(0.001, wait_seconds)))
        return not stop_can_interrupt

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
    def _resolve_sample_rate(device_info: Mapping[str, Any]) -> int:
        return int(round(float(device_info["default_samplerate"])))

    @staticmethod
    def _resolve_channels(device_info: Mapping[str, Any]) -> int:
        return int(device_info["max_output_channels"])


'''


def pinking_sos(sample_rate: int) -> NDArray[np.float64]:
    """Adapt the 44.1 kHz pinking filter to preserve its frequencies in hertz."""
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive")
    if sample_rate == PINKING_REFERENCE_SAMPLE_RATE:
        return PINKING_SOS.copy()

    raw_zeros, raw_poles, gain = sos2zpk(PINKING_SOS)
    zeros = np.asarray(raw_zeros, dtype=np.complex128)
    poles = np.asarray(raw_poles, dtype=np.complex128)
    rate_ratio = PINKING_REFERENCE_SAMPLE_RATE / float(sample_rate)
    adapted_zeros = _matched_z_roots(zeros, rate_ratio)
    adapted_poles = _matched_z_roots(poles, rate_ratio)
    adapted = cast(
        NDArray[np.float64],
        zpk2sos(adapted_zeros, adapted_poles, gain),
    )

    reference_frequency = min(
        PINKING_GAIN_REFERENCE_FREQUENCY,
        sample_rate / 4.0,
    )
    _, reference_response = sosfreqz(
        PINKING_SOS,
        worN=[reference_frequency],
        fs=PINKING_REFERENCE_SAMPLE_RATE,
    )
    _, adapted_response = sosfreqz(
        adapted,
        worN=[reference_frequency],
        fs=sample_rate,
    )
    reference_gain = np.abs(
        np.asarray(reference_response, dtype=np.complex128).reshape(-1)[0]
    )
    current_gain = np.abs(
        np.asarray(adapted_response, dtype=np.complex128).reshape(-1)[0]
    )
    adapted[0, :3] *= reference_gain / current_gain
    return adapted


def _matched_z_roots(
    roots: NDArray[np.complex128],
    rate_ratio: float,
) -> NDArray[np.complex128]:
    """Move digital roots while preserving their decay rates in seconds."""
    result = np.zeros_like(roots)
    nonzero = np.abs(roots) > np.finfo(np.float64).tiny
    result[nonzero] = np.exp(np.log(roots[nonzero]) * rate_ratio)
    return result
