from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Event, Thread

import numpy as np
from numpy.typing import NDArray

from audioanalysis import (
    ASignal,
    FrequencyBand,
    RTAConfig,
    RTAResult,
    analyze_rta,
    pink_noise,
)
from spectrum_app.core.audio import CLIPPING_THRESHOLD, AudioInput, AudioOutput

LevelCallback = Callable[[tuple[float, float]], None]
SnapshotCallback = Callable[[ASignal], None]
CompleteCallback = Callable[[str | None, bool], None]
ClippingCallback = Callable[[str], None]
AnalysisResponse = tuple[int, ASignal, RTAResult | None, str | None]


@dataclass(frozen=True)
class RTARuntimeConfig:
    band: FrequencyBand
    noise: bool
    level_db: float
    window_seconds: float
    hop_seconds: float
    pre_silence: float
    fade_in: float
    fade_out: float


class RTANoiseGenerator(Thread):
    """Generate one continuous, faded pink-noise stream ahead of the I/O worker."""

    BASE_AMPLITUDE = 0.9

    def __init__(
        self,
        *,
        sample_rate: int,
        block_size: int,
        band: FrequencyBand,
        level_db: float,
        pre_silence: float,
        fade_in: float,
        fade_out: float,
        on_clipping: ClippingCallback,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(name="rta-generator", daemon=True)
        self.sample_rate = sample_rate
        self.block_size = max(1, int(block_size))
        self.band = band
        self.level_gain = 10.0 ** (float(level_db) / 20.0)
        self.pre_samples = max(0, int(round(pre_silence * sample_rate)))
        self.fade_in_samples = max(0, int(round(fade_in * sample_rate)))
        self.fade_out_samples = max(0, int(round(fade_out * sample_rate)))
        self.on_clipping = on_clipping
        self._condition = Condition()
        self._block: NDArray[np.float32] | None = None
        self._finished = False
        self._stop_requested = False
        self._abort_requested = False
        self._error: BaseException | None = None
        self._zi: NDArray[np.float64] | None = None
        self._rng = rng or np.random.default_rng()
        self._pre_position = 0
        self._noise_position = 0
        self._envelope_gain = 0.0 if self.fade_in_samples else 1.0
        self._fade_out_position: int | None = None
        self._fade_out_start_gain = 0.0
        self._clipping_reported = False

    def request_stop(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()

    def abort(self) -> None:
        with self._condition:
            self._abort_requested = True
            self._block = None
            self._condition.notify_all()

    def take(self) -> NDArray[np.float32] | None:
        with self._condition:
            while (
                self._block is None
                and not self._finished
                and self._error is None
                and not self._abort_requested
            ):
                self._condition.wait()
            if self._error is not None:
                raise RuntimeError(str(self._error) or self._error.__class__.__name__)
            block = self._block
            self._block = None
            self._condition.notify_all()
            return block

    def run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._block is not None and not self._abort_requested:
                        self._condition.wait()
                    if self._abort_requested:
                        return
                    block = self._generate_block()
                    if block is None:
                        self._finished = True
                        self._condition.notify_all()
                        return
                    self._block = block
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def _generate_block(self) -> NDArray[np.float32] | None:
        if self._stop_requested and self._pre_position < self.pre_samples:
            return None
        if self._pre_position < self.pre_samples:
            samples = min(self.block_size, self.pre_samples - self._pre_position)
            self._pre_position += samples
            return np.zeros(samples, dtype=np.float32)

        if self._stop_requested and self._fade_out_position is None:
            if self.fade_out_samples <= 0 or self._envelope_gain <= 0.0:
                return None
            self._fade_out_position = 0
            self._fade_out_start_gain = self._envelope_gain

        samples = self.block_size
        if self._fade_out_position is not None:
            remaining = self.fade_out_samples - self._fade_out_position
            if remaining <= 0:
                return None
            samples = min(samples, remaining)

        signal, self._zi = pink_noise(
            samples,
            self.sample_rate,
            self.band,
            amplitude=self.BASE_AMPLITUDE,
            rng=self._rng,
            zi=self._zi,
        )
        data = signal.as_array(np.float64)[:, 0]
        envelope = self._envelope(samples)
        output = data * envelope * self.level_gain
        if np.any(np.abs(output) >= CLIPPING_THRESHOLD):
            if not self._clipping_reported:
                self._clipping_reported = True
                message = "Output clipping detected; reduce the RTA noise level"
                self.on_clipping(message)
                raise RuntimeError(message)
        return np.asarray(output, dtype=np.float32)

    def _envelope(self, samples: int) -> NDArray[np.float64]:
        if self._fade_out_position is not None:
            start = self._fade_out_position
            positions = np.arange(start, start + samples, dtype=np.float64)
            if self.fade_out_samples <= 1:
                gains = np.zeros(samples, dtype=np.float64)
            else:
                gains = self._fade_out_start_gain * np.clip(
                    1.0 - positions / (self.fade_out_samples - 1),
                    0.0,
                    1.0,
                )
            self._fade_out_position += samples
            self._envelope_gain = float(gains[-1]) if len(gains) else 0.0
            return gains

        start = self._noise_position
        positions = np.arange(start, start + samples, dtype=np.float64)
        if self.fade_in_samples <= 1:
            gains = np.ones(samples, dtype=np.float64)
        else:
            gains = np.clip(positions / (self.fade_in_samples - 1), 0.0, 1.0)
        self._noise_position += samples
        self._envelope_gain = float(gains[-1]) if len(gains) else 1.0
        return gains


class RTAIOWorker(Thread):
    """Own blocking audio calls and publish rolling input windows."""

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        config: RTARuntimeConfig,
        *,
        on_level: LevelCallback,
        on_snapshot: SnapshotCallback,
        on_complete: CompleteCallback,
        on_clipping: ClippingCallback,
    ) -> None:
        super().__init__(name="rta-io", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.config = config
        self.on_level = on_level
        self.on_snapshot = on_snapshot
        self.on_complete = on_complete
        self.on_clipping = on_clipping
        self._stop_requested = Event()
        self._abort_requested = Event()
        self._generator: RTANoiseGenerator | None = None
        self._input_rate = 0
        self._window_samples = 0
        self._hop_samples = 0
        self._buffer = np.empty((0, 2), dtype=np.float32)
        self._samples_since_snapshot = 0
        self._snapshot_sent = False

    def request_stop(self) -> None:
        self._stop_requested.set()
        if self._generator is not None:
            self._generator.request_stop()

    def abort(self) -> None:
        self._abort_requested.set()
        self._stop_requested.set()
        if self._generator is not None:
            self._generator.abort()
        self.audio_input.close()
        self.audio_output.close()

    def run(self) -> None:
        error_message: str | None = None
        try:
            if not self.audio_input.open():
                raise RuntimeError("Cannot open audio input")
            self._input_rate = self.audio_input.sample_rate
            if self._input_rate <= 0:
                raise RuntimeError("Audio input sample rate is unavailable")
            self._window_samples = max(
                2,
                int(round(self.config.window_seconds * self._input_rate)),
            )
            self._hop_samples = max(
                1,
                int(round(self.config.hop_seconds * self._input_rate)),
            )

            if self.config.noise:
                self._run_with_output()
            else:
                self._run_input_only()
        except Exception as error:
            if not self._abort_requested.is_set():
                error_message = str(error) or error.__class__.__name__
        finally:
            generator = self._generator
            if generator is not None:
                generator.abort()
                if generator.is_alive():
                    generator.join(timeout=1.0)
            self.audio_output.close()
            self.audio_input.close()
            self.on_complete(error_message, self._stop_requested.is_set())

    def _run_with_output(self) -> None:
        if not self.audio_output.open():
            raise RuntimeError("Cannot open audio output")
        output_rate = self.audio_output.sample_rate
        if output_rate <= 0:
            raise RuntimeError("Audio output sample rate is unavailable")
        self.config.band.validate(nyquist=output_rate / 2.0)
        generator = RTANoiseGenerator(
            sample_rate=output_rate,
            block_size=self.audio_output.blocksize,
            band=self.config.band,
            level_db=self.config.level_db,
            pre_silence=self.config.pre_silence,
            fade_in=self.config.fade_in,
            fade_out=self.config.fade_out,
            on_clipping=self.on_clipping,
        )
        self._generator = generator
        generator.start()

        while not self._abort_requested.is_set():
            if self._stop_requested.is_set():
                generator.request_stop()
            output = generator.take()
            if output is None:
                break
            blocksize = self.audio_output.blocksize
            if len(output) != blocksize:
                padded = np.zeros(blocksize, dtype=np.float32)
                padded[: len(output)] = output
                output = padded
            self.audio_output.write(output)
            block = self.audio_input.read(self.audio_input.blocksize)
            self._consume_input(block)

    def _run_input_only(self) -> None:
        while not self._stop_requested.is_set() and not self._abort_requested.is_set():
            self._consume_input(self.audio_input.read(self.audio_input.blocksize))

    def _consume_input(self, data: NDArray[np.float32]) -> None:
        block = np.asarray(data, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != 2:
            raise RuntimeError("Audio input must return logical channels A and B")
        if not len(block):
            raise RuntimeError("Audio input returned no samples")
        peaks = np.max(np.abs(block), axis=0)
        clipped = [
            label
            for label, peak in zip(("A", "B"), peaks, strict=True)
            if peak >= CLIPPING_THRESHOLD
        ]
        if clipped:
            message = f"Input clipping detected on channel {'/'.join(clipped)}"
            self._stop_requested.set()
            self.on_clipping(message)
            raise RuntimeError(message)
        self.on_level((float(peaks[0]), float(peaks[1])))
        if self._stop_requested.is_set():
            return

        self._buffer = np.concatenate((self._buffer, block), axis=0)
        if len(self._buffer) > self._window_samples:
            self._buffer = self._buffer[-self._window_samples :]
        self._samples_since_snapshot += len(block)
        ready = len(self._buffer) >= self._window_samples
        due = (
            not self._snapshot_sent or self._samples_since_snapshot >= self._hop_samples
        )
        if ready and due:
            self._snapshot_sent = True
            self._samples_since_snapshot = 0
            self.on_snapshot(ASignal(self._buffer.copy(), self._input_rate))


class RTAAnalyzer(Thread):
    """Analyze only the newest rolling-window snapshot."""

    def __init__(self) -> None:
        super().__init__(name="rta-analyzer", daemon=True)
        self._condition = Condition()
        self._stop_requested = False
        self._request: tuple[int, ASignal, RTAConfig] | None = None
        self._response: AnalysisResponse | None = None

    def submit(self, revision: int, recording: ASignal, config: RTAConfig) -> None:
        with self._condition:
            self._request = (revision, recording, config)
            self._condition.notify()

    def poll(self) -> AnalysisResponse | None:
        with self._condition:
            response = self._response
            self._response = None
            return response

    def shutdown(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stop_requested = True
            self._request = None
            self._condition.notify_all()
        if self.is_alive():
            self.join(timeout=timeout)

    def run(self) -> None:
        while True:
            with self._condition:
                while self._request is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return
                request = self._request
                self._request = None
            if request is None:
                continue
            revision, recording, config = request
            try:
                response: AnalysisResponse = (
                    revision,
                    recording,
                    analyze_rta(recording, config),
                    None,
                )
            except Exception as error:
                response = (
                    revision,
                    recording,
                    None,
                    str(error) or error.__class__.__name__,
                )
            with self._condition:
                self._response = response
