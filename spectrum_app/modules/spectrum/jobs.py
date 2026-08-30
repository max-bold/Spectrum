from collections.abc import Callable
from dataclasses import dataclass
from threading import Condition, Event, Lock, Thread

import numpy as np
from scipy.signal import periodogram, welch

from audioanalysis import (
    ASignal,
    FrequencyBand,
    SpectrumConfig,
    SpectrumResult,
    analyze_spectrum,
    extend_log_sweep_band,
    log_chirp,
    pink_noise,
    power_db,
    smooth_power_spectrum,
)
from spectrum_app.core.audio import CLIPPING_THRESHOLD, AudioInput, AudioOutput


SnapshotCallback = Callable[[ASignal, ASignal], None]
LevelCallback = Callable[[np.ndarray, np.ndarray], None]
CompleteCallback = Callable[
    [ASignal | None, ASignal | None, np.ndarray, np.ndarray, str | None],
    None,
]
AnalysisResponse = tuple[int, bool, SpectrumResult | None, str | None]


@dataclass(frozen=True)
class BatchAnalysisRequest:
    revision: int
    final: bool
    signals: tuple[ASignal, ...]
    config: SpectrumConfig


@dataclass(frozen=True)
class OnlineAnalysisRequest:
    revision: int
    signal: ASignal
    generator: ASignal | None
    config: SpectrumConfig
    running_mean: bool


@dataclass(frozen=True)
class OnlineAnalysisStream:
    revision: int
    samples: int
    sample_rate: int
    generator: ASignal | None
    config: SpectrumConfig
    running_mean: bool


ONLINE_OVERLAP_DIVISOR = 4
HANN_SQUARED_OVERLAP_GAIN = 1.5
ONLINE_PUBLISH_SECONDS = 0.5


class SpectrumAcquisition(Thread):
    """Records input and plays a generated signal using blocking audio APIs."""

    LEVEL_UPDATE_SECONDS = 0.1

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        *,
        generator_mode: str,
        band: FrequencyBand,
        duration: float,
        pre_silence: float,
        post_silence: float,
        fade_in: float,
        fade_out: float,
        online_samples: int | None,
        on_level: LevelCallback,
        on_snapshot: SnapshotCallback,
        on_complete: CompleteCallback,
    ) -> None:
        super().__init__(name="spectrum-acquisition", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.generator_mode = generator_mode
        self.band = band
        self.duration = duration
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.online_samples = online_samples
        self.on_level = on_level
        self.on_snapshot = on_snapshot
        self.on_complete = on_complete
        self._stop_event = Event()
        self._writer_error: Exception | None = None
        self._writer_error_lock = Lock()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        chunks: list[np.ndarray] = []
        level_time: list[float] = []
        level_values: list[np.ndarray] = []
        input_rate = 0
        generator: ASignal | None = None
        writer: Thread | None = None
        error_message: str | None = None
        try:
            if self._stop_event.is_set():
                return
            if not self.audio_input.open():
                raise RuntimeError("Cannot open audio input")
            if not self.audio_output.open():
                raise RuntimeError("Cannot open audio output")
            if self._stop_event.is_set():
                return

            generator = self._generate_signal()
            writer = Thread(
                target=self._write_signal,
                args=(generator,),
                name="spectrum-output",
                daemon=True,
            )
            writer.start()

            input_rate = self.audio_input.sample_rate
            output_rate = self.audio_output.sample_rate
            target_samples = int(
                np.ceil(generator.sample_count * input_rate / output_rate)
            )
            level_update_samples = max(
                1,
                int(round(self.LEVEL_UPDATE_SECONDS * input_rate)),
            )
            next_level_update = 0
            recorded_samples = 0

            while recorded_samples < target_samples and not self._stop_event.is_set():
                writer_error = self._get_writer_error()
                if writer_error is not None:
                    raise writer_error
                block = self.audio_input.read(self.audio_input.blocksize)
                if len(block) == 0:
                    raise RuntimeError("Audio input returned no samples")
                remaining = target_samples - recorded_samples
                active = block[:remaining]
                if active.ndim != 2 or active.shape[1] != 2:
                    raise RuntimeError(
                        "Audio input must return logical channels A and B"
                    )
                peaks = np.max(np.abs(active), axis=0)
                clipped = [
                    label
                    for label, peak in zip(("A", "B"), peaks, strict=True)
                    if peak >= CLIPPING_THRESHOLD
                ]
                if clipped:
                    raise RuntimeError(
                        f"Input clipping detected on channel {'/'.join(clipped)}"
                    )
                chunks.append(active)
                recorded_samples += len(active)
                level_time.append(recorded_samples / input_rate)
                level_values.append(
                    np.max(np.abs(active[:, :2]), axis=0).astype(np.float64)
                )
                if recorded_samples >= next_level_update:
                    self.on_level(
                        np.asarray(level_time, dtype=np.float64),
                        np.vstack(level_values),
                    )
                    next_level_update = recorded_samples + level_update_samples
                if self.online_samples is not None:
                    self.on_snapshot(
                        ASignal(active.copy(), input_rate),
                        generator,
                    )

            if writer is not None:
                writer.join()
            writer_error = self._get_writer_error()
            if writer_error is not None and not self._stop_event.is_set():
                raise writer_error
        except Exception as error:
            if not self._stop_event.is_set():
                error_message = str(error)
        finally:
            self._stop_event.set()
            if writer is not None and writer.is_alive():
                writer.join()
            self.audio_output.close()
            self.audio_input.close()
            recording = (
                ASignal(np.concatenate(chunks, axis=0), input_rate) if chunks else None
            )
            times = np.asarray(level_time, dtype=np.float64)
            levels = (
                np.vstack(level_values)
                if level_values
                else np.empty((0, 0), dtype=np.float64)
            )
            if levels.size:
                self.on_level(times, levels)
            self.on_complete(recording, generator, times, levels, error_message)

    def _generate_signal(self) -> ASignal:
        sample_rate = self.audio_output.sample_rate
        active_duration = self.fade_in + self.duration + self.fade_out
        samples = max(1, int(round(active_duration * sample_rate)))
        if self.generator_mode == "log chirp":
            sweep_band = extend_log_sweep_band(
                self.band,
                self.duration,
                self.fade_in,
                self.fade_out,
            )
            sweep_band.validate(nyquist=sample_rate / 2.0)
            signal = log_chirp(
                samples,
                sample_rate,
                sweep_band,
            )
        elif self.generator_mode == "pink noise":
            signal, _ = pink_noise(
                samples,
                sample_rate,
                self.band,
            )
        else:
            raise ValueError(f"Unknown generator mode: {self.generator_mode}")
        signal = signal.fade(
            in_=int(round(self.fade_in * sample_rate)),
            out=int(round(self.fade_out * sample_rate)),
        )
        return signal.pad(
            in_=int(round(self.pre_silence * sample_rate)),
            out=int(round(self.post_silence * sample_rate)),
        )

    def _write_signal(self, generator: ASignal) -> None:
        try:
            data = generator.as_array()
            position = 0
            blocksize = self.audio_output.blocksize
            while position < len(data) and not self._stop_event.is_set():
                end = min(position + blocksize, len(data))
                samples = end - position
                block = np.zeros(blocksize, dtype=np.float32)
                block[:samples] = data[position:end, 0]
                self.audio_output.write(block)
                position = end
        except Exception as error:
            if not self._stop_event.is_set():
                with self._writer_error_lock:
                    self._writer_error = error

    def _get_writer_error(self) -> Exception | None:
        with self._writer_error_lock:
            return self._writer_error


class SpectrumAnalyzer(Thread):
    """Calculates final spectra and incrementally accumulates online Welch PSD."""

    def __init__(self) -> None:
        super().__init__(name="spectrum-analyzer", daemon=True)
        self._condition = Condition()
        self._stop_requested = False
        self._request: BatchAnalysisRequest | None = None
        self._response: AnalysisResponse | None = None
        self._online_stream: OnlineAnalysisStream | None = None
        self._online_buffer = np.empty((0, 2), dtype=np.float32)
        self._online_revision: int | None = None
        self._online_sum: np.ndarray | None = None
        self._online_count = 0
        self._generator_sum: np.ndarray | None = None
        self._generator_count = 0
        self._online_publish_every = 1

    def submit(
        self,
        revision: int,
        final: bool,
        signals: tuple[ASignal, ...],
        config: SpectrumConfig,
    ) -> None:
        if not signals:
            raise ValueError("Spectrum analysis requires at least one signal")
        with self._condition:
            # A final/reanalysis request supersedes any remaining online backlog.
            self._online_stream = None
            self._online_buffer = np.empty((0, 2), dtype=np.float32)
            self._request = BatchAnalysisRequest(
                revision,
                final,
                signals,
                config,
            )
            self._condition.notify()

    def feed_online(
        self,
        revision: int,
        signal: ASignal,
        samples: int,
        generator: ASignal | None,
        config: SpectrumConfig,
        *,
        running_mean: bool,
    ) -> None:
        """Append newly recorded samples to the single online analysis stream."""
        if samples <= 0:
            raise ValueError("Online Spectrum bucket size must be positive")
        data = signal.as_array(np.float32)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("Online Spectrum input must contain channels A and B")
        with self._condition:
            stream = self._online_stream
            if stream is None or stream.revision != revision:
                self._online_stream = OnlineAnalysisStream(
                    revision,
                    samples,
                    signal.sample_rate,
                    generator,
                    config,
                    running_mean,
                )
                self._online_buffer = data.copy()
            else:
                if stream.samples != samples:
                    raise ValueError("Online Spectrum bucket size changed")
                if stream.sample_rate != signal.sample_rate:
                    raise ValueError("Online Spectrum sample rate changed")
                self._online_buffer = np.concatenate(
                    (self._online_buffer, data),
                    axis=0,
                )
            self._condition.notify()

    def _next_online_request(self) -> OnlineAnalysisRequest | None:
        stream = self._online_stream
        if stream is None or len(self._online_buffer) < stream.samples:
            return None
        return OnlineAnalysisRequest(
            stream.revision,
            ASignal(
                self._online_buffer[: stream.samples].copy(),
                stream.sample_rate,
            ),
            stream.generator,
            stream.config,
            stream.running_mean,
        )

    def _online_ready(self) -> bool:
        stream = self._online_stream
        return stream is not None and len(self._online_buffer) >= stream.samples

    def _advance_online(self, revision: int) -> None:
        stream = self._online_stream
        if stream is None or stream.revision != revision:
            return
        hop = max(1, stream.samples // ONLINE_OVERLAP_DIVISOR)
        self._online_buffer = self._online_buffer[hop:]

    def _online_response(
        self,
        request: OnlineAnalysisRequest,
    ) -> AnalysisResponse | None:
        try:
            result = self._analyze_online(request)
        except Exception as error:
            return (request.revision, False, None, str(error))
        if result is None:
            return None
        return (
            request.revision,
            False,
            result,
            None,
        )

    def _batch_response(
        self,
        request: BatchAnalysisRequest,
    ) -> AnalysisResponse:
        try:
            result = self._analyze_batch(request)
            return (
                request.revision,
                request.final,
                result,
                None,
            )
        except Exception as error:
            return (request.revision, request.final, None, str(error))

    def poll(self) -> AnalysisResponse | None:
        with self._condition:
            response = self._response
            self._response = None
            return response

    def shutdown(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify()
        if self.is_alive():
            self.join()

    def run(self) -> None:
        while True:
            with self._condition:
                while (
                    self._request is None
                    and not self._online_ready()
                    and not self._stop_requested
                ):
                    self._condition.wait()
                if self._stop_requested:
                    return
                batch_request = self._request
                if batch_request is not None:
                    self._request = None
                    online_request = None
                else:
                    online_request = self._next_online_request()

            if batch_request is not None:
                response = self._batch_response(batch_request)
            elif online_request is not None:
                response = self._online_response(online_request)
            else:
                continue

            with self._condition:
                if online_request is not None:
                    self._advance_online(online_request.revision)
                if response is not None:
                    # Online display results are snapshots; the newest one wins.
                    self._response = response

    def _analyze_batch(self, request: BatchAnalysisRequest) -> SpectrumResult:
        results = [
            analyze_spectrum(signal, request.config) for signal in request.signals
        ]
        frequency = results[0].frequency
        if any(
            result.frequency.shape != frequency.shape
            or not np.allclose(result.frequency, frequency)
            for result in results[1:]
        ):
            raise ValueError("Spectrum frequency grids do not match")
        average_power = np.mean(
            np.stack([result.values for result in results]),
            axis=0,
        )
        return SpectrumResult(frequency, power_db(average_power))

    def _analyze_online(
        self,
        request: OnlineAnalysisRequest,
    ) -> SpectrumResult | None:
        frequency, bucket = _hann_periodogram(request.signal)
        if request.revision != self._online_revision:
            self._online_revision = request.revision
            self._online_sum = np.zeros_like(bucket)
            self._online_count = 0
            self._generator_sum = None
            self._generator_count = 0
            hop = max(
                1,
                request.signal.sample_count // ONLINE_OVERLAP_DIVISOR,
            )
            self._online_publish_every = max(
                1,
                int(round(ONLINE_PUBLISH_SECONDS * request.signal.sample_rate / hop)),
            )

        if self._online_sum is None:
            raise RuntimeError("Online Spectrum accumulator is unavailable")
        self._online_sum += bucket
        self._online_count += 1
        publish = (
            self._online_count == 1
            or self._online_count % self._online_publish_every == 0
        )
        if not publish:
            return None

        if request.generator is not None and self._generator_sum is None:
            (
                generator_frequency,
                self._generator_sum,
                self._generator_count,
            ) = _sum_hann_periodograms(
                request.generator,
                request.signal.sample_count,
            )
            if not np.allclose(generator_frequency, frequency):
                self._generator_sum = None
                self._generator_count = 0
                raise ValueError("Online generator frequency grid does not match")

        input_spectrum = self._normalize_online_sum(
            self._online_sum,
            self._online_count,
            request.running_mean,
        )
        if self._generator_sum is not None:
            generator_spectrum = self._normalize_online_sum(
                self._generator_sum,
                self._generator_count,
                request.running_mean,
            )
            input_spectrum = np.column_stack(
                (input_spectrum[:, 0], generator_spectrum[:, 0])
            )

        result = smooth_power_spectrum(
            frequency,
            input_spectrum,
            request.config,
        )
        return SpectrumResult(result.frequency, power_db(result.values))

    @staticmethod
    def _normalize_online_sum(
        spectrum_sum: np.ndarray,
        count: int,
        running_mean: bool,
    ) -> np.ndarray:
        if count <= 0:
            raise ValueError("Online Spectrum accumulator is empty")
        if running_mean:
            return spectrum_sum / count
        return spectrum_sum / HANN_SQUARED_OVERLAP_GAIN


def _hann_periodogram(
    signal: ASignal,
) -> tuple[np.ndarray, np.ndarray]:
    frequency, spectrum = periodogram(
        signal.as_array(np.float64),
        signal.sample_rate,
        window="hann",
        axis=0,
    )
    return (
        np.asarray(frequency, dtype=np.float64),
        np.asarray(spectrum, dtype=np.float64),
    )


def _sum_hann_periodograms(
    signal: ASignal,
    samples: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    if signal.sample_count < samples:
        raise ValueError("Generator is shorter than the online Welch bucket")
    hop = max(1, samples // ONLINE_OVERLAP_DIVISOR)
    count = 1 + (signal.sample_count - samples) // hop
    frequency, spectrum_mean = welch(
        signal.as_array(np.float64),
        signal.sample_rate,
        window="hann",
        nperseg=samples,
        noverlap=samples - hop,
        axis=0,
    )
    return (
        np.asarray(frequency, dtype=np.float64),
        np.asarray(spectrum_mean, dtype=np.float64) * count,
        count,
    )
