from collections.abc import Callable
from threading import Condition, Event, Lock, Thread

import numpy as np

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
)
from spectrum_app.core.audio import CLIPPING_THRESHOLD, AudioInput, AudioOutput


SnapshotCallback = Callable[[ASignal, ASignal], None]
LevelCallback = Callable[[np.ndarray, np.ndarray], None]
CompleteCallback = Callable[
    [ASignal | None, ASignal | None, np.ndarray, np.ndarray, str | None],
    None,
]
AnalysisResponse = tuple[int, bool, SpectrumResult | None, str | None]


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
        online_interval: float | None,
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
        self.online_interval = online_interval
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
            update_samples = (
                max(1, int(self.online_interval * input_rate))
                if self.online_interval is not None
                else None
            )
            next_update = update_samples
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
                if (
                    next_update is not None
                    and update_samples is not None
                    and recorded_samples >= next_update
                ):
                    self.on_snapshot(
                        ASignal(np.concatenate(chunks, axis=0), input_rate),
                        generator,
                    )
                    next_update = recorded_samples + update_samples

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
    """Calculates only the newest requested spectrum without building a queue."""

    def __init__(self) -> None:
        super().__init__(name="spectrum-analyzer", daemon=True)
        self._condition = Condition()
        self._stop_requested = False
        self._request: tuple[int, bool, ASignal, SpectrumConfig] | None = None
        self._response: AnalysisResponse | None = None

    def submit(
        self,
        revision: int,
        final: bool,
        signal: ASignal,
        config: SpectrumConfig,
    ) -> None:
        with self._condition:
            self._request = (revision, final, signal, config)
            self._condition.notify()

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
                while self._request is None and not self._stop_requested:
                    self._condition.wait()
                if self._stop_requested:
                    return
                request = self._request
                self._request = None
            if request is None:
                continue

            revision, final, signal, config = request
            try:
                result = analyze_spectrum(signal, config)
                response: AnalysisResponse = (
                    revision,
                    final,
                    SpectrumResult(result.frequency, power_db(result.values)),
                    None,
                )
            except Exception as error:
                response = (revision, final, None, str(error))
            with self._condition:
                self._response = response
