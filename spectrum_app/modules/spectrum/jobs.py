from collections.abc import Callable
from threading import Condition, Event, Lock, Thread

import numpy as np

from audioanalysis import (
    ASignal,
    AnalysisMethod,
    FrequencyBand,
    SpectrumConfig,
    SpectrumResult,
    analyze_spectrum,
    log_chirp,
    pink_noise,
    power_db,
)
from spectrum_app.core.audio import AudioInput, AudioOutput


SnapshotCallback = Callable[[ASignal, ASignal], None]
CompleteCallback = Callable[[ASignal | None, ASignal | None, str | None], None]
AnalysisResponse = tuple[int, bool, SpectrumResult | None, str | None]


class SpectrumAcquisition(Thread):
    """Records input and plays a generated signal using blocking audio APIs."""

    LEADING_SILENCE_SECONDS = 0.2
    RECORDING_TAIL_SECONDS = 1.0

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        *,
        generator_mode: str,
        band: FrequencyBand,
        duration: float,
        online_interval: float | None,
        on_snapshot: SnapshotCallback,
        on_complete: CompleteCallback,
    ) -> None:
        super().__init__(name="spectrum-acquisition", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.generator_mode = generator_mode
        self.band = band
        self.duration = duration
        self.online_interval = online_interval
        self.on_snapshot = on_snapshot
        self.on_complete = on_complete
        self._stop_event = Event()
        self._writer_error: Exception | None = None
        self._writer_error_lock = Lock()

    def stop(self) -> None:
        self._stop_event.set()
        self.audio_input.close()
        self.audio_output.close()

    def run(self) -> None:
        chunks: list[np.ndarray] = []
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
                + self.RECORDING_TAIL_SECONDS * input_rate
            )
            update_samples = (
                max(1, int(self.online_interval * input_rate))
                if self.online_interval is not None
                else None
            )
            next_update = update_samples
            recorded_samples = 0

            while recorded_samples < target_samples and not self._stop_event.is_set():
                writer_error = self._get_writer_error()
                if writer_error is not None:
                    raise writer_error
                samples = min(self.audio_input.block_size, target_samples - recorded_samples)
                block = self.audio_input.read(samples)
                if len(block) == 0:
                    raise RuntimeError("Audio input returned no samples")
                chunks.append(block)
                recorded_samples += len(block)
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
            self.audio_output.close()
            self.audio_input.close()
            if writer is not None and writer.is_alive():
                writer.join(timeout=1.0)
            recording = (
                ASignal(np.concatenate(chunks, axis=0), input_rate)
                if chunks
                else None
            )
            self.on_complete(recording, generator, error_message)

    def _generate_signal(self) -> ASignal:
        sample_rate = self.audio_output.sample_rate
        samples = max(1, int(round(self.duration * sample_rate)))
        if self.generator_mode == "log chirp":
            signal = log_chirp(
                samples,
                sample_rate,
                self.band,
                channels=1,
                pad=0,
                fade=0,
            )
        elif self.generator_mode == "pink noise":
            signal = pink_noise(
                samples,
                sample_rate,
                self.band,
                channels=1,
                pad=0,
                fade=0,
            )
        else:
            raise ValueError(f"Unknown generator mode: {self.generator_mode}")
        leading_silence = int(round(self.LEADING_SILENCE_SECONDS * sample_rate))
        return signal.pad(in_=leading_silence)

    def _write_signal(self, generator: ASignal) -> None:
        try:
            data = generator.as_array()
            position = 0
            while position < len(data) and not self._stop_event.is_set():
                end = min(position + self.audio_output.block_size, len(data))
                block = np.repeat(
                    data[position:end],
                    self.audio_output.channels,
                    axis=1,
                )
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
