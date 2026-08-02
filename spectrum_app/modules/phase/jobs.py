from collections.abc import Callable
from threading import Condition, Event, Lock, Thread

import numpy as np

from audioanalysis import (
    ASignal,
    FrequencyBand,
    PhaseConfig,
    PhaseResult,
    analyze_phase,
    log_chirp,
)
from spectrum_app.core.audio import AudioInput, AudioOutput

LevelCallback = Callable[[np.ndarray, np.ndarray, tuple[float, float]], None]
CompleteCallback = Callable[
    [ASignal | None, ASignal | None, np.ndarray, np.ndarray, str | None, bool],
    None,
]
AnalysisResponse = tuple[int, bool, PhaseResult | None, str | None]


class PhaseAcquisition(Thread):
    """Record A/B while a mono logarithmic sweep is written independently."""

    LEVEL_UPDATE_SECONDS = 0.1
    OUTPUT_LEVEL = 0.9

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        *,
        band: FrequencyBand,
        duration: float,
        pre_silence: float,
        post_silence: float,
        fade: float,
        on_level: LevelCallback,
        on_complete: CompleteCallback,
    ) -> None:
        super().__init__(name="phase-acquisition", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.band = band
        self.duration = duration
        self.pre_silence = pre_silence
        self.post_silence = post_silence
        self.fade = fade
        self.on_level = on_level
        self.on_complete = on_complete
        self._stop_event = Event()
        self._writer_error: Exception | None = None
        self._writer_lock = Lock()

    def stop(self) -> None:
        self._stop_event.set()
        self.audio_input.close()
        self.audio_output.close()

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
            input_rate = self.audio_input.sample_rate
            generator = self._generate_signal()
            writer = Thread(
                target=self._write_signal,
                args=(generator,),
                name="phase-output",
                daemon=True,
            )
            writer.start()

            target_samples = int(
                round(
                    (self.pre_silence + self.duration + self.post_silence) * input_rate
                )
            )
            update_samples = max(
                1,
                int(round(self.LEVEL_UPDATE_SECONDS * input_rate)),
            )
            next_update = 0
            recorded = 0
            while recorded < target_samples and not self._stop_event.is_set():
                writer_error = self._get_writer_error()
                if writer_error is not None:
                    raise writer_error
                samples = min(self.audio_input.block_size, target_samples - recorded)
                block = np.asarray(self.audio_input.read(samples), dtype=np.float32)
                if block.ndim != 2 or block.shape[1] != 2:
                    raise RuntimeError(
                        "Audio input must return logical channels A and B"
                    )
                if len(block) == 0:
                    raise RuntimeError("Audio input returned no samples")
                chunks.append(block)
                recorded += len(block)
                level_time.append(recorded / input_rate)
                level_values.append(np.max(np.abs(block), axis=0).astype(np.float64))
                if recorded >= next_update:
                    levels = np.vstack(level_values)
                    self.on_level(
                        np.asarray(level_time, dtype=np.float64),
                        levels,
                        (float(levels[-1, 0]), float(levels[-1, 1])),
                    )
                    next_update = recorded + update_samples

            if writer is not None:
                writer.join(timeout=2.0)
                if writer.is_alive() and not self._stop_event.is_set():
                    raise RuntimeError("Audio output did not stop")
            writer_error = self._get_writer_error()
            if writer_error is not None and not self._stop_event.is_set():
                raise writer_error
        except Exception as error:
            if not self._stop_event.is_set():
                error_message = str(error) or error.__class__.__name__
        finally:
            cancelled = self._stop_event.is_set()
            self._stop_event.set()
            self.audio_output.close()
            self.audio_input.close()
            if writer is not None and writer.is_alive():
                writer.join(timeout=1.0)
            times = np.asarray(level_time, dtype=np.float64)
            levels = (
                np.vstack(level_values)
                if level_values
                else np.empty((0, 2), dtype=np.float64)
            )
            if levels.size:
                self.on_level(
                    times,
                    levels,
                    (float(levels[-1, 0]), float(levels[-1, 1])),
                )
            recording = (
                ASignal(np.concatenate(chunks, axis=0), input_rate)
                if chunks and input_rate > 0
                else None
            )
            self.on_complete(
                recording,
                generator,
                times,
                levels,
                error_message,
                cancelled,
            )

    def _generate_signal(self) -> ASignal:
        sample_rate = self.audio_output.sample_rate
        sweep_samples = max(1, int(round(self.duration * sample_rate)))
        fade_samples = min(
            sweep_samples // 2,
            max(0, int(round(self.fade * sample_rate))),
        )
        sweep = log_chirp(
            sweep_samples,
            sample_rate,
            self.band,
            amplitude=self.OUTPUT_LEVEL,
            channels=1,
            pad=0,
            fade=fade_samples,
        )
        return sweep.pad(
            in_=int(round(self.pre_silence * sample_rate)),
            out=int(round(self.post_silence * sample_rate)),
        )

    def _write_signal(self, generator: ASignal) -> None:
        try:
            data = generator.as_array(np.float32)[:, 0]
            position = 0
            while position < len(data) and not self._stop_event.is_set():
                end = min(position + self.audio_output.block_size, len(data))
                self.audio_output.write(data[position:end])
                position = end
        except Exception as error:
            if not self._stop_event.is_set():
                with self._writer_lock:
                    self._writer_error = error

    def _get_writer_error(self) -> Exception | None:
        with self._writer_lock:
            return self._writer_error


class PhaseAnalyzer(Thread):
    """Calculate only the newest requested phase response off the UI thread."""

    def __init__(self) -> None:
        super().__init__(name="phase-analyzer", daemon=True)
        self._condition = Condition()
        self._stop_requested = False
        self._request: tuple[int, bool, ASignal, PhaseConfig] | None = None
        self._response: AnalysisResponse | None = None

    def submit(
        self,
        revision: int,
        finishing: bool,
        recording: ASignal,
        config: PhaseConfig,
    ) -> None:
        with self._condition:
            self._request = (revision, finishing, recording, config)
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
            self._condition.notify()
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
            revision, finishing, recording, config = request
            try:
                response: AnalysisResponse = (
                    revision,
                    finishing,
                    analyze_phase(recording, config),
                    None,
                )
            except Exception as error:
                response = (
                    revision,
                    finishing,
                    None,
                    str(error) or error.__class__.__name__,
                )
            with self._condition:
                self._response = response
