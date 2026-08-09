from collections.abc import Callable
from threading import Condition, Event, Lock, Thread

import numpy as np

from audioanalysis import (
    ASignal,
    SemiAnalogTHDConfig,
    SemiAnalogTHDResult,
    THDMaskFit,
    analyze_semi_analog_thd,
    calibrate_semi_analog_thd_mask,
    generate_semi_analog_thd_sweep,
)
from spectrum_app.core.audio import CLIPPING_THRESHOLD, AudioInput, AudioOutput


LevelCallback = Callable[[np.ndarray, np.ndarray, float], None]
CompleteCallback = Callable[
    [ASignal | None, ASignal | None, np.ndarray, np.ndarray, str | None, bool],
    None,
]
Signature = tuple[object, ...]
AnalysisResponse = tuple[
    int,
    bool,
    SemiAnalogTHDResult | None,
    THDMaskFit | None,
    Signature,
    Signature,
    str | None,
]


class THDAcquisition(Thread):
    """Play a swept sine and record logical input A on independent I/O paths."""

    LEADING_SILENCE_SECONDS = 0.2
    RECORDING_TAIL_SECONDS = 0.5
    LEVEL_UPDATE_SECONDS = 0.1

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        output_config: SemiAnalogTHDConfig,
        *,
        on_level: LevelCallback,
        on_complete: CompleteCallback,
    ) -> None:
        super().__init__(name="thd-acquisition", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.output_config = output_config
        self.on_level = on_level
        self.on_complete = on_complete
        self._stop_event = Event()
        self._writer_error: Exception | None = None
        self._writer_lock = Lock()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        chunks: list[np.ndarray] = []
        level_time: list[float] = []
        level_values: list[float] = []
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

            input_rate = self.audio_input.sample_rate
            generator = generate_semi_analog_thd_sweep(self.output_config)
            writer = Thread(
                target=self._write_signal,
                args=(generator,),
                name="thd-output",
                daemon=True,
            )
            writer.start()

            target_samples = int(
                round(
                    (
                        self.LEADING_SILENCE_SECONDS
                        + self.output_config.duration
                        + self.RECORDING_TAIL_SECONDS
                    )
                    * input_rate
                )
            )
            next_level_update = 0
            level_update_samples = max(
                1,
                int(round(self.LEVEL_UPDATE_SECONDS * input_rate)),
            )
            recorded = 0
            while recorded < target_samples and not self._stop_event.is_set():
                writer_error = self._get_writer_error()
                if writer_error is not None:
                    raise writer_error
                block = self.audio_input.read(self.audio_input.blocksize)
                if len(block) == 0:
                    raise RuntimeError("Audio input returned no samples")
                remaining = target_samples - recorded
                channel_1 = np.asarray(block[:remaining, [0]], dtype=np.float32)
                if float(np.max(np.abs(channel_1))) >= CLIPPING_THRESHOLD:
                    raise RuntimeError("Input clipping detected on channel A")
                chunks.append(channel_1)
                recorded += len(channel_1)
                level_time.append(recorded / input_rate)
                level_values.append(float(np.max(np.abs(channel_1[:, 0]))))
                if recorded >= next_level_update:
                    self.on_level(
                        np.asarray(level_time, dtype=np.float64),
                        np.asarray(level_values, dtype=np.float64),
                        level_values[-1],
                    )
                    next_level_update = recorded + level_update_samples

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
            if writer is not None and writer.is_alive():
                writer.join()
            self.audio_output.close()
            self.audio_input.close()
            times = np.asarray(level_time, dtype=np.float64)
            levels = np.asarray(level_values, dtype=np.float64)
            if levels.size:
                self.on_level(times, levels, float(levels[-1]))
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

    def _write_signal(self, generator: ASignal) -> None:
        try:
            leading = int(
                round(self.LEADING_SILENCE_SECONDS * self.audio_output.sample_rate)
            )
            self._write_array(
                np.zeros((leading, 1), dtype=np.float32),
            )
            self._write_array(generator.as_array(np.float32))
        except Exception as error:
            if not self._stop_event.is_set():
                with self._writer_lock:
                    self._writer_error = error

    def _write_array(self, data: np.ndarray) -> None:
        position = 0
        blocksize = self.audio_output.blocksize
        while position < len(data) and not self._stop_event.is_set():
            end = min(position + blocksize, len(data))
            samples = end - position
            block = np.zeros(blocksize, dtype=np.float32)
            block[:samples] = data[position:end, 0]
            self.audio_output.write(block)
            position = end

    def _get_writer_error(self) -> Exception | None:
        with self._writer_lock:
            return self._writer_error


class THDAnalyzer(Thread):
    """Run only the newest requested THD calculation outside the UI thread."""

    def __init__(self) -> None:
        super().__init__(name="thd-analyzer", daemon=True)
        self._condition = Condition()
        self._stop_requested = False
        self._request: (
            tuple[
                int,
                bool,
                ASignal,
                SemiAnalogTHDConfig,
                THDMaskFit | None,
                Signature,
                Signature,
            ]
            | None
        ) = None
        self._response: AnalysisResponse | None = None

    def submit(
        self,
        revision: int,
        finishing: bool,
        recording: ASignal,
        config: SemiAnalogTHDConfig,
        mask_fit: THDMaskFit | None,
        mask_signature: Signature,
        analysis_signature: Signature,
    ) -> None:
        with self._condition:
            self._request = (
                revision,
                finishing,
                recording,
                config,
                mask_fit,
                mask_signature,
                analysis_signature,
            )
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

            (
                revision,
                finishing,
                recording,
                config,
                mask_fit,
                mask_signature,
                analysis_signature,
            ) = request
            try:
                fitted_mask = mask_fit or calibrate_semi_analog_thd_mask(config)
                result = analyze_semi_analog_thd(
                    recording,
                    config,
                    mask_fit=fitted_mask,
                )
                response: AnalysisResponse = (
                    revision,
                    finishing,
                    result,
                    fitted_mask,
                    mask_signature,
                    analysis_signature,
                    None,
                )
            except Exception as error:
                response = (
                    revision,
                    finishing,
                    None,
                    None,
                    mask_signature,
                    analysis_signature,
                    str(error) or error.__class__.__name__,
                )
            with self._condition:
                self._response = response
