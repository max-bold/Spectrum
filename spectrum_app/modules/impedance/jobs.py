from collections.abc import Callable
from threading import Event, Lock, Thread

import numpy as np

from audioanalysis import ASignal
from spectrum_app.core.audio import CLIPPING_THRESHOLD, AudioInput, AudioOutput


LevelCallback = Callable[[tuple[float, float]], None]
CaptureCallback = Callable[[ASignal | None, str | None, bool], None]


class ImpedanceCapture(Thread):
    """Plays a mono signal and records two input channels."""

    def __init__(
        self,
        audio_input: AudioInput,
        audio_output: AudioOutput,
        signal: ASignal,
        *,
        recording_tail: float = 0.25,
        loop: bool = False,
        on_level: LevelCallback,
        on_complete: CaptureCallback,
    ) -> None:
        super().__init__(name="impedance-capture", daemon=True)
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.signal = signal
        self.recording_tail = recording_tail
        self.loop = loop
        self.on_level = on_level
        self.on_complete = on_complete
        self._stop_event = Event()
        self._writer_error: Exception | None = None
        self._writer_lock = Lock()

    def request_stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        chunks: list[np.ndarray] = []
        writer: Thread | None = None
        error_message: str | None = None
        input_rate = 0
        try:
            if not self.audio_input.open():
                raise RuntimeError("Cannot open audio input")
            if not self.audio_output.open():
                raise RuntimeError("Cannot open audio output")
            input_rate = self.audio_input.sample_rate
            if input_rate != self.audio_output.sample_rate:
                raise RuntimeError("Input and output sample rates must be equal")
            if self.signal.sample_rate != input_rate:
                raise RuntimeError("Generated signal sample rate does not match audio device")

            writer = Thread(
                target=self._write_signal,
                name="impedance-output",
                daemon=True,
            )
            writer.start()
            target_samples = self.signal.sample_count + int(
                round(self.recording_tail * input_rate)
            )
            recorded = 0
            while not self._stop_event.is_set() and (
                self.loop or recorded < target_samples
            ):
                writer_error = self._get_writer_error()
                if writer_error is not None:
                    raise writer_error
                block = self.audio_input.read(self.audio_input.blocksize)
                if len(block) == 0:
                    raise RuntimeError("Audio input returned no samples")
                remaining = len(block) if self.loop else target_samples - recorded
                active = np.asarray(block[:remaining, :2], dtype=np.float32)
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
                recorded += len(active)
                self.on_level(
                    (
                        float(np.max(np.abs(active[:, 0]))),
                        float(np.max(np.abs(active[:, 1]))),
                    )
                )

            if writer is not None and not self.loop:
                writer.join()
            writer_error = self._get_writer_error()
            if writer_error is not None and not self._stop_event.is_set():
                raise writer_error
        except Exception as error:
            if not self._stop_event.is_set():
                error_message = str(error)
        finally:
            cancelled = self._stop_event.is_set()
            self._stop_event.set()
            if writer is not None and writer.is_alive():
                writer.join()
            self.audio_output.close()
            self.audio_input.close()
            recording = (
                ASignal(np.concatenate(chunks, axis=0), input_rate)
                if chunks and not self.loop
                else None
            )
            self.on_complete(recording, error_message, cancelled)

    def _write_signal(self) -> None:
        try:
            data = self.signal.as_array(np.float32)[:, 0]
            if not len(data):
                raise RuntimeError("Generated signal is empty")
            position = 0
            blocksize = self.audio_output.blocksize
            while not self._stop_event.is_set():
                block = np.zeros(blocksize, dtype=np.float32)
                filled = 0
                while filled < blocksize and not self._stop_event.is_set():
                    if position >= len(data):
                        if not self.loop:
                            break
                        position = 0
                    samples = min(blocksize - filled, len(data) - position)
                    block[filled : filled + samples] = data[
                        position : position + samples
                    ]
                    filled += samples
                    position += samples
                if filled == 0 or self._stop_event.is_set():
                    return
                self.audio_output.write(block)
                if not self.loop and position >= len(data):
                    return
        except Exception as error:
            if not self._stop_event.is_set():
                with self._writer_lock:
                    self._writer_error = error

    def _get_writer_error(self) -> Exception | None:
        with self._writer_lock:
            return self._writer_error
