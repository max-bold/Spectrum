import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import numpy as np
from threading import Thread, Lock, Event
from queue import Empty, Full
from time import sleep, time
from typing import Callable, Literal
from numpy.typing import NDArray
from abc import ABC
from queue import Queue
from scipy.signal import chirp, butter, sosfilt, sosfilt_zi, welch, periodogram
import sounddevice as sd


class AnalyserPipeline(Thread):
    def __init__(self) -> None:
        # Generator params
        self.length: float = 10  # in seconds
        self.band: tuple[float, float] = (20, 20e3)
        self.gen_mode: Literal["pink noise", "log sweep"] = "log sweep"
        self.output_queue: Queue[NDArray[np.float64]] = Queue(10)
        self.gen_running = Event()
        self.end_padding: float = 2.0  # seconds

        # AudioIO  params
        self.sample_rate: int = 96000
        self.chunk_size: int = 4096
        self.device: tuple[int, int] | None = None
        self.input_queue: Queue[NDArray[np.float64]] = Queue(10)
        self.audio_running = Event()
        self.audio_mode: Literal["normal", "silent"] = "normal"
        self.stream: sd.Stream | None = None

        # Recorder params
        self.record = np.empty((0, 2), dtype=np.float64)
        self.record_lock = Lock()
        self.recorder_running = Event()
        self.levels = np.empty((0, 2), dtype=np.float64)
        self.levels_lock = Lock()

        # Analyzer params
        self.analyzer_mode: Literal["rta", "recording"] = "recording"
        self.ref: Literal["none", "channel B", "generator"] = "channel B"
        self.weighting: None | Literal["pink"] = None
        self.rta_bucket_size: int = int(0.5 * self.sample_rate)
        self.freq_length = 1024
        self.window_width = 1 / 10
        self.fft_result = np.empty((2, 0), np.float64)
        self.fft_result_lock = Lock()

        # Pipeline params
        self.stop_flag = Event()
        self.run_flag = Event()

        return super().__init__()

    def pink_noise_gen(self):
        band_sos = np.array(
            butter(4, self.band, "bandpass", False, "sos", self.sample_rate), np.float64
        )
        pinking_sos = np.array(
            [
                [0.04992203, -0.00539063, 0.0, 1.0, -0.55594526, 0.0],
                [1.0, -1.81488818, 0.81786161, 1.0, -1.93901074, 0.93928204],
            ],
            np.float64,
        )
        combined_sos = np.vstack([pinking_sos, band_sos])
        zi = sosfilt_zi(combined_sos)
        self.gen_running.set()
        n = int(self.length * self.sample_rate // self.chunk_size)
        for i in range(n):
            if not self.run_flag.is_set():
                break
            white = np.random.uniform(-1, 1, self.chunk_size)
            pink, zi = sosfilt(combined_sos, white, -1, zi)
            pink = np.array(pink, np.float64)
            chunk = np.column_stack((pink, pink))
            self.output_queue.put(chunk)
        for i in range(int((self.end_padding * self.sample_rate) // self.chunk_size)):
            if not self.run_flag.is_set():
                break
            zeros = np.zeros((self.chunk_size, 2))
            self.output_queue.put(zeros)
        self.gen_running.clear()

    def log_sweep_gen(self):
        n = int(self.length * self.sample_rate // self.chunk_size) * self.chunk_size
        ts = np.arange(n)
        f0 = self.band[0] / self.sample_rate
        f1 = self.band[1] / self.sample_rate
        self.gen_running.set()
        for start in range(0, n, self.chunk_size):
            if not self.run_flag.is_set():
                break
            end = start + self.chunk_size
            chunk = chirp(ts[start:end], f0, n, f1, method="logarithmic") * 0.5
            chunk = np.column_stack((chunk, chunk))
            self.output_queue.put(chunk)
        # Write silence
        for i in range(int(self.end_padding * self.sample_rate // self.chunk_size)):
            if not self.run_flag.is_set():
                break
            chunk = np.zeros((self.chunk_size, 2), np.float64)
            self.output_queue.put(chunk)
        self.gen_running.clear()

    def init_stream(self):
        self.stream = sd.Stream(
            blocksize=self.chunk_size, device=self.device, channels=2
        )
        self.sample_rate = self.stream.samplerate

    def audio_io(self):
        if self.stream is None:
            raise RuntimeError(
                "Stream must be initialized before calling starting audio_io"
            )
        else:
            self.stream.start()
            self.audio_running.set()
            while self.gen_running.is_set() or not self.output_queue.empty():
                if not self.run_flag.is_set():
                    break
                try:
                    output_chunk = self.output_queue.get(False)
                    if self.audio_mode == "normal":
                        self.stream.write(output_chunk.astype(np.float32))
                        input_chunk = self.stream.read(len(output_chunk))[0]
                    else:
                        input_chunk = output_chunk
                        sleep(self.chunk_size / self.sample_rate)
                    if self.ref == "generator":
                        input_chunk[:, 1] = output_chunk[:, 0]
                    try:
                        self.input_queue.put_nowait(input_chunk)
                    except Full:
                        print("Input queue full. Dropping a chunk.")
                except Empty:
                    print("Output queue empty. Sleeping for 0.1s.")
                    sleep(0.1)
            self.stream.stop()
            self.audio_running.clear()

    def recorder(self):
        self.recorder_running.set()
        while self.audio_running.is_set() or not self.input_queue.empty():
            if not self.run_flag.is_set():
                break
            try:
                chunk = self.input_queue.get(False)
                with self.record_lock:
                    self.record = np.append(self.record, chunk, 0)
                with self.levels_lock:
                    self.levels = np.append(
                        self.levels, np.max(chunk, axis=0).reshape(1, 2), 0
                    )
            except:
                sleep(0.1)
        self.recorder_running.clear()

    def get_levels(self) -> NDArray[np.float64]:
        with self.levels_lock:
            return self.levels

    def log_filter(
        self, yf: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = int(
            self.freq_length / np.log(20e3 / 20) * np.log(self.band[1] / self.band[0])
        )
        # print(points)
        log_f = np.logspace(
            np.log10(self.band[0]),
            np.log10(self.band[1]),
            num=n,
            dtype=np.float64,
        )
        fft_step = self.sample_rate / len(yf) / 2
        half_w = 2 ** (self.window_width / 2)
        window_start = np.astype(log_f / half_w / fft_step, int)
        window_end = np.astype(log_f * half_w / fft_step, int)
        log_y = np.zeros_like(log_f)
        for i, start, end in zip(range(len(log_f)), window_start, window_end):
            if end <= start:
                log_y[i] = yf[start]
            else:
                log_y[i] = np.median(yf[start:end])
        return log_f, log_y

    def analyzer(self):
        while self.recorder_running.is_set():
            # while True:
            if not self.run_flag.is_set():
                break
            chunk = None
            with self.record_lock:
                if self.analyzer_mode == "recording":
                    if len(self.record) > 0:
                        chunk = self.record.copy()
                elif self.analyzer_mode == "rta":
                    if len(self.record) > self.rta_bucket_size:
                        chunk = self.record[: self.rta_bucket_size]
                        self.record = self.record[self.rta_bucket_size :]
                else:
                    raise ValueError(f"Unknown mode: {self.analyzer_mode}")
            if chunk is None:
                sleep(0.1)
            else:
                fs = self.sample_rate
                nperseg = min(fs / 2, len(chunk))
                x, p = welch(chunk, fs, "hann", nperseg, axis=0)
                if self.ref == "none":
                    fft = p[:, 0]
                else:
                    fft = p[:, 0] / p[:, 1]
                if self.weighting == "pink":
                    fft *= x
                log_f, log_p = self.log_filter(fft)
                result = np.vstack((log_f, 10 * np.log10(log_p.clip(1e-20))))
                with self.fft_result_lock:
                    self.fft_result = result.copy()

    def get_fft(self) -> NDArray[np.float64]:
        with self.fft_result_lock:
            return self.fft_result

    def run(self):
        while not self.stop_flag.is_set():
            self.run_flag.wait()
            if not self.stop_flag.is_set():
                self.init_stream()
                if self.gen_mode == "log sweep":
                    gen = Thread(None, self.log_sweep_gen)
                elif self.gen_mode == "pink noise":
                    gen = Thread(None, self.pink_noise_gen)
                else:
                    raise ValueError(f"Unknown generator mode: {self.gen_mode}")
                gen.start()
                self.gen_running.wait()
                io = Thread(None, self.audio_io)
                io.start()
                self.audio_running.wait()
                recorder = Thread(None, self.recorder)
                recorder.start()
                self.recorder_running.wait()
                analyzer = Thread(None, self.analyzer)
                analyzer.start()
                print("Finished initialization. Waiting workers to stop")
                gen.join()
                print("Generator stopped")
                io.join()
                print("Audio IO stopped")
                recorder.join()
                print("Recorder stopped")
                analyzer.join()
                print("Analyzer stopped")
                print("Clearing run flag")
            self.run_flag.clear()

    def stop(self):
        self.stop_flag.set()
        self.run_flag.set()


if __name__ == "__main__":

    # Example usage:

    pipe = AnalyserPipeline()
    pipe.start()

    pipe.gen_mode = "pink noise"
    pipe.analyzer_mode = "recording"
    pipe.ref = "none"
    pipe.weighting = "pink"
    pipe.audio_mode = "silent"

    pipe.band = (100, 5000)
    pipe.length = 10

    plt.ion()
    _, ax = plt.subplots()
    ax: Axes = ax
    ax.grid(True, which="both")
    line: Line2D = ax.semilogx([], [])[0]
    ax.set_xlim(20, 20e3)
    ax.set_ylim(-100, 20)
    pipe.run_flag.set()
    while pipe.run_flag.is_set():
        freq_data = pipe.get_fft()
        line.set_data(freq_data)
        plt.draw()
        plt.pause(0.1)
    pipe.stop()
    plt.ioff()
    plt.show()
