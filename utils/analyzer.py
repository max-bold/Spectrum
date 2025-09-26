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
from scipy.signal.windows import blackman
import sounddevice as sd


class AnalyserPipeline(Thread):
    """
    AnalyserPipeline
    A threaded audio analysis pipeline for real-time signal generation, playback, recording, and spectral analysis.
    This class orchestrates the generation of test signals (such as pink noise or logarithmic sweeps), real-time audio I/O, recording, and spectral analysis in a multi-threaded environment. It is designed for use in audio measurement and analysis applications, supporting both real-time analysis (RTA) and post-recording analysis modes.
    Attributes:
        Generator params:
            length (float): Duration of the generated signal in seconds.
            band (tuple[float, float]): Frequency band for signal generation and analysis (Hz).
            gen_mode (Literal["pink noise", "log sweep"]): Type of signal to generate.
            output_queue (Queue[NDArray[np.float64]]): Queue for generated audio chunks.
            gen_running (Event): Event flag indicating if the generator is running.
            end_padding (float): Duration of silence appended after signal generation (seconds).
        AudioIO  params:
            sample_rate (int): Audio sample rate (Hz).
            chunk_size (int): Size of audio chunks for processing.
            device (tuple[int, int] | None): Audio device indices (input, output).
            input_queue (Queue[NDArray[np.float64]]): Queue for incoming audio chunks.
            audio_running (Event): Event flag indicating if audio I/O is running.
            audio_mode (Literal["normal", "silent"]): Audio I/O mode.
            stream (sd.Stream | None): Sounddevice stream object.
        Recorder params:
            record (NDArray[np.float64]): Recorded audio data.
            record_lock (Lock): Lock for thread-safe access to recorded data.
            recorder_running (Event): Event flag indicating if the recorder is running.
            levels (NDArray[np.float64]): Peak levels for each audio chunk.
            levels_lock (Lock): Lock for thread-safe access to levels.
        Analyzer params:
            analyzer_mode (Literal["rta", "recording"]): Analysis mode.
            ref (Literal["none", "channel B", "generator"]): Reference channel for analysis.
            weighting (None | Literal["pink"]): Optional spectral weighting.
            rta_bucket_size (int): Chunk size for real-time analysis.
            freq_length (int): Number of frequency bins for analysis.
            window_width (float): Width of the frequency window for log filtering.
            fft_result (NDArray[np.float64]): Latest FFT analysis result.
            fft_result_lock (Lock): Lock for thread-safe access to FFT results.
        Global flags:
            stop_flag (Event): Event flag to stop the pipeline.
            run_flag (Event): Event flag to start or pause the pipeline.
    Methods:
        pink_noise_gen(): Generates pink noise, applies bandpass filtering, and outputs stereo audio chunks.
        log_sweep_gen(): Generates and outputs logarithmic frequency sweep (chirp) audio chunks.
        padding_gen(): Appends silence (zero chunks) to the output queue after signal generation.
        init_stream(): Initializes the audio stream for I/O.
        audio_io(): Handles real-time audio playback and recording, feeding data to the input queue.
        recorder(): Records incoming audio chunks and tracks peak levels.
        get_levels() -> NDArray[np.float64]: Returns the recorded peak levels.
        log_filter(yf: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]: Applies logarithmic frequency binning to FFT data.
        analyzer(): Performs spectral analysis on recorded data and updates FFT results.
        get_fft() -> NDArray[np.float64]: Returns the latest FFT analysis result.
        run(): Main pipeline loop, coordinating all worker threads.
        stop(): Signals the pipeline to stop and releases all waiting threads.
    Usage:
        Instantiate the pipeline, configure parameters as needed, and start the thread.
        Use the `run_flag` to start processing and `stop()` to terminate the pipeline.
    Thread Safety:
        Uses threading Events and Locks to coordinate and protect shared resources across multiple worker threads.
    """

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
        self.device = self.get_default_io()
        self.input_queue: Queue[NDArray[np.float64]] = Queue(10)
        self.audio_running = Event()
        self.audio_mode: Literal["normal", "silent"] = "normal"
        self.stream: sd.Stream | None = None

        # Recorder params
        self.record = np.empty((0, 2), dtype=np.float64)
        self.record_lock = Lock()
        self.recorder_running = Event()
        self.levels = np.empty((2, 0), dtype=np.float64)
        self.times = np.empty(0, np.float64)
        self.levels_lock = Lock()
        self.record_upd = Event()

        # Analyzer params
        self.analyzer_mode: Literal["rta", "recording"] = "recording"
        self.ref: Literal["none", "channel B", "generator"] = "channel B"
        self.weighting: Literal["none", "pink"] = "none"
        self.rta_bucket_size: int = 2**15
        self.freq_length = 1024
        self.window_width = 1 / 10
        self.fft_result = np.empty((2, 0), np.float64)
        self.fft_result_lock = Lock()
        self.final_fft_ready = Event()

        # Pipeline params
        self.stop_flag = Event()
        self.run_flag = Event()

        return super().__init__(daemon=True)

    def pink_noise_gen(self) -> None:
        """
        Generates pink noise, applies bandpass filtering, and outputs stereo audio chunks.
        This generator method creates pink noise, applies a pinking filter and a bandpass filter,
        and outputs the result in stereo chunks to the output queue. The process continues for the
        duration specified by `self.length` or until `self.run_flag` is cleared. After generation,
        it calls `self.padding_gen()` to handle any necessary padding and clears the
        `self.gen_running` event flag.
        """

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
        s_pad = np.rint(0.5 * self.sample_rate / self.chunk_size).astype(int)
        n = np.rint(self.length * self.sample_rate / self.chunk_size).astype(int)
        e_pad = np.rint(self.end_padding * self.sample_rate / self.chunk_size).astype(
            int
        )
        att = 2**-10
        for i in range(s_pad + n + e_pad):
            if not self.run_flag.is_set():
                break
            if i <= s_pad:
                white = np.zeros(self.chunk_size)
            elif i <= s_pad + n:
                white = np.random.uniform(-1, 1, self.chunk_size)
            else:
                white = np.zeros(self.chunk_size)
            pink, zi = sosfilt(combined_sos, white, -1, zi)
            pink = np.array(pink, np.float64)
            att = min(1, att * 2)
            chunk = np.column_stack((pink, pink)) * att * 3
            self.output_queue.put(chunk)
        self.gen_running.clear()

    def log_sweep_gen(self) -> None:
        """
        Generates and outputs logarithmic frequency sweep (chirp) audio chunks.
        This generator method creates a series of audio chunks representing a logarithmic frequency sweep
        between the frequencies specified in `self.band`, sampled at `self.sample_rate`, and divided into
        chunks of size `self.chunk_size`. Each chunk is generated using a logarithmic chirp signal and is
        output as a stereo signal (duplicated across two channels). The generated chunks are placed into
        `self.output_queue` for further processing or playback.
        The method runs until all chunks are generated or until `self.run_flag` is cleared. After
        generation, it calls `self.padding_gen()` to handle any necessary padding and clears the
        `self.gen_running` event flag.
        Raises:
            None directly, but may propagate exceptions from queue operations or signal generation.
        """

        n = np.rint(self.length * self.sample_rate / self.chunk_size).astype(int)
        ts = np.arange(0, n * self.chunk_size)
        f0 = self.band[0] / self.sample_rate
        f1 = self.band[1] / self.sample_rate
        t1 = ts[-1]
        self.gen_running.set()
        for i in range(n):
            if not self.run_flag.is_set():
                break
            start = i * self.chunk_size
            end = (i + 1) * self.chunk_size
            chunk = chirp(ts[start:end], f0, t1, f1, method="logarithmic") * 0.5
            chunk = np.column_stack((chunk, chunk))
            self.output_queue.put(chunk)
        pass
        # self.padding_gen(self.end_padding)
        self.gen_running.clear()

    def padding_gen(self, length: float) -> None:
        """
        Generates and enqueues zero-padding audio chunks to the output queue.
        This generator method creates a specified number of zero-filled audio chunks,
        each of shape (chunk_size, 2), corresponding to stereo audio. The number of
        chunks is determined by the end_padding duration, sample_rate, and chunk_size.
        The method checks the run_flag event; if it is cleared, the generation stops early.
        """
        for i in range(int((length * self.sample_rate) // self.chunk_size)):
            if not self.run_flag.is_set():
                break
            zeros = np.zeros((self.chunk_size, 2))
            self.output_queue.put(zeros)

    def host_api_is_wasapi(self):
        


    def init_stream(self):
        """
        Initializes the audio stream for input/output using the specified device and chunk size.
        This method creates a new sounddevice Stream object with the configured chunk size,
        device, and sets the number of channels to 2 (stereo). It also updates the sample_rate
        attribute with the stream's sample rate.
        Raises:
            sounddevice.PortAudioError: If the stream cannot be initialized with the given parameters.
        """
        if self.audio_mode == "normal":
            self.stream = sd.Stream(
                blocksize=self.chunk_size, device=self.device, channels=2
            )
            self.sample_rate = self.stream.samplerate

    @staticmethod
    def get_default_io() -> tuple[int, int]:
        dev = []
        for k in ["input", "output"]:
            info = sd.query_devices(kind=k)
            if isinstance(info, dict) and info["index"]:
                dev.append(info["index"])
        return tuple(dev)

    def audio_io(self):
        """
        Handles real-time audio input/output processing using a stream.
        This method manages the audio stream lifecycle, reading from an output queue and writing audio data to the stream.
        It also reads input from the stream (if in "normal" audio mode), processes it, and puts it into an input queue for further handling.
        The method supports two audio modes: "normal" (full duplex) and an alternative mode where output is looped back as input.
        It synchronizes with threading events to control execution and handles queue underflow/overflow gracefully.
        Raises:
            RuntimeError: If the audio stream is not initialized before calling this method.
        """

        if self.stream is None and self.audio_mode == "normal":
            raise RuntimeError(
                "Stream must be initialized before calling starting audio_io"
            )
        else:
            if self.stream is not None:
                self.stream.start()
            self.audio_running.set()
            while self.gen_running.is_set() or not self.output_queue.empty():
                if not self.run_flag.is_set():
                    try:
                        self.output_queue.get(False)
                    except Empty:
                        pass
                    break
                try:
                    output_chunk = self.output_queue.get(False)
                    if self.stream is not None:
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
            if self.stream is not None:
                self.stream.stop()
            self.audio_running.clear()

    def recorder(self):
        """
        Continuously records audio data from the input queue and appends it to the internal buffers.
        This method runs in a loop while audio recording is active or there is data left in the input queue.
        It checks a run flag to allow for early termination. Audio chunks are retrieved from the input queue,
        and appended to the `self.record` array under a thread lock to ensure thread safety. The maximum levels
        of each chunk are also computed and appended to the `self.levels` array under a separate lock.
        If the input queue is empty, the method waits briefly before retrying.
        The method sets and clears the `self.recorder_running` event to indicate its running state.
        """

        self.recorder_running.set()
        tot_samples = 0
        while self.audio_running.is_set() or not self.input_queue.empty():
            if not self.run_flag.is_set():
                break
            try:
                chunk = self.input_queue.get(False)
                tot_samples += len(chunk)
                with self.record_lock:
                    self.record = np.append(self.record, chunk, 0)
                    self.record_upd.set()
                with self.levels_lock:
                    self.levels = np.append(
                        self.levels,
                        np.reshape(np.max(np.abs(chunk), axis=0), (2, 1)),
                        1,
                    )
                    self.times = np.append(self.times, tot_samples / self.sample_rate)
            except Empty:
                sleep(0.1)
        self.recorder_running.clear()

    def get_levels(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Retrieve the current levels array in a thread-safe manner.
        Returns:
            NDArray[np.float64]: The array containing the current levels.
        """

        with self.levels_lock:
            return self.times, self.levels

    def log_filter(
        self, yf: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Applies a logarithmic filter to the input frequency spectrum.
        This method computes a logarithmically spaced frequency axis within the specified band,
        and for each frequency bin, aggregates the corresponding values from the input spectrum
        using a mean filter within a window centered at each log-spaced frequency.
        Parameters:
            yf (NDArray[np.float64]): The input frequency spectrum (e.g., magnitude or power values).
        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64]]:
                - log_f: Logarithmically spaced frequency bins within the specified band.
                - log_y: Median-filtered spectrum values corresponding to each log-spaced frequency bin.
        """
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
        window_start = np.rint(log_f / half_w / fft_step).astype(int)
        window_end = np.rint(log_f * half_w / fft_step).astype(int) + 1
        log_y = np.zeros_like(log_f)
        for i, start, end in zip(range(len(log_f)), window_start, window_end):
            # if end <= start:
            #     log_y[i] = yf[start]
            # else:
            ys = yf[start:end]
            # w = blackman(end - start)
            log_y[i] = np.mean(ys)
            # log_y[i] = np.average(ys, None, w)
        return log_f, log_y

    def analyzer(self):
        """
        Continuously processes audio data from the recorder in either 'recording' or 'rta' (real-time analysis) mode.
        The method runs in a loop while the recorder is active and the run flag is set. Depending on the analyzer mode:
            - In 'recording' mode, it processes the entire recorded data.
            - In 'rta' mode, it processes data in buckets of size `rta_bucket_size`.
        For each chunk of data:
            - Computes the power spectral density using Welch's method.
            - Applies reference normalization if specified.
            - Applies pink noise weighting if selected.
            - Applies logarithmic filtering to the spectrum.
            - Stores the processed FFT result in a thread-safe manner.
        Raises:
            ValueError: If an unknown analyzer mode is specified.
        """

        while self.recorder_running.is_set() or self.record_upd.is_set():
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
                self.record_upd.clear()
            if chunk is None:
                sleep(0.1)
            else:
                fs = self.sample_rate
                nperseg = min(fs / 2, len(chunk))
                x, p = welch(chunk, fs, "hann", nperseg, axis=0)
                # x, p = periodogram(chunk, fs, axis=0)
                # print(x[1])
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
        if self.analyzer_mode == "recording":
            with self.record_lock:
                rec = self.record.copy()
            x, p = periodogram(rec, self.sample_rate, axis=0)
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
            self.final_fft_ready.set()

    def get_fft(self) -> NDArray[np.float64]:
        """
        Returns the current FFT (Fast Fourier Transform) result.
        Returns:
            NDArray[np.float64]: The FFT result as a NumPy array of float64 values.
        """

        with self.fft_result_lock:
            self.final_fft_ready.clear()
            return self.fft_result.copy()
        

    def run(self):
        """
        Runs the main processing loop for the analyzer.
        This method manages the lifecycle of several worker threads responsible for signal generation,
        audio input/output, recording, and analysis. It waits for the `run_flag` event to be set before
        initializing and starting the threads according to the selected generator mode (`log sweep` or
        `pink noise`). Each worker thread is started in sequence, and the method waits for each to signal
        readiness before proceeding. After all threads are running, it waits for them to complete before
        clearing the `run_flag` and potentially repeating the process unless a stop is requested.
        Raises:
            ValueError: If an unknown generator mode is specified.
        """

        while not self.stop_flag.is_set():
            self.run_flag.wait()
            if not self.stop_flag.is_set():
                self.init_stream()
                with self.levels_lock:
                    self.levels = np.empty((2, 0), dtype=np.float64)
                    self.times = np.empty(0, np.float64)
                with self.record_lock:
                    self.record = np.empty((0, 2), dtype=np.float64)
                if self.gen_mode == "log sweep":
                    gen = Thread(None, self.log_sweep_gen, daemon=True)
                elif self.gen_mode == "pink noise":
                    gen = Thread(None, self.pink_noise_gen, daemon=True)
                else:
                    raise ValueError(f"Unknown generator mode: {self.gen_mode}")
                gen.start()
                self.gen_running.wait()
                io = Thread(None, self.audio_io, daemon=True)
                io.start()
                self.audio_running.wait()
                recorder = Thread(None, self.recorder, daemon=True)
                recorder.start()
                self.recorder_running.wait()
                analyzer = Thread(None, self.analyzer, daemon=True)
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
        """
        Stops the analyzer by setting the stop flag and ensuring the run flag is set.
        This method signals the analyzer to stop its operation by setting the `stop_flag`.
        It also triggers the `run_flag`, if pipeline is paused.
        """
        self.stop_flag.set()
        if self.run_flag.is_set():
            self.run_flag.clear()
        else:
            self.run_flag.set()


if __name__ == "__main__":

    # Example usage of AnalyserPipeline with real-time plotting

    # Create and start the analyzer pipeline thread
    pipe = AnalyserPipeline()
    pipe.start()

    # Configure pipeline parameters
    pipe.gen_mode = "log sweep"  # Use pink noise as the test signal
    pipe.analyzer_mode = "recording"  # Analyze the full recording after playback
    pipe.ref = "generator"  # No reference channel normalization
    # pipe.weighting = "pink"  # Apply pink noise spectral weighting
    # pipe.audio_mode = "silent"  # Do not use actual audio hardware
    pipe.device = (12, 11)
    pipe.rta_bucket_size = int(pipe.sample_rate / 4)

    pipe.band = (20, 20e3)  # Frequency band for generation/analysis
    pipe.length = 20
    pipe.window_width = 1 / 10  # Duration of the test signal in seconds

    # Set up interactive plotting
    plt.ion()
    _, axs = plt.subplots(2)
    axs: list[Axes]
    axs[0].grid(True, which="both")
    axs[1].grid(True, which="both")
    line: Line2D = axs[0].semilogx([], [])[0]
    axs[0].set_xlim(20, 20e3)
    axs[0].set_ylim(-100, 20)
    lineL = axs[1].plot([], [])[0]
    lineR = axs[1].plot([], [])[0]
    axs[1].set_xlim(-1, pipe.length + 5)
    axs[1].set_ylim(0, 1)

    # Start the pipeline processing
    pipe.run_flag.set()
    while pipe.run_flag.is_set():
        # Fetch the latest FFT result and update the plot
        freq_data = pipe.get_fft()
        line.set_data(freq_data)
        ts, levels = pipe.get_levels()
        lineL.set_data(ts, (levels[0].clip(1e-20)))
        lineR.set_data(ts, (levels[1].clip(1e-20)))
        plt.draw()
        plt.pause(0.1)
    if pipe.analyzer_mode == "recording":
        pipe.final_fft_ready.wait()
        freq_data = pipe.get_fft()
        line.set_data(freq_data)

    # Stop the pipeline and finalize the plot
    pipe.stop()
    plt.ioff()
    plt.show()
