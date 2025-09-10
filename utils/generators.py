import numpy as np

# import matplotlib.pyplot as plt
import math
from scipy import signal
from typing import Generator
from abc import ABC, abstractmethod

# from time import time
from numpy.typing import NDArray
from threading import Thread, Event
from queue import Queue


def linsweep(
    length: float = 10,
    rate: int = 44100,
    band: tuple[float, float] = (20, 20e3),
) -> NDArray[np.float64]:
    """
    Generate a linear frequency sweep from start to stop over the specified length in seconds.

    Parameters:
    - start: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the sweep in seconds.
    - rate: Sample rate in Hz.

    Returns:
    - A numpy array containing the audio data for the sweep.

    Notes:
    - Result amplitude is normalized to the range [-1, 1].
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    waveform = np.sin(t * np.pi * (t * (band[1] - band[0]) / length + 2 * band[0]))
    return waveform


def logsweep(
    length: float = 10,
    rate: int = 44100,
    band: tuple[float, float] = (20, 20e3),
    padding: float = 0.2,
) -> NDArray[np.float64]:
    """
    Generate a logarithmic frequency sweep from start to stop over the specified length in seconds.

    Parameters:
    - start: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the sweep in seconds.
    - rate: Sample rate in Hz.
    - format: Data type for the output array, e.g., 'int16'.
    - level: Amplitude level of the sine wave (0 to 1).

    Returns:
    - A numpy array containing the audio data for the sweep.

    Notes:
    - Result amplitude is normalized to the range [-1, 1].
    """
    pad_samples = int(padding * rate)
    num_samples = int(length * rate) - pad_samples * 2
    t = np.linspace(
        0,
        length - padding * 2,
        num_samples,
        endpoint=False,
        dtype=np.float64,
    )
    K = length * band[0] / math.log(band[1] / band[0])
    L = length / math.log(band[1] / band[0])
    waveform = np.zeros(num_samples + pad_samples * 2, dtype=np.float64)
    waveform[pad_samples:-pad_samples] = np.sin(2 * np.pi * K * (np.exp(t / L) - 1))

    return waveform


def tone(
    length: float = 10,
    rate=44100,
    freq: float = 1e3,
) -> NDArray[np.float64]:
    """
    Generates a sine wave of a specified frequency, duration, and amplitude.
    Parameters:
        freq (float): Frequency of the sine wave in Hertz. Default is 20 Hz.
        length (float): Duration of the generated wave in seconds. Default is 10 seconds.
        rate (int): Sampling rate in samples per second (Hz). Default is 44100.
        format (str): Data type for the output NumPy array (e.g., 'int16', 'float32'). Default is 'int16'.
        level (float): Amplitude scaling factor (0.0 to 1.0). Default is 0.9.
    Returns:
        np.ndarray: NumPy array containing the generated sine wave samples in the specified format.

    Notes:
    - Result amplitude is normalized to the range [-1, 1].
    """

    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    waveform = np.sin(2 * np.pi * freq * t)

    return waveform


def pink_noise(
    length: float,
    rate: int,
    band: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """
    Generate pink noise over the specified length in seconds.

    Parameters:
    - band[0]: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the noise in seconds.
    - rate: Sample rate in Hz.

    Returns:
    - A numpy array containing the audio data for the pink noise.

    Notes:
    - Result amplitude is normalized to the range [-1, 1].
    """
    num_samples = int(length * rate)
    waveform = np.random.uniform(-1, 1, num_samples)

    # Filter the noise for 3 db/octave slope
    # Coeffs for pink noise approximation from https://www.dsprelated.com/freebooks/SASP/Example_Synthesis_1_F_Noise.html
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    waveform = signal.lfilter(b, a, waveform)

    if band:
        # Filter the noise to given band
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")
        waveform = signal.sosfilt(sos, waveform)

    # Normalize the waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform))

    return waveform


def pink_noise_gen(
    chunksize: int,
    rate: int = 44100,
    band: tuple[float, float] | None = None,
) -> Generator[NDArray[np.float64], None, None]:
    sos = None
    if band:
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")

    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    while True:
        waveform = np.random.uniform(-1, 1, chunksize)
        waveform = signal.lfilter(b, a, waveform)
        if sos:
            waveform = signal.sosfilt(sos, waveform)
        yield np.asarray(waveform, np.float64) / 0.3


class Signal_generator(Thread, ABC):
    """
    A base class for signal generators, inheriting from Thread and ABC.
    Attributes:
        output_queue (Queue): A queue to hold generated signal data, with a maximum size of 10.
        stop_signal (Event): An event to signal the generator to stop.
    Methods:
        __init__(): Initializes the signal generator, output queue, and stop signal. When overridden, remember to call super().__init__().
        run(): An abstract method that must be implemented by subclasses to define the signal generation logic.
        stop(): Signals the generator to stop. If the output queue is full, removes one item before setting the stop signal.
    """

    def __init__(self) -> None:
        self.output_queue = Queue(maxsize=10)
        self._stop_signal = Event()
        return super().__init__()

    @abstractmethod
    def run(self) -> None:
        pass

    def stop(self):
        if self.output_queue.full():
            self.output_queue.get()
        self._stop_signal.set()
        return self.join()


class PinkNoiseGenerator(Signal_generator):
    """
    PinkNoiseGenerator generates pink noise signals with optional bandpass filtering.
    Args:
        rate (int): Sample rate in Hz.
        chunksize (int): Number of samples per generated chunk.
        length (float | None, optional): Total duration of noise in seconds. If None, runs indefinitely.
        band (tuple[float, float] | None, optional): Frequency band (low, high) for bandpass filtering. If None, no filtering is applied.
        boost (float, optional): Amplitude scaling factor for the generated noise. Default is 3.0.
    Methods:
        start(): Starts generating pink noise and puts chunks into the output queue until stopped or length is reached.
        stop(): Signals the generator to stop.
    """

    def __init__(
        self,
        rate: int,
        chunksize: int,
        length: float | None = None,
        band: tuple[float, float] | None = None,
        boost: float = 3.0,
    ) -> None:
        self.rate = rate
        self.chunksize = chunksize
        self.length = int(length * rate) if length else None
        self.band = band
        self.boost = boost
        return super().__init__()

    def run(self) -> None:
        sent_samples = 0
        sos = None
        if self.band:
            sos = signal.butter(
                4,
                self.band,
                btype="bandpass",
                fs=self.rate,
                output="sos",
            )
        B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        A = [1, -2.494956002, 2.017265875, -0.522189400]
        while not self._stop_signal.is_set() or (
            self.length and sent_samples < self.length
        ):
            chunk_l = (
                min(self.chunksize, self.length - sent_samples)
                if self.length
                else self.chunksize
            )
            noise = np.random.uniform(-1, 1, chunk_l)
            noise = signal.lfilter(B, A, noise)
            if sos is not None:
                noise = signal.sosfilt(sos, noise)
            noise = np.asarray(noise, dtype=np.float64) * self.boost
            self.output_queue.put(noise)
            sent_samples += chunk_l
        return


class LogSweepGenerator(Signal_generator):
    def __init__(
        self,
        rate: int,
        chunksize: int,
        band: tuple[float, float] = (20, 20e3),
        length: float = 10,
    ) -> None:
        self.rate = rate
        self.chunksize = chunksize
        self.band = np.array(band) / rate
        self.length = int(length * rate)
        return super().__init__()

    def run(self) -> None:
        k = self.length * self.band[0] / np.log(self.band[1] / self.band[0])
        l = self.length / np.log(self.band[1] / self.band[0])
        t = np.arange(0, self.length, dtype=np.float64)
        for start in range(0, self.length, self.chunksize):
            if self._stop_signal.is_set():
                break
            stop = min(start + self.chunksize, self.length)
            chunk_t = t[start:stop]
            waveform = np.sin(2 * np.pi * k * (np.exp(chunk_t / l) - 1))
            self.output_queue.put(waveform)


if __name__ == "__main__":
    from scipy.fft import rfft, rfftfreq
    import matplotlib.pyplot as plt

    RATE = 96000
    CHUNKSIZE = 1024 * 40
    BAND = (100, 1e3)
    LENGTH = 30  # seconds

    # Generate pink noise

    # chunks_to_read = int(RATE * LENGTH / CHUNKSIZE)
    # gen = PinkNoiseGenerator(RATE, CHUNKSIZE)
    # gen.start()
    # data = gen.output_queue.get()
    # for i in range(chunks_to_read):
    #     data = np.concatenate((data, gen.output_queue.get()))
    # gen.stop()
    # gen.join()

    # Generate log sweep

    # gen = LogSweepGenerator(RATE, CHUNKSIZE, BAND, LENGTH)
    # gen.start()

    # data = gen.output_queue.get()
    # while gen.is_alive() or not gen.output_queue.empty():
    #     data = np.concatenate((data, gen.output_queue.get()))

    # Calculate FFT and plot result

    # print(np.max(data))
    # xf = rfftfreq(len(data), 1 / RATE)
    # yf = np.abs(rfft(data)) / len(data)
    # yf *= np.sqrt(xf)
    # filtered_yf = signal.medfilt(yf, 11)
    # filtered_yf = np.clip(filtered_yf, 1e-12, None)
    # plt.semilogx(xf, 20 * np.log10(filtered_yf))
    # plt.xlim(20, 20e3)
    # # xt = np.arange(len(data)) / RATE
    # # plt.plot(xt, data)

    # plt.show()
