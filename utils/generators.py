import numpy as np

import math
from scipy import signal
from typing import Generator
from abc import ABC, abstractmethod

from numpy.typing import NDArray
from threading import Thread, Event
from queue import Queue


def linsweep(
    length: float = 10,
    rate: int = 44100,
    band: tuple[float, float] = (20, 20e3),
) -> NDArray[np.float64]:
    """
    Generate a linear frequency sweep signal.

    Args:
        length (float): Duration of sweep in seconds.
        rate (int): Sample rate in Hz.
        band (tuple): (start_freq, stop_freq) in Hz.

    Returns:
        np.ndarray: Linear sweep signal, normalized to [-1, 1].
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    sweep = np.sin(t * np.pi * (t * (band[1] - band[0]) / length + 2 * band[0]))
    sweep /= np.max(np.abs(sweep))
    return sweep


def logsweep(
    length: float = 10,
    rate: int = 44100,
    band: tuple[float, float] = (20, 20e3),
    padding: float = 0.2,
) -> NDArray[np.float64]:
    """
    Generate a logarithmic frequency sweep signal.

    Args:
        length (float): Duration of sweep in seconds.
        rate (int): Sample rate in Hz.
        band (tuple): (start_freq, stop_freq) in Hz.
        padding (float): Silence padding at start/end in seconds.

    Returns:
        np.ndarray: Logarithmic sweep signal, normalized to [-1, 1].
    """
    num_samples = int(length * rate)
    t = np.linspace(
        0,
        length - padding * 2,
        num_samples,
        endpoint=False,
        dtype=np.float64,
    )
    K = length * band[0] / math.log(band[1] / band[0])
    L = length / math.log(band[1] / band[0])
    sweep = np.sin(2 * np.pi * K * (np.exp(t / L) - 1))
    return sweep


def tone(
    length: float = 10,
    rate: int = 44100,
    freq: float = 1000,
) -> NDArray[np.float64]:
    """
    Generate a sine wave signal.

    Args:
        length (float): Duration in seconds.
        rate (int): Sample rate in Hz.
        freq (float): Frequency in Hz.

    Returns:
        np.ndarray: Sine wave, normalized to [-1, 1].
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    wave = np.sin(2 * np.pi * freq * t)
    return wave


def pink_noise(
    length: float,
    rate: int,
    band: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """
    Generate pink noise signal.

    Args:
        length (float): Duration in seconds.
        rate (int): Sample rate in Hz.
        band (tuple, optional): (low, high) bandpass filter in Hz.

    Returns:
        np.ndarray: Pink noise, normalized to [-1, 1].
    """
    num_samples = int(length * rate)
    noise = np.random.uniform(-1, 1, num_samples)
    # Pink noise filter coefficients (see dsprelated.com)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    noise = signal.lfilter(b, a, noise)
    if band:
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")
        noise = signal.sosfilt(sos, noise)
    noise /= np.max(np.abs(noise))
    return noise


class SignalGenerator(Thread, ABC):
    """
    Abstract base class for threaded signal generators.

    Attributes:
        output_queue (Queue): Queue for generated signal chunks.
        _stop_signal (Event): Event to signal stopping.
    """

    def __init__(self) -> None:
        self.output_queue = Queue(maxsize=10)
        self._stop_signal = Event()
        super().__init__()

    @abstractmethod
    def run(self) -> None:
        """Implement signal generation logic in subclasses."""
        pass

    def stop(self):
        """Signal the generator to stop and wait for thread to finish."""
        if self.output_queue.full():
            """Most of generators will block on full queue,
            so we need to free one slot for it to reshoot
            and proceed the _stop_signal."""
            self.output_queue.get()
        self._stop_signal.set()
        return self.join()
    
    def get(self) -> NDArray[np.float64]:
        """Get one chunk of generated signal from the output queue."""
        chunk = self.output_queue.get()
        self.output_queue.task_done()
        return chunk


class PinkNoiseGenerator(SignalGenerator):
    """
    Threaded pink noise generator with optional bandpass filtering.

    Args:
        rate (int): Sample rate in Hz.
        chunksize (int): Samples per chunk.
        length (float|None): Duration in seconds (None for infinite).
        band (tuple|None): (low, high) bandpass filter in Hz.
        boost (float): Amplitude scaling factor.
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
        super().__init__()

    def run(self) -> None:
        sent_samples = 0
        sos = None
        if self.band:
            sos = signal.butter(
                4, self.band, btype="bandpass", fs=self.rate, output="sos"
            )
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        while not self._stop_signal.is_set() and (
            self.length is None or sent_samples < self.length
        ):
            chunk_l = (
                min(self.chunksize, self.length - sent_samples)
                if self.length
                else self.chunksize
            )
            noise = np.random.uniform(-1, 1, chunk_l)
            noise = signal.lfilter(b, a, noise)
            if sos is not None:
                noise = signal.sosfilt(sos, noise)
            noise = np.asarray(noise, dtype=np.float64)
            noise *= self.boost
            self.output_queue.put(noise)
            sent_samples += chunk_l
        print(f"Generator stoped after {sent_samples} samples.")


class LogSweepGenerator(SignalGenerator):
    """
    Threaded logarithmic sweep generator.

    Args:
        rate (int): Sample rate in Hz.
        chunksize (int): Samples per chunk.
        band (tuple): (start_freq, stop_freq) in Hz.
        length (float): Duration in seconds.
    """

    def __init__(
        self,
        rate: int,
        chunksize: int,
        band: tuple[float, float] = (20, 20000),
        length: float = 10,
    ) -> None:
        self.rate = rate
        self.chunksize = chunksize
        self.band = np.array(band) / rate
        self.length = int(length * rate)
        super().__init__()

    def run(self) -> None:
        k = self.length * self.band[0] / np.log(self.band[1] / self.band[0])
        l = self.length / np.log(self.band[1] / self.band[0])
        t = np.arange(0, self.length, dtype=np.float64)
        for start in range(0, self.length, self.chunksize):
            if self._stop_signal.is_set():
                break
            stop = min(start + self.chunksize, self.length)
            chunk_t = t[start:stop]
            sweep = np.sin(2 * np.pi * k * (np.exp(chunk_t / l) - 1))
            self.output_queue.put(sweep)
        print(f"Sweep generator stoped after {self.length} samples.")


if __name__ == "__main__":
    from scipy.fft import rfft, rfftfreq
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    RATE = 96000
    CHUNKSIZE = 1024 * 4
    BAND = (100, 1e3)
    LENGTH = 30  # seconds

    # Generate pink noise

    # chunks_to_read = int(RATE * LENGTH / CHUNKSIZE)
    # gen = PinkNoiseGenerator(RATE, CHUNKSIZE, band=BAND)
    # gen.start()
    # data = gen.output_queue.get()
    # for i in range(chunks_to_read):
    #     data = np.concatenate((data, gen.output_queue.get()))
    # gen.stop()
    # gen.join()

    # Generate log sweep

    gen = LogSweepGenerator(RATE, CHUNKSIZE, BAND, LENGTH)
    gen.start()

    data = gen.output_queue.get()
    while gen.is_alive() or not gen.output_queue.empty():
        data = np.concatenate((data, gen.output_queue.get()))

    # Calculate FFT and plot result

    print(np.max(data))
    xf = rfftfreq(len(data), 1 / RATE)
    yf = np.array(rfft(data))
    yf = np.abs(yf) / len(data)
    yf *= np.sqrt(xf)
    filtered_yf = signal.medfilt(yf, 101)
    filtered_yf = np.clip(filtered_yf, 1e-12, None)
    xt = np.arange(len(data)) / RATE

    fig, axes = plt.subplots(2, height_ratios=(10, 5))
    axes: list[Axes] = axes
    axes[0].semilogx(xf, 20 * np.log10(filtered_yf))
    axes[0].set_xlim(20, 20e3)
    axes[0].grid(True, "both")
    axes[0].autoscale(True, "y")
    axes[1].plot(xt, data)
    axes[1].grid(True, "both")

    plt.show()
