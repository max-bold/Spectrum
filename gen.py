import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from typing import Generator
from time import time
from numpy.typing import NDArray


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
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    K = length * band[0] / math.log(band[1] / band[0])
    L = length / math.log(band[1] / band[0])
    waveform = np.sin(2 * np.pi * K * (np.exp(t / L) - 1))

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

    if band:
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")

    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    while True:
        waveform = np.random.uniform(-1, 1, chunksize)
        waveform = signal.lfilter(b, a, waveform)
        if band:
            waveform = signal.sosfilt(sos, waveform)
        yield waveform / 0.3


if __name__ == "__main__":
    RATE = 44100

    # lin = linsweep(500, 5000, 10, RATE)
    # log = logsweep(500, 5000, 10, RATE)
    # pnoise = pink_noise(500, 5000, 10, RATE)

    # from analyse import calc_fft

    # linx, liny = calc_fft(lin, RATE)
    # logx, logy = calc_fft(log, RATE)
    # pnoisex, pnoisey = calc_fft(pnoise, RATE)

    # plt.semilogx(linx, 20 * np.log10(liny))
    # plt.semilogx(logx, 20 * np.log10(logy))
    # plt.semilogx(pnoisex, 20 * np.log10(pnoisey))

    # plt.title("FFT of Linear Sweep")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power (dB)")
    # plt.xlim(20, 20000)
    # plt.show()

    # # # pink_noise_gen example usage
    # gen = pink_noise_gen(1024, RATE)
    # for i in range(100):
    #     chunk = next(gen)

    # wf = pink_noise(100, RATE)
    # print(np.median(np.abs(wf)))
