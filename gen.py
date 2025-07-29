import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from typing import Generator
from time import time


def linsweep(
    start: float = 20, stop: float = 20000, length: float = 10, rate=44100
) -> np.ndarray[np.float64]:
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
    waveform = np.sin(
        t * np.pi * (t * (stop - start) / length + 2 * start),
        dtype=np.float64,
    )
    return waveform


def logsweep(
    start: float = 20,
    stop: float = 20000,
    length: float = 10,
    rate=44100,
) -> np.ndarray[np.float64]:
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
    K = length * start / math.log(stop / start)
    L = length / math.log(stop / start)
    waveform = np.sin(2 * np.pi * K * (np.exp(t / L) - 1))

    return waveform


def tone(
    freq: float = 20,
    length: float = 10,
    rate=44100,
) -> np.ndarray[np.float64]:
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


def noise(length: float = 10, rate=44100) -> np.ndarray[np.float64]:
    """
    Generates a normalized white noise waveform.
    Args:
        length (float, optional): Duration of the noise in seconds. Defaults to 10.
        rate (int, optional): Sample rate in Hz. Defaults to 44100.
    Returns:
        np.ndarray: Array containing the normalized white noise samples.
    """

    num_samples = int(length * rate)
    waveform = np.random.randn(num_samples, dtype=np.float64)
    return waveform


def pink_noise(
    length: float,
    rate: int,
    band: tuple[float, float] | None = None,
) -> np.ndarray[np.float64]:
    """
    Generate pink noise over the specified length in seconds.

    Parameters:
    - start: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the noise in seconds.
    - rate: Sample rate in Hz.

    Returns:
    - A numpy array containing the audio data for the pink noise.

    Notes:
    - Result amplitude is normalized to the range [-1, 1].
    """
    num_samples = int(length * rate)
    waveform = np.random.randn(num_samples)
    # Filter the noise to given band
    if band:
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")
        waveform = signal.sosfilt(sos, waveform)
    # sos2 = signal.butter(1, 1, btype="low", fs=rate, output="sos")

    # Filter the noise for 3 db/octave slope
    # Coeffs for pink noise approximation from https://www.dsprelated.com/freebooks/SASP/Example_Synthesis_1_F_Noise.html
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]

    waveform = signal.lfilter(b, a, waveform)

    # Normalize the waveform to the range [-1, 1]
    waveform = waveform / np.max(np.abs(waveform))

    return waveform


def pink_noise_gen(
    length: float,
    chunksize: int,
    rate: int,
    band: tuple[float, float] | None = None,
) -> Generator[np.ndarray[np.float64], None, None]:
    if band:
        sos = signal.butter(4, band, btype="band", fs=rate, output="sos")
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    st = time()
    med = 1
    while time() - st < length:
        waveform = np.random.randn(chunksize)
        if band:
            waveform = signal.sosfilt(sos, waveform)
        waveform = signal.lfilter(b, a, waveform)
        med = (
            20 * med + np.max(np.abs(waveform))
        ) / 21  # Using sigma filtering for volume control
        waveform = waveform / med / 2
        # print(np.max(waveform), med)
        yield waveform


if __name__ == "__main__":
    RATE = 44100

    lin = linsweep(500, 5000, 10, RATE)
    log = logsweep(500, 5000, 10, RATE)
    pnoise = pink_noise(500, 5000, 10, RATE)

    from analyse import calc_fft

    linx, liny = calc_fft(lin, RATE)
    logx, logy = calc_fft(log, RATE)
    pnoisex, pnoisey = calc_fft(pnoise, RATE)

    plt.semilogx(linx, 20 * np.log10(liny))
    plt.semilogx(logx, 20 * np.log10(logy))
    plt.semilogx(pnoisex, 20 * np.log10(pnoisey))

    plt.title("FFT of Linear Sweep")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.xlim(20, 20000)
    plt.show()
