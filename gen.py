import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal


def linsweep(
    start: float = 20,
    stop: float = 20000,
    length: float = 10,
    rate=44100,
    format: str = "int16",
    level=0.9,
) -> np.ndarray:
    """
    Generate a linear frequency sweep from start to stop over the specified length in seconds.

    Parameters:
    - start: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the sweep in seconds.
    - rate: Sample rate in Hz.
    - format: Data type for the output array, e.g., 'int16'.
    - level: Amplitude level of the sine wave (0 to 1).

    Returns:
    - A numpy array containing the audio data for the sweep.
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    waveform = np.sin(
        t * np.pi * (t * (stop - start) / length + 2 * start), dtype=np.float64
    ) * (2**15 * level)

    return waveform.astype(format)


def logsweep(
    start: float = 20,
    stop: float = 20000,
    length: float = 10,
    rate=44100,
    format: str = "int16",
    level=0.9,
) -> np.ndarray:
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
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    K = length * start / math.log(stop / start)
    L = length / math.log(stop / start)
    waveform = np.sin(2 * np.pi * K * (np.exp(t / L) - 1)) * (2**15 * level)

    # Convert to specified format; note: amplitude scaling is suitable for integer types (e.g., 'int16'), 
    # but for float types (e.g., 'float32'), consider normalizing to [-1.0, 1.0] as needed.
    return waveform.astype(format)


def tone(
    freq: float = 20,
    length: float = 10,
    rate=44100,
    format: str = "int16",
    level=0.9,
) -> np.ndarray:
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
    """
    num_samples = int(length * rate)
    t = np.linspace(0, length, num_samples, endpoint=False, dtype=np.float64)
    if np.issubdtype(np.dtype(format), np.integer):
        scale = 2**15 * level
    else:
        scale = level
    waveform = np.sin(2 * np.pi * freq * t) * scale

    return waveform.astype(format)  # Convert to specified format


def noise(
    length: float = 10, rate=44100, format: str = "int16", level=0.9
) -> np.ndarray:
    num_samples = int(length * rate)
    waveform = np.random.uniform(-1, 1, num_samples)
    waveform = waveform / np.max(waveform)
    waveform *= 2**15 * level  # Scale to the specified level
    return waveform.astype(format)


def pink_noise(
    start=20, stop=20000, length=10, rate=44100, format: str = "int16", level=0.9
) -> np.ndarray:
    """
    Generate pink noise over the specified length in seconds.

    Parameters:
    - start: Starting frequency in Hz.
    - stop: Ending frequency in Hz.
    - length: Length of the noise in seconds.
    - rate: Sample rate in Hz.
    - format: Data type for the output array, e.g., 'int16'.
    - level: Amplitude level of the noise (0 to 1).

    Returns:
    - A numpy array containing the audio data for the pink noise.
    """
    num_samples = int(length * rate)
    waveform = np.random.randn(num_samples)
    # Apply a filter to approximate pink noise
    sos1 = signal.butter(4, [start, stop], btype="band", fs=rate, output="sos")
    sos2 = signal.butter(1, 1, btype="low", fs=rate, output="sos")
    waveform = signal.sosfilt(np.vstack([sos1, sos2]), waveform)
    waveform = waveform / np.max(waveform)
    waveform *= 2**15 * level  # Scale to the specified level
    return waveform.astype(format)  # Convert to specified format


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
