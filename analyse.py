from scipy import signal, fft
from scipy.interpolate import interp1d
import numpy as np


def clac_welch(
    waveform: np.ndarray,
    rate: int,
    start: float = 20,
    stop: float = 20000,
    outlength: int = 1024,
    window=101,
) -> tuple[np.ndarray, np.ndarray]:
    """
     Performs a frequency analysis of an audio waveform using Welch's method and returns a log-spaced spectrum.
        waveform (np.ndarray): Input audio signal as a 1D numpy array.
        rate (int): Sampling rate of the audio signal in Hz.
        outlength (int, optional): Number of frequency bins in the output spectrum (default: 1024).
        window (int, optional): Kernel size for median filtering the spectrum (default: 101).
        tuple[np.ndarray, np.ndarray]:
            - log_xf: Logarithmically spaced frequency bins (in Hz).
            - filtered_log_yf: Median-filtered spectral magnitudes corresponding to log_xf.
    Notes:
        - The function uses Welch's method to estimate the power spectral density.
        - The resulting spectrum is interpolated onto a logarithmic frequency scale between 20 Hz and 20,000 Hz.
        - Median filtering is applied to smooth the spectrum and reduce noise.
    """
    N = len(waveform)
    xf, yf = signal.welch(waveform, rate, nperseg=N / 4, scaling="spectrum")
    log_xf = np.logspace(np.log10(start), np.log10(stop), num=outlength)
    interp_func = interp1d(xf, yf, kind="linear", bounds_error=False, fill_value=0)
    log_yf = interp_func(log_xf)
    filtered_log_yf = signal.medfilt(log_yf, window)
    return log_xf, filtered_log_yf


def calc_fft(
    waveform: np.ndarray,
    rate: int,
    start: float = 20,
    stop: float = 20000,
    outlength: int = 1024,
    window=11,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform FFT on the waveform and return the frequency bins and magnitudes.

    Parameters:
    - waveform: Input audio signal as a 1D numpy array.
    - rate: Sampling rate of the audio signal in Hz.
    - outlength: Number of frequency bins in the output spectrum (default: 1024).
    - window: Kernel size for median filtering the spectrum (default: 101).

    Returns:
    - A tuple containing:
        - xf: Frequency bins (in Hz).
        - yf: Magnitudes of the FFT (in linear scale).
    """
    N = len(waveform)
    yf = np.square(np.abs(fft.rfft(waveform))) / N / rate
    xf = fft.rfftfreq(N, 1 / rate)

    # Interpolate to get a fixed number of output points
    interp_func = interp1d(xf, yf, kind="linear", bounds_error=False, fill_value=0)
    log_xf = np.logspace(np.log10(start), np.log10(stop), num=outlength)
    log_yf = interp_func(log_xf)

    # Apply median filter
    filtered_log_yf = signal.medfilt(log_yf, window)

    return log_xf, filtered_log_yf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gen import pink_noise, linsweep, logsweep

    RATE = 44100

    waveform = pink_noise(20, 20000, 60, RATE)  # Example waveform
    # waveform = logsweep(500, 5000, 10, RATE)  # Example waveform
    xf, yf = calc_fft(waveform, RATE, outlength=1024*16, window=101)

    plt.figure(figsize=(10, 6))
    plt.semilogx(xf, 10 * np.log10(yf))
    plt.title("FFT of Waveform")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.grid(which="both", axis="both")
    plt.show()
