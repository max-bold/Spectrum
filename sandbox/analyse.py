from scipy import signal, fft
from scipy.interpolate import interp1d
import numpy as np


def calc_welch(
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


def calc_psd(
    waveform: np.ndarray,
    rate: int,
    band: tuple[float, float] = (20, 20000),
    outlength: int = 1024,
    window: float = 1 / 3,
    norm=True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of an audio waveform using FFT and median filtering.

    Parameters:
        waveform (np.ndarray): Input audio signal as a 1D numpy array.
        rate (int): Sampling rate of the audio signal in Hz.
        band (tuple[float, float], optional): Frequency range (in Hz) to analyze, as (low, high). Default is (20, 20000).
        outlength (int, optional): Number of frequency bins in the output spectrum. Default is 1024.
        window (float, optional): Width (in octaves) of the median filter window applied to the spectrum. Default is 1/3.
        norm (bool, optional): If True, normalize the output magnitudes to a maximum of 1. Default is True.
    
    Returns:
        tuple[np.ndarray, np.ndarray]:
            - log_xf: Logarithmically spaced frequency bins (in Hz).
            - filtered_yf: Median-filtered PSD magnitudes (linear scale), optionally normalized.
    """
    n = len(waveform)
    yf = np.square(np.abs(fft.rfft(waveform))) / n / rate
    xf = fft.rfftfreq(n, 1 / rate)
    log_xf = np.logspace(np.log10(band[0]), np.log10(band[1]), num=outlength)
    filtered_yf = np.zeros(len(log_xf), dtype=yf.dtype)
    starts = np.searchsorted(xf, log_xf / 2 ** (window / 2), side="left")
    ends = np.searchsorted(xf, log_xf * 2 ** (window / 2), side="right")
    for i, (start, end) in enumerate(zip(starts, ends)):
        filtered_yf[i] = np.median(yf[start:end])
    if norm:
        filtered_yf /= np.max(filtered_yf)
    return log_xf, filtered_yf


def octave_centers(start, stop):
    """
    Calculate 1/3 octave center frequencies within a specified range.
    Parameters
    ----------
    start : float
        The lower frequency bound (in Hz).
    stop : float
        The upper frequency bound (in Hz).
    Returns
    -------
    numpy.ndarray
        Array of 1/3 octave center frequencies (rounded to nearest 10 Hz) within the specified range,
        according to ISO 266 standard.
    Notes
    -----
    The center frequencies are calculated using the formula:
        f_c = 1000 * 2^(n/3)
    where n is an integer such that the resulting frequency is within [start, stop].
    """

    # 1/3 octave center frequencies (ISO 266)
    # f_c = 1000 * 2^(n/3), n integer
    centers = []
    n_start = int(np.ceil(3 * np.log2(start / 1000)))
    n_stop = int(np.floor(3 * np.log2(stop / 1000)))
    for n in range(n_start, n_stop + 1):
        f = round(1000 * 2 ** (n / 3) / 10) * 10
        if start <= f <= stop:
            centers.append(f)
    return np.array(centers)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.generators import pink_noise, linsweep, logsweep
    from matplotlib.ticker import FixedLocator, FixedFormatter
    import mplcursors

    RATE = 96000
    START = 20
    STOP = 20000
    waveform = pink_noise(120, RATE, (500, 5000))  # Example waveform
    # waveform = logsweep(120, RATE, (10, 30000))  # Example waveform

    xf, yf = calc_psd(waveform, RATE)

    # Apply correction factor for pink noise 3db/octave
    yf *= xf / xf[0]

    # yf = yf / np.median(yf)  # Normalize to [0, 1]

    # Plotting the results
    plt.semilogx(xf, 10 * np.log10(yf))
    plt.title("FFT of Waveform")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dB]")
    plt.grid(which="both", axis="both")
    mplcursors.cursor(multiple=True)
    plt.show()
