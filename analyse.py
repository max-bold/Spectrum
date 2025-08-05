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
    n = len(waveform)
    yf = np.square(np.abs(fft.rfft(waveform))) / n / rate
    xf = fft.rfftfreq(n, 1 / rate)

    # Interpolate to log scale for filtering
    # interp_func = interp1d(xf, yf, kind="linear", bounds_error=False, fill_value=0)
    # start = max(band[0], xf[0])
    # stop = min(band[1], xf[-1])
    # num = np.sum((xf >= band[0]) & (xf <= band[1]))
    log_xf = np.logspace(np.log10(band[0]), np.log10(band[1]), num=outlength)
    # log_yf = interp_func(log_xf)

    # Ensure window size is odd for median filtering
    # if window % 2 == 0:
    #     window += 1

    # Apply median filter
    # filtered_log_yf = signal.medfilt(log_yf, window)
    # filtered_log_yf = signal.savgol_filter(log_yf, window, 3, mode="nearest")

    # if norm:
    #     # Normalize the filtered spectrum to the range [0, 1]
    #     filtered_log_yf /= np.max(filtered_log_yf)

    # return log_xf, filtered_log_yf
    # start = np.searchsorted(xf, band[0], side="left")  # включительно
    # end = np.searchsorted(xf, band[1], side="right")  # включительно
    # yf = signal.medfilt(yf, window)
    # yf = signal.savgol_filter(yf, window, 3, mode="nearest")
    # yf/=np.max(yf[start:end])
    # window = 1 / 10
    filtered_yf = np.zeros(len(log_xf), dtype=yf.dtype)
    for i in range(len(log_xf)):
        f = log_xf[i]
        start = np.searchsorted(xf, f / 2 ** (window / 2), side="left")
        end = np.searchsorted(xf, f * 2 ** (window / 2), side="right")
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
    from gen import pink_noise, linsweep, logsweep
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
