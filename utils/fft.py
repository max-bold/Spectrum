from scipy.signal import periodogram, welch
import numpy as np
from windows import Windows, log_filter2, grid_filter


def log_periodogram(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (20, 20000),
    n_output: int = 100,
    window: str = "boxcar",
    scaling: str = "density",
    axis: int = -1,
    log_window_func: Windows = Windows.GAUSSIAN,
    log_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = periodogram(
        x, fs, window, return_onesided=True, scaling=scaling, axis=axis
    )
    return log_filter2(f, Pxx, band, log_window_func, log_window_width, n_output)


def log_welch(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (20, 20000),
    n_output: int = 100,
    window: str = "hann",
    nperseg: int = 256,
    scaling: str = "density",
    axis: int = -1,
    log_window_func: Windows = Windows.GAUSSIAN,
    log_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = welch(
        x, fs, window, nperseg, return_onesided=True, scaling=scaling, axis=axis
    )
    return log_filter2(f, Pxx, band, log_window_func, log_window_width, n_output)


def grid_periodogram(
    x: np.ndarray,
    fs: float,
    out_f: np.ndarray,
    window: str = "boxcar",
    scaling: str = "density",
    axis: int = -1,
    log_window_func: Windows = Windows.GAUSSIAN,
    log_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = periodogram(
        x, fs, window, return_onesided=True, scaling=scaling, axis=axis
    )
    return grid_filter(f, Pxx, out_f, log_window_func, log_window_width)


def grid_welch(
    x: np.ndarray,
    fs: float,
    out_f: np.ndarray,
    window: str = "hann",
    nperseg: int = 256,
    scaling: str = "density",
    axis: int = -1,
    log_window_func: Windows = Windows.GAUSSIAN,
    log_window_width: float = 1 / 3,
) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = welch(
        x, fs, window, nperseg, return_onesided=True, scaling=scaling, axis=axis
    )
    return grid_filter(f, Pxx, out_f, log_window_func, log_window_width)
