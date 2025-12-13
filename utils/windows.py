import numpy as np
from typing import Literal
from .classes import ListableEnum


class Windows(ListableEnum):
    """Enumeration of available window functions for frequency domain filtering.

    Available window types:
        FLAT: Rectangular window (uniform weighting)
        COSINE: Cosine-shaped window for smooth transitions
        GAUSSIAN: Bell-shaped window with exponential decay
        TRIANGULAR: Linear decay window with constant dB/octave falloff
    """

    FLAT = "flat"
    COSINE = "cosine"
    GAUSSIAN = "gaussian"
    TRIANGULAR = "triangular"


def log_window(
    window: Windows,
    fc: float,
    df: float,
    w: float,
) -> tuple[np.ndarray, int, int]:
    """Generate a logarithmic frequency window centered at a specific frequency.

    Creates a window function in the frequency domain with logarithmic spacing,
    useful for frequency analysis where equal ratios (octaves) are more important
    than equal differences.

    Args:
        window (Windows): Shape of the window function:
            - FLAT: Rectangular window (uniform weighting)
            - COSINE: Cosine-shaped window for smooth transitions
            - GAUSSIAN: Bell-shaped window with exponential decay
            - TRIANGULAR: Linear decay window with constant dB/octave falloff
        fc (float): Central frequency in Hz around which the window is centered
        df (float): Frequency step size (sampling_rate / fft_size)
        w (float): Window width in octaves (total width, not half-width)

    Returns:
        tuple[np.ndarray, int, int]: A tuple containing:
            - window_weights: Array of window weights for each frequency bin
            - start_index: Starting index in the frequency array
            - end_index: Ending index in the frequency array

    Raises:
        ValueError: If an unknown window type is specified
    """
    # Calculate half-width for symmetric window around center frequency
    half_width = w / 2

    # Calculate frequency bounds (min and max frequencies)
    f_min = fc / (2**half_width)
    f_max = fc * (2**half_width)

    # Generate frequency array within the window bounds
    frequencies = np.arange(f_min, f_max, df)

    # Convert to logarithmic scale relative to center frequency
    log_frequencies = np.log2(frequencies) - np.log2(fc)

    # Generate window weights based on the specified window type
    if window == Windows.FLAT:
        window_weights = np.ones_like(log_frequencies)

    elif window == Windows.GAUSSIAN:
        # Gaussian window with exponential decay
        window_weights = np.exp(-((log_frequencies / half_width * 4) ** 2) / 2)

    elif window == Windows.COSINE:
        # Cosine-shaped window for smooth transitions
        window_weights = np.cos(np.pi * log_frequencies / half_width) / 2 + 0.5

    elif window == Windows.TRIANGULAR:
        # Triangular window with constant dB/octave falloff
        decay_factor = 10 ** (-30 / half_width / 10)
        window_weights = np.power(decay_factor, np.abs(log_frequencies))

    else:
        raise ValueError(f"Unknown window shape: {window}")

    # Calculate array indices for the window
    start_index = int(np.rint(f_min / df))
    end_index = start_index + len(window_weights)

    return window_weights, start_index, end_index


def log_filter(
    frequency_array: np.ndarray,
    fft_data: np.ndarray,
    window_function: Windows = Windows.GAUSSIAN,
    window_width: float = 1 / 10,
    points: int = 1024,
    frequency_band: tuple[float, float] = (20, 20000),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply logarithmic frequency filtering to FFT data using windowed averaging.

    This function performs frequency domain smoothing by applying a window function
    at logarithmically spaced frequency points. Each output point represents a
    weighted average of nearby FFT bins, producing a smoothed frequency response
    that emphasizes perceptually relevant frequency relationships.

    The logarithmic spacing means that higher frequencies are analyzed with
    proportionally wider windows, which matches human auditory perception
    where frequency discrimination decreases at higher frequencies.

    Args:
        frequency_array (np.ndarray): Array of frequency values corresponding to FFT bins.
            Should be linearly spaced (e.g., from np.fft.fftfreq).
        fft_data (np.ndarray): Complex or magnitude FFT data to be filtered.
            Must have the same length as frequency_array.
        window_function (Windows, optional): Type of window function to apply for
            smoothing. Each window type provides different smoothing characteristics.
            Defaults to Windows.GAUSSIAN for smooth, bell-shaped weighting.
        window_width (float, optional): Width of the smoothing window in octaves.
            Smaller values provide more frequency resolution but less smoothing.
            Larger values provide more smoothing but less frequency detail.
            Defaults to 0.1 (1/10 octave).
        points (int, optional): Number of logarithmically spaced output points.
            More points provide higher frequency resolution in the output.
            Defaults to 1024.
        frequency_band (tuple[float, float], optional): Frequency range (min_freq, max_freq)
            in Hz to analyze. Points outside this range are ignored.
            Defaults to (20, 20000) covering typical audio range.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - output_frequencies: Array of logarithmically spaced frequencies
              where filtering was applied
            - filtered_magnitudes: Array of filtered/smoothed FFT values at each
              output frequency point

    Raises:
        ValueError: If frequency_band values are invalid or if array dimensions don't match.

    Example:
        >>> import numpy as np
        >>> # Generate test signal and FFT
        >>> fs = 44100
        >>> t = np.linspace(0, 1, fs)
        >>> signal = np.sin(2 * np.pi * 1000 * t)  # 1 kHz sine wave
        >>> fft_result = np.fft.fft(signal)
        >>> freqs = np.fft.fftfreq(len(signal), 1/fs)
        >>>
        >>> # Apply logarithmic filtering
        >>> log_freqs, log_mags = log_filter(freqs, np.abs(fft_result))
    """
    # Generate logarithmically spaced frequency points for analysis
    output_frequencies = np.geomspace(frequency_band[0], frequency_band[1], points)

    # Calculate frequency step from input array (assumes linear spacing)
    frequency_step = frequency_array[1] - frequency_array[0]

    # Apply windowed filtering at each output frequency
    filtered_magnitudes = []

    for center_freq in output_frequencies:
        # Get window weights and indices for current center frequency
        window_weights, start_idx, end_idx = log_window(
            window=window_function,
            fc=center_freq,
            df=frequency_step,
            w=window_width,
        )

        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(fft_data))
        # Apply weighted average to FFT data within the window
        windowed_fft = (
            fft_data[start_idx:end_idx] * window_weights[: end_idx - start_idx]
        )
        weighted_sum = np.sum(windowed_fft)
        normalization_factor = np.sum(window_weights)

        # Store normalized result
        filtered_magnitudes.append(weighted_sum / normalization_factor)

    return output_frequencies, np.array(filtered_magnitudes)


def log_filter2(
    f: np.ndarray,
    Pxx: np.ndarray,
    band: tuple[float, float] = (20, 20000),
    window: Windows = Windows.GAUSSIAN,
    w: float = 1 / 3,
    n_output: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    log_f = np.geomspace(band[0], band[1], n_output)
    log_Pxx = np.zeros_like(log_f)
    df = f[1]
    for i, fc in enumerate(log_f):
        win, si, ei = log_window(window, fc, df, w)
        ei = min(ei, len(f))
        if si < ei:
            log_Pxx[i] = np.average(Pxx[si:ei], axis=-1, weights=win[: ei - si])
    return log_f, log_Pxx

def grid_filter(
        f:np.ndarray,
        Pxx:np.ndarray,
        grid:np.ndarray,
        window: Windows = Windows.GAUSSIAN,
        w: float = 1 / 3,
        n_output: int = 256,
)-> tuple[np.ndarray, np.ndarray]:
    """Aply frequency filtering using logarithmic windows centered at specified grid points."""
    grid_Pxx = np.zeros_like(grid)
    df = f[1]
    for i, fc in enumerate(grid):
        win, si, ei = log_window(window, fc, df, w)
        ei = min(ei, len(f))
        if si < ei:
            grid_Pxx[i] = np.average(Pxx[si:ei], axis=-1, weights=win[: ei - si])
    return grid, grid_Pxx


if __name__ == "__main__":
    pass
