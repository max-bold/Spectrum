"""Audio spectrum analyzer module.

This module provides real-time audio spectrum analysis capabilities using
various algorithms like periodogram and Welch's method.
"""

import numpy as np
from threading import Thread, Event
from scipy.signal import welch, periodogram
from .classes import RefMode, AnalyzerMode, WeightingMode
from .windows import log_filter, Windows


class Analyzer(Thread):
    """Real-time audio spectrum analyzer.

    A threaded analyzer that processes audio data using various spectral analysis
    methods (periodogram, Welch's method) with configurable parameters including
    reference modes, weighting, and windowing functions.

    Attributes:
        analyzer_mode (AnalyzerMode): Analysis method (PERIODIOGRAM or WELCH)
        ref (RefMode): Reference channel mode for ratio calculations
        weighting (WeightingMode): Frequency weighting mode
        welch_n (int): Number of samples for Welch's method
        sample_rate (int): Audio sampling rate in Hz
        record (np.ndarray): Input audio data buffer
        freq_length (int): Number of frequency bins in output
        window_func (Windows): Window function for frequency smoothing
        window_width (float): Width parameter for window function
        running (Event): Thread synchronization event for processing
        completed (Event): Thread synchronization event for completion
        result (np.ndarray): Analysis result [frequencies, magnitudes]
        band (tuple): Frequency range (min_freq, max_freq) in Hz
    """

    def __init__(self) -> None:
        """Initialize the spectrum analyzer with default settings.

        Sets up the analyzer with standard audio analysis parameters:
        - Periodogram analysis mode
        - Channel B reference
        - No frequency weighting
        - 96 kHz sample rate
        - 20Hz - 20kHz frequency range
        """
        # Analysis configuration
        self.analyzer_mode: AnalyzerMode = AnalyzerMode.PERIODIOGRAM
        self.ref: RefMode = RefMode.CHANNEL_B
        self.weighting: WeightingMode = WeightingMode.NONE

        # Signal processing parameters
        self.welch_n: int = 2**13  # 8192 samples for Welch's method
        self.sample_rate: int = 96000  # 96 kHz sampling rate
        self.freq_length: int = 1024  # Number of frequency bins

        # Window function parameters
        self.window_func: Windows = Windows.GAUSSIAN
        self.window_width: float = 0.1  # 1/10 octave width

        # Data buffers
        self.record: np.ndarray = np.empty((0, 2))  # Stereo audio buffer
        self.result: np.ndarray = np.empty(0)  # Analysis result

        # Frequency range (Hz)
        self.band: tuple[float, float] = (20.0, 20000.0)  # Human hearing range

        # Thread synchronization
        self.running = Event()  # Signal to start processing
        self.completed = Event()  # Signal when processing is done

        super().__init__(daemon=True)

    def run(self) -> None:
        """Main analyzer thread loop.

        Continuously processes audio data when triggered by the running event.
        Performs spectral analysis using the configured method and parameters,
        then applies reference mode, weighting, and logarithmic filtering.

        The results are stored in self.result as a 2D array where:
        - First row: frequency values (Hz)
        - Second row: magnitude values (dB)

        Raises:
            ValueError: If analyzer_mode or ref mode is unknown
        """
        while True:
            # Wait for signal to start processing
            self.running.wait()

            # Perform spectral analysis based on selected mode
            frequencies, power_spectrum = self._compute_spectrum()

            # Apply reference mode (channel comparison)
            processed_spectrum = self._apply_reference_mode(power_spectrum)

            # Apply frequency weighting if enabled
            if self.weighting == WeightingMode.PINK:
                processed_spectrum *= frequencies

            # Apply logarithmic frequency filtering and smoothing
            log_frequencies, log_power = log_filter(
                frequencies,
                processed_spectrum,
                self.window_func,
                self.window_width,
                self.freq_length,
                self.band,
            )

            # Convert to dB scale and store result
            magnitude_db = 10 * np.log10(log_power.clip(1e-20))  # Clip to avoid log(0)
            self.result = np.vstack((log_frequencies, magnitude_db))

            # Signal completion and reset
            self.completed.set()
            self.running.clear()

    def _compute_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute power spectrum using the selected analysis method.

        Returns:
            tuple[np.ndarray, np.ndarray]: Frequency array and power spectrum

        Raises:
            ValueError: If analyzer_mode is unknown
        """
        if self.analyzer_mode == AnalyzerMode.PERIODIOGRAM:
            return periodogram(self.record, self.sample_rate, axis=0)
        elif self.analyzer_mode == AnalyzerMode.WELCH:
            # Use minimum of configured size and available data
            welch_window_size = min(self.welch_n, len(self.record))
            return welch(
                self.record,
                self.sample_rate,
                window="hann",
                nperseg=welch_window_size,
                axis=0,
            )
        else:
            raise ValueError(f"Unknown analyzer mode: {self.analyzer_mode}")

    def _apply_reference_mode(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Apply reference mode for channel comparison.

        Args:
            power_spectrum (np.ndarray): Input power spectrum

        Returns:
            np.ndarray: Processed spectrum based on reference mode

        Raises:
            ValueError: If reference mode is unknown
        """
        if self.ref == RefMode.NONE:
            return power_spectrum[:, 0]  # Use only channel A
        elif self.ref == RefMode.CHANNEL_B:
            # Ratio of channel A to channel B
            return power_spectrum[:, 0] / power_spectrum[:, 1]
        else:
            raise ValueError(f"Unknown reference mode: {self.ref}")


if __name__ == "__main__":
    pass
