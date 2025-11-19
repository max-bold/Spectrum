import numpy as np
from threading import Thread, Event
from scipy.signal import (
    welch,
    periodogram,
)
from .classes import RefMode, AnalyzerMode, WeightingMode
from .windows import log_filter, Windows


class Analyzer(Thread):
    def __init__(
        self,
    ) -> None:
        self.analyzer_mode: AnalyzerMode = AnalyzerMode.PERIODIOGRAM
        self.ref: RefMode = RefMode.CHANNEL_B
        self.weighting: WeightingMode = WeightingMode.NONE
        self.welch_n: int = 2**13
        self.sample_rate: int = 96000
        self.record: np.ndarray = np.empty((0, 2))
        self.freq_length: int = 1024
        self.window_func: Windows = Windows.GAUSSIAN
        self.window_width: float = 1 / 10
        self.running = Event()
        self.completed = Event()
        self.result: np.ndarray = np.empty(0)
        self.band: tuple[float, float] = (20, 20000)

        return super().__init__(daemon=True)

    def run(self) -> None:
        while True:
            self.running.wait()
            if self.analyzer_mode == AnalyzerMode.PERIODIOGRAM:
                x, p = periodogram(self.record, self.sample_rate, axis=0)
            elif self.analyzer_mode == AnalyzerMode.WELCH:
                n = min(self.welch_n, len(self.record))
                x, p = welch(self.record, self.sample_rate, "hann", n, axis=0)
            else:
                raise ValueError(f"Unknown mode: {self.analyzer_mode}")
            if self.ref == RefMode.NONE:
                fft = p[:, 0]
            elif self.ref == RefMode.CHANNEL_B:
                fft = p[:, 0] / p[:, 1]
            else:
                raise ValueError(f"Unknown reference mode: {self.ref}")
            if self.weighting == WeightingMode.PINK:
                fft *= x
            log_f, log_p = log_filter(
                x, fft, self.window_func, self.window_width, self.freq_length, self.band
            )
            self.result = np.vstack((log_f, 10 * np.log10(log_p.clip(1e-20))))
            self.completed.set()
            self.running.clear()


if __name__ == "__main__":
    pass
