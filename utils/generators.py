import numpy as np
from numpy.typing import NDArray
from scipy.signal import chirp, butter, sosfilt


def log_sweep(
    n: int, fs: int = 44100, band: tuple[float, float] = (20, 20e3)
) -> NDArray[np.float32]:

    ts = np.arange(n)
    t1 = ts[-1]
    f0 = band[0] / fs
    f1 = band[1] / fs
    sweep = chirp(ts, f0, t1, f1, method="logarithmic", phi=-90) * 0.9
    return np.column_stack((sweep, sweep)).astype(np.float32)


def pink_noise(
    n: int, fs: int = 44100, band: tuple[float, float] = (20, 20e3)
) -> NDArray[np.float32]:
    band_sos = butter(4, band, "bandpass", False, "sos", fs)
    pinking_sos = [
        [0.04992203, -0.00539063, 0.0, 1.0, -0.55594526, 0.0],
        [1.0, -1.81488818, 0.81786161, 1.0, -1.93901074, 0.93928204],
    ]
    combined_sos = np.vstack([pinking_sos, band_sos])  # type: ignore
    white = np.random.uniform(-1, 1, n)
    pink = sosfilt(combined_sos, white)
    pink /= np.max(np.abs(pink)) / 0.9
    return np.column_stack((pink, pink)).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import welch, periodogram

    fs = 96000
    n = 5 * fs
    signal = log_sweep(n, fs)
    # signal = pink_noise(n, fs)

    ts = np.linspace(0, 5, n)
    # plt.plot(ts, signal[:, 0])

    f, p = periodogram(signal[:, 0], fs)
    plt.semilogx(f, 10 * np.log10((f * p).clip(1e-20)))
    plt.show()
