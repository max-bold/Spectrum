"""Experimental music-file analysis script."""

import scipy.integrate as sint
import numpy as np


def interate_log(x, y):
    return 10 * np.log10(np.sum(sint.trapezoid(y, x)))


def run_experiment(music_path: str) -> None:
    import matplotlib.pyplot as plt
    import soundfile
    from scipy.signal import periodogram
    from utils.windows import Windows, log_window
    from time import time

    data, fs = soundfile.read(music_path)
    x = data[60 * fs : 70 * fs, 0]
    f, p = periodogram(x, fs)
    df = f[1]
    plt.semilogx(f, 10 * np.log10((p * f).clip(1e-20)), linewidth=0.2)

    n = int(np.log(20e3 / 20) / np.log(df / 20 + 1))
    log_f = np.geomspace(20, 20e3, n)
    log_p = np.zeros_like(log_f)

    st = time()
    for i, fc in enumerate(log_f):
        w, s, e = log_window(Windows.GAUSSIAN, fc, df, 1 / 10)
        if e > len(p):
            w = w[: len(p) - e]
        log_p[i] = np.average(p[s:e], None, w)
    print(f"{n}: 1/{1 / (time() - st):.3f}")

    plt.semilogx(log_f, 10 * np.log10((log_p * log_f).clip(1e-20)))

    print(interate_log(f, p), interate_log(log_f, log_p))
    plt.xlim(20, 20e3)
    plt.ylim(-100, 0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [dbv/Hz]")
    plt.xticks([20, 100, 1000, 10000, 20000], ["20", "100", "1k", "10k", "20k"])
    plt.grid(True, "both")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Uncomment and adjust the path when using this file as a standalone experiment.
    # run_experiment(r"D:\Music\example.mp3")
    raise SystemExit("Set music_path in sandbox/music.py before running this experiment.")
