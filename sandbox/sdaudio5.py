import scipy.signal
import sounddevice as sd
import numpy as np
import scipy
from time import sleep
import matplotlib.pyplot as plt
from scipy.signal import periodogram

record = np.empty((0, 2), np.float32)
record_s = 0
signal = np.empty((0, 2), np.float32)
signal_s = 0


def input_callback(indata: np.ndarray, frames, time, status) -> None:
    global record, record_s
    s = record_s
    e = min(s + frames, len(record))
    record[s:e] = indata[: e - s]
    record_s = e


def output_callback(outdata: np.ndarray, frames, time, status) -> None:
    global signal, signal_s
    s = signal_s
    e = min(s + frames, len(signal))
    outdata.fill(0)
    outdata[: e - s] = signal[s:e]
    signal_s = e


if __name__ == "__main__":

    t = 30  # seconds

    print(sd.query_devices())

    in_stream = sd.InputStream(callback=input_callback, device=19)
    in_fs = in_stream.samplerate
    record = np.zeros((int(in_fs * t), 2), np.float32)

    out_stream = sd.OutputStream(callback=output_callback, device=16)
    out_fs = out_stream.samplerate
    ts = np.linspace(0, t, int(t * out_fs))
    chirp = (
        scipy.signal.chirp(ts, f0=20, t1=t, f1=30000, method="logarithmic", phi=90)
        * 0.5
    )
    sin = np.sin(2 * np.pi * ts * 100) * 0.5
    signal = np.column_stack([chirp, chirp]).astype(np.float32)
    signal = np.pad(signal, ((int(out_fs * 0.1), 0), (0, 0)))

    in_stream.start()
    out_stream.start()

    while record_s < len(record) or signal_s < len(signal):
        sleep(0)

    in_stream.stop()
    out_stream.stop()

    a, b = record.T
    print(f"{in_fs=}, {out_fs=}")

    # plt.plot(ts, a)
    # plt.plot(ts, b)

    fs, Pxx_a = periodogram(a, fs=in_fs)
    fs, Pxx_b = periodogram(b, fs=in_fs)

    Pxx_a*=fs
    Pxx_b*=fs

    plt.semilogx(fs, 10 * np.log10(Pxx_a.clip(1e-20)))
    plt.semilogx(fs, 10 * np.log10(Pxx_b.clip(1e-20)))

    plt.grid(True, "both")
    plt.show()
