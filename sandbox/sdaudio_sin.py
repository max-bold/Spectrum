from audioop import mul
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from numpy import ndarray
from time import sleep

info = sd.query_devices(14)
if isinstance(info, dict):
    fs = info["default_samplerate"]

    print(fs)

    length = 5

    ts = np.arange(length * fs)
    f = 100 / fs

    signal = np.sin(2 * np.pi * f * ts)*0.5
    mult = np.linspace(0, 1, int(fs / 20))
    signal[: len(mult)] = signal[: len(mult)] * mult
    signal = np.column_stack((signal, signal)).astype(np.float32)
    rec = np.zeros_like(signal)

    start = 0

    def sd_callback(
        indata: ndarray, outdata: ndarray, frames: int, time, status
    ) -> None:
        global start
        end = min(len(signal), start + frames)
        rec[start:end] = indata[: end - start]
        outdata[: end - start] = signal[start:end]
        start = end

    ws = sd.WasapiSettings(exclusive=True)
    stream = sd.Stream(
        device=(14, 12), extra_settings=ws, blocksize=1024, callback=sd_callback
    )

    stream.start()
    sleep(5)
    stream.stop()
    plt.plot(ts / fs, signal[:, 0])
    plt.plot(ts / fs, rec[:, 0])
    plt.grid(True, "both")
    plt.show()
