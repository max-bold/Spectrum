import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt
from matplotlib.axes import Axes

fs = 100000
T = 1  # seconds
f1 = 20
f2 = 20000
t = np.linspace(0, T, int(fs * T))
N = len(t)
L = T / np.log(f2 / f1)
sweep = np.sin(2 * np.pi * f1 * L * (np.exp(t / L) - 1))
output = sweep

# For example filter out to get in
sos = butter(4, (100, 1000), btype="bandpass", fs=fs, output="sos")
input = sosfilt(sos, output)

xf = rfftfreq(N, 1 / fs)
yf_out = 2 * np.abs(np.array(rfft(output))) / N
yf_in = 2 * np.abs(np.array(rfft(input))) / N
yf = yf_in / yf_out

_, axs = plt.subplots(2, 1, figsize=(8, 8))
axs: list[Axes]
axs[0].plot(t, output)
axs[0].plot(t, input)
axs[1].plot(xf, 20 * np.log10(yf_out.clip(1e-12)), label="output")
axs[1].plot(xf, 20 * np.log10(yf_in.clip(1e-12)), label="input")
axs[1].plot(xf, 20 * np.log10(yf.clip(1e-12)), label="response")
axs[1].set(xscale="log", xlim=[10, 20000])
axs[1].legend()
plt.show()
