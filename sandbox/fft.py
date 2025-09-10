import numpy as np
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt

RATE = 96000
LENGTH = 3  # seconds

t = np.linspace(0, LENGTH, RATE * LENGTH)

signal = np.random.uniform(-0.1, 0.1, RATE * LENGTH)  # Random signal
signal += np.sin(2 * np.pi * 440 * t)  # Add a 440 Hz sine wave
signal += np.sin(2 * np.pi * 1300 * t) * 0.3  # Add a 1300 Hz sine wave
signal += np.sin(2 * np.pi * 3000 * t) * 0.1  # Add a 3000 Hz sine wave

yf = 2 * np.abs(rfft(signal)) / len(signal)
xf = rfftfreq(len(signal), 1 / RATE)

log_f = np.logspace(np.log10(20), np.log10(20e3), 1000)
fft_step = RATE / len(signal)
w_start = (log_f / 2 ** (1 / 30) / fft_step).astype(int)
w_end = (log_f * 2 ** (1 / 30) / fft_step).astype(int)
log_y = np.zeros_like(log_f)
for i, start, end in zip(range(len(log_f)), w_start, w_end):
    log_y[i] = np.median(yf[start:end])

plt.semilogx(xf, 20 * np.log10(yf), label="FFT of signal")
plt.semilogx(log_f, 20 * np.log10(log_y), label="Filtered FFT")
plt.axis([20, 20e3, -100, 0])
plt.legend()
plt.show()
