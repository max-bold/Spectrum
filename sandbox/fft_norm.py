import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Setting sweep params
rate: int = 96000
chunksize: int = 4096
band = np.asarray((50, 2000), np.float64) / rate
t_end = 30
length = int(t_end * rate)
nk = np.sqrt(length * rate / np.log10(band[1] / band[0]) / 10) * 10 ** (0.35726 / 20)

# Generating sweep
k = length * band[0] / np.log(band[1] / band[0])
l = length / np.log(band[1] / band[0])
t = np.arange(0, length, dtype=np.float64)
sweep = np.sin(2 * np.pi * k * (np.exp(t / l) - 1))

# Initialise plt in interactive mode
plt.ion()
_, ax = plt.subplots()
ax: Axes = ax
ax.grid(True, which="both")
(line1,) = ax.semilogx([], [])
(line2,) = ax.semilogx([], [])
line1: Line2D = line1
line2: Line2D = line2
ax.set_xlim(20, 20e3)
ax.set_ylim(-60, 60)


for start in range(0, length, chunksize):
    end = min(start + chunksize, int(length))
    chunk = sweep[:end]

    xf = np.fft.rfftfreq(len(chunk), 1 / rate)

    yf = np.abs(np.fft.rfft(chunk))

    yf1 = 2 * yf / len(chunk)
    # pinked_yf1 = yf1 * np.sqrt(xf)
    log_yf1 = 20 * np.log10(yf1.clip(1e-12, None))
    line1.set_data(xf, log_yf1)

    yf2 = yf / nk
    # pinked_yf2 = yf2 * np.sqrt(xf)
    log_yf2 = 20 * np.log10(yf2.clip(1e-12, None))
    line2.set_data(xf, log_yf2)

    plt.draw()
    plt.pause(0.1)

plt.ioff()
plt.show()
