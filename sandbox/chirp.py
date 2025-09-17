from scipy.signal import chirp, periodogram, welch
import numpy as np
import matplotlib.pyplot as plt

rate: int = 96000
t1: float = 10
n = int(t1 * rate)
# t = np.linspace(0, t1, n)
sweep = chirp(np.arange(n), 100 / rate, n, 2000 / rate, "logarithmic")
xf = np.fft.rfftfreq(n, 1 / rate)
yf = np.abs(np.fft.rfft(sweep)) ** 2 * (xf) / n / rate
xf_p, yf_p = periodogram(sweep, rate)
yf_p *= xf_p
xf_w, yf_w = welch(sweep, rate, nperseg=1024 * 4 * 4 * 4)
yf_w *= xf_w
plt.semilogx(xf, 10 * np.log10(yf.clip(1e-20)), label="rfft")
plt.semilogx(xf_p, 10 * np.log10(yf_p.clip(1e-20)), label="periodogram")
plt.semilogx(xf_w, 10 * np.log10(yf_w.clip(1e-20)), label="welch")
plt.grid(True, "both")
plt.legend()
plt.xlim(20, 20e3)
plt.show()
