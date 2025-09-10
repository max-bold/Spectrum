import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import medfilt
from utils.generators import logsweep
from analyse import calc_psd

FREQ = 1000
RATE = 96000
LEN = 30

# noise = np.random.uniform(-1, 1, LEN*RATE)
noise = logsweep(LEN, RATE, (500, 5e3))

xf = rfftfreq(len(noise), 1 / RATE)
yf = np.square(np.abs(rfft(noise))) / LEN / RATE
yf /= np.max(yf)
yf *= xf / 1000


# Vectorized filtering
log_xf = np.logspace(np.log10(20), np.log10(20e3), 1000)
starts = np.searchsorted(xf, log_xf / 2, side="left")
ctrs = np.searchsorted(xf, log_xf, side="left")
ends = np.searchsorted(xf, log_xf * 2, side="right")

# Preallocate the weight matrix (sparse, but we fill only relevant parts)
filtered_yf = np.zeros_like(log_xf)
w_len = np.zeros_like(log_xf)
for i, (f, start, ctr, end) in enumerate(zip(log_xf, starts, ctrs, ends)):
    xfs = xf[start:end]
    w = np.zeros_like(xfs)
    s_ctr = ctr - start
    w[:s_ctr] = (xfs[:s_ctr] / f) ** 8
    w[s_ctr:] = (f / xfs[s_ctr:]) ** 8
    w_sum = np.sum(w)
    w /= w_sum
    filtered_yf[i] = np.sum(yf[start:end] * w)
    w_len[i] = len(w)

# yf /= np.max(filtered_yf)
# filtered_yf /= np.max(filtered_yf)
yf = medfilt(yf, 1001)
x2f, y2f = calc_psd(noise, RATE)
y2f *= x2f / x2f[0]
# fig, axs = plt.subplots(2)
ps = np.searchsorted(xf, 20, side="left")
pe = np.searchsorted(xf, 20e3, side="left")
plt.semilogx(xf[ps:pe], 10 * np.log10(yf[ps:pe]))
plt.semilogx(log_xf, 10 * np.log10(filtered_yf), label="Filtered")
plt.semilogx(x2f, 10 * np.log10(y2f), label="Filtered")
# axs[1].semilogx(log_xf, w_len, label="Weights Length")
plt.grid(True, which="both")
plt.show()
