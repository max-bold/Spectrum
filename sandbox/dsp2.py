import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfilt

fs = 100_000
T = 1.0
f1, f2 = 20, 20_000

t = np.linspace(0, T, int(fs*T), endpoint=False)
N = len(t)
L = T / np.log(f2/f1)

# Exponential (log) sweep
sweep = np.sin(2*np.pi*f1*L*(np.exp(t/L) - 1.0))

# “Device under test”: here just a band-pass to illustrate
sos = butter(4, (100, 1000), btype="bandpass", fs=fs, output="sos")
y = sosfilt(sos, sweep)   # output
x = sweep                 # input

# Frequency axis and spectra
f = rfftfreq(N, d=1/fs)
X = rfft(x)
Y = rfft(y)

# Avoid divide-by-zero at DC etc.
eps = 1e-20
H = Y / (X + eps)

# Plots
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
axs[0].plot(t, x, label="input")
axs[0].plot(t, y, label="output", alpha=0.8)
axs[0].set_title("Time domain")
axs[0].legend()
axs[0].grid(True, which="both")

axs[1].semilogx(f, 20*np.log10(np.maximum(np.abs(H), 1e-12)), label="|H(f)| (dB)")
axs[1].set_xlim([10, 20_000])
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_ylabel("Magnitude (dB)")
axs[1].grid(True, which="both")
axs[1].legend()

plt.tight_layout()
plt.show()
