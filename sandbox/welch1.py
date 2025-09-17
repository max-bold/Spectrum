import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

fs = 96000
n = int(10 * fs)
noise = np.random.uniform(-1, 1, n)
noise = np.column_stack((noise, noise))
f, p = welch(noise, fs, nperseg=fs, axis=0)
f1 = np.fft.rfftfreq(n,1/fs)
p1 = np.square(np.abs(np.fft.rfft(noise[:,0])))*2/n/fs
print(f.shape, p.shape)
print(f[-1] / len(p))

plt.semilogx(f, 10 * np.log10(p[:, 0].clip(1e-20)))
plt.semilogx(f1, 10 * np.log10(p1.clip(1e-20)),"o")
plt.grid(True, "both")
plt.show()
