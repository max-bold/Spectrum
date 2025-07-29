import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import mplcursors

fs = 44100*4  # Sampling frequency

# Coeffs for pink noise approximation
b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
a = [1, -2.494956002, 2.017265875, -0.522189400]

w,h = signal.freqz(b, a, worN=1024, fs=fs)

plt.semilogx(w, 20*np.log10(h))
mplcursors.cursor(multiple = True)
plt.show()
