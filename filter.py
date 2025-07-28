from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sosfiltfilt

# b, a = signal.butter(1, 100, "low", analog=True)
# w, h = signal.freqs(b, a)
# sos1 = signal.butter(4, [500, 5000], btype="band", fs=44100, output="sos")
sos2 = signal.butter(1, 1, btype="low", fs=44100, output="sos")
sos3 = signal.bessel(1,1, btype="low", fs=44100, output="sos")
# # Combine the two sos filters by stacking them
# sos_combined = np.vstack([sos1, sos2])
w, h = signal.freqz_sos(sos2, worN=1024, fs=44100)

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title("Butterworth filter frequency response")
plt.xlabel("Frequency [rad/s]")
plt.ylabel("Amplitude [dB]")
plt.margins(0, 0.1)
plt.grid(which="both", axis="both")
plt.axvline(500, color="green")
plt.axvline(5000, color="green")  # cutoff frequency
plt.show()
