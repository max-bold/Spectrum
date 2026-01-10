import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, periodogram
from progress.bar import ChargingBar
from time import sleep
import sys

sys.path.append("C:\\Files\\Code\\Spectrum\\")
from utils.fft import grid_periodogram

SAMPLERATE = 96000  # Hz
BUFFER_SIZE = 4096  # samples
BAND = (20, 20000)  # Hz
TIME = 30  # seconds
FADE = int(1 * SAMPLERATE)  # fade length in samples
PADDING = int(0.5 * SAMPLERATE)  # padding length in samples

# Generate logarithmic chirp signal
t = np.arange(0, TIME, 1 / SAMPLERATE)
r = np.log(BAND[1] / BAND[0])
k = TIME / r
phi = 2 * np.pi * BAND[0] * k * (np.exp(t / k) - 1)
sweep = (0.5 * np.sin(phi)).astype(np.float32)

# Add fading to reduce clicks
sweep[:FADE] *= np.linspace(0, 1, FADE)
sweep[-FADE:] *= np.linspace(1, 0, FADE)

# Pad signal with silence
signal = np.pad(sweep, (PADDING, PADDING), "constant")

# Build Farina inverse filter
env = np.exp(t / k)
inv_filter = sweep[::-1] / env

# Play and record the signal
record = sd.playrec(
    signal,
    samplerate=SAMPLERATE,
    channels=1,
    device=(35, 30),
    blocksize=BUFFER_SIZE,
)[:, 0]
with ChargingBar("Recording", max=TIME * 10) as bar:
    for _ in range(TIME * 10):
        sleep(0.1)
        bar.next()
sd.wait()

# # Plot input signal and recorded signal
# plt.plot(signal, label="Signal")
# plt.plot(record[:, 0], label="Recorded")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# # Plot periodogram for debugging
# f, s_Pxx = periodogram(signal, SAMPLERATE)
# _, r_Pxx = periodogram(record[:, 0], SAMPLERATE)
# s_Pxx *= f
# r_Pxx *= f
# mask = (f >= BAND[0]) & (f <= BAND[1])
# plt.semilogx(f[mask], 10 * np.log10(s_Pxx[mask].clip(1e-20)), label="Signal")
# plt.semilogx(f[mask], 10 * np.log10(r_Pxx[mask].clip(1e-20)), label="Recorded")
# plt.title("Input Signal and Recorded Signal Periodograms")
# plt.legend()
# plt.grid(True, which="both", ls="--")
# plt.show()

# # Normalize record and filter
# record = (record / np.max(np.abs(record)))[:, 0].astype(np.float64)
# inv_filter = (inv_filter / np.max(np.abs(inv_filter))).astype(np.float64)

# Compute impulse response via convolution with inverse filter
h = fftconvolve(record, inv_filter, mode="full")
time_axis = np.arange(len(h)) / SAMPLERATE

# Find IR time
t1 = np.argmax(np.abs(h)) / SAMPLERATE

# Number of harmonics to analyze
HN = 10

# Print expected harmonic times
for n in range(HN):
    tn = t1 - TIME / r * np.log(n + 1)
    print(f"{n+1}th Harmonic expected at: {tn:.3f}s")

# Plot impulse response
plt.plot(time_axis, h)
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True, which="both")
plt.show()

# Plot harmonics spectra
log_f = np.geomspace(BAND[0], BAND[1], 1024)
THDn = np.zeros_like(log_f)
H1 = np.zeros_like(log_f)
for n in range(HN):
    st = t1 - TIME / r * np.log(n + 1.5)
    et = t1 - TIME / r * np.log(n + 0.5)
    mask = (time_axis >= st) & (time_axis < et)
    # print(f"{n+1}th IR length is {len(h[mask])}")
    f, Pxx = grid_periodogram(h[mask], SAMPLERATE, log_f, window="hann")
    if n > 0:
        THDn += Pxx
    else:
        H1 = Pxx
    # f,Pxx = periodogram(h[mask], SAMPLERATE)
    # plt.semilogx(f, 10 * np.log10(Pxx.clip(1e-20)), label=f"{n+1}th Harmonic")

# Calculate and plot noise floor
et = t1 - TIME / r * np.log(HN + 0.5)
noise_mask = time_axis < et
f, Pxx = grid_periodogram(h[noise_mask], SAMPLERATE, log_f, window="hann")
THDn += Pxx
# f,Pxx = periodogram(h[noise_mask], SAMPLERATE)
# plt.semilogx(f, 10 * np.log10(Pxx.clip(1e-20)), label="Noise Floor")

# plt.title("Harmonics Spectrum")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power/Frequency [dB/Hz]")
# plt.grid(True, which="both")
# plt.legend()
# plt.show()

#
THDn = np.sqrt(THDn / H1)
plt.semilogx(f, THDn*100)
# plt.title("THD+n")
plt.xlabel("Frequency [Hz]")
plt.ylabel("THD+n [%]")
plt.grid(True, which="both")
plt.show()
