from numpy import fft
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, chirp, fftconvolve
from progress.bar import ChargingBar
from time import sleep

SAMPLERATE = 96000  # Hz
BUFFER_SIZE = 4096  # samples
BAND = (20, 20000)  # Hz
TIME = 10  # seconds
FADE = int(0.1 * SAMPLERATE)  # fade length in samples
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
    channels=2,
    device=(19, 17),
    blocksize=BUFFER_SIZE,
)
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

# Get start of record (removing delay and padding)
rec_start = np.argmax(np.abs(record[:, 0]) > 0.05) / SAMPLERATE
# print(f"Record starts at {rec_start / SAMPLERATE:.3f}s")

# t1 = TIME + rec_start
# t2 = t1 - t1/r * np.log(2)
# t3 = t1 - t1/r * np.log(3)
# t4 = t1 - t1/r * np.log(4)
# print(
#     f"Expected reflections at: {t1:.3f}s (direct), {t2:.3f}s (1st), {t3:.3f}s (2nd), {t4:.3f}s (3rd)"
# )

# Normalize record and filter
record = (record / np.max(np.abs(record)))[:, 0].astype(np.float64)
inv_filter = (inv_filter / np.max(np.abs(inv_filter))).astype(np.float64)

# Compute impulse response via convolution with inverse filter
h = fftconvolve(record, inv_filter, mode="full")

nw = int(0.2 * SAMPLERATE)
n1 = np.argmax(np.abs(h))
n2 = np.argmax(np.abs(h[: n1 - nw]))
n3 = np.argmax(np.abs(h[: n2 - nw]))
n4 = np.argmax(np.abs(h[: n3 - nw]))
ns = [n1, n2, n3, n4]

t1 = n1 / SAMPLERATE
t2 = n2 / SAMPLERATE
t3 = n3 / SAMPLERATE
t4 = n4 / SAMPLERATE
print(
    f"Expected reflections at: {t1:.3f}s (direct), {t2:.3f}s (1st), {t3:.3f}s (2nd), {t4:.3f}s (3rd)"
)
# Plot impulse response
time_axis = np.arange(len(h)) / SAMPLERATE
plt.plot(time_axis, h)
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Plot harmonics spectra
for i, n in enumerate(ns):
    f, Pxx = periodogram(
        h[n - nw : n + nw], SAMPLERATE, scaling="spectrum", window="hann"
    )
    f_mask = (f >= 20) & (f <= 20000)
    plt.semilogx(
        f[f_mask], 10 * np.log10(Pxx[f_mask].clip(1e-20)), label=f"{i+1}th Harmonic"
    )

# Calculate and plot noise floor
f, Pxx = periodogram(h[: n4 - nw], SAMPLERATE, scaling="spectrum", window="hann")
f_mask = (f >= 20) & (f <= 20000)
plt.semilogx(f[f_mask], 10 * np.log10(Pxx[f_mask].clip(1e-20)), label="Noise Floor")


plt.title("Harmonics Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
