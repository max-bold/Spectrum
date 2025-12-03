from operator import le
from scipy.signal import periodogram, chirp, fftconvolve
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from progress.bar import ChargingBar

SAMPLE_RATE = 96000  # Hz
PADDING = int(0.5 * SAMPLE_RATE)
BUFFER_SIZE = 4096  # samples
BAND = (20, 20000)  # Hz
TIME = 10  # seconds

# Generate logarithmic chirp signal
t = np.arange(0, TIME, 1 / SAMPLE_RATE)
k = TIME / np.log(BAND[1] / BAND[0])
phi = 2 * np.pi * BAND[0] * k * (np.exp(t / k) - 1)
# signal = np.array(
#     chirp(t, f0=BAND[0], f1=BAND[1], t1=TIME, method="logarithmic", phi=90) * 0.5,
#     dtype=np.float32,
# )
signal = 0.5 * np.sin(phi).astype(np.float32)

# Pad signal with silence
signal = np.pad(signal, (PADDING, PADDING), "constant", constant_values=(0, 0))

# Add fade in and fade out
fade_length = int(0.1 * SAMPLE_RATE)
fade_in = np.linspace(0, 1, fade_length)
fade_out = np.linspace(1, 0, fade_length)
signal[PADDING : fade_length + PADDING] *= fade_in
signal[-fade_length - PADDING : -PADDING] *= fade_out

# Init record array
record = np.zeros_like(signal)

# Plot signal peridiogram for debuging
# f,Pxx = periodogram(signal, SAMPLE_RATE)
# plt.semilogx(f, 10 * np.log10(Pxx.clip(1e-20)))
# plt.title("Input Signal Periodogram")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power/Frequency [dB/Hz]")
# plt.grid(True, which="both", ls="--")
# plt.show()

# Play and record the signal
print("Starting playback and recording...")
iostream = sd.Stream(
    samplerate=SAMPLE_RATE, channels=2, device=(19, 17), blocksize=BUFFER_SIZE
)
N = len(signal)
iostream.start()
with ChargingBar("Processing", max=N // BUFFER_SIZE) as bar:
    for start in range(0, N, BUFFER_SIZE):
        end = min(start + BUFFER_SIZE, N)
        iostream.write(np.column_stack((signal[start:end], signal[start:end])))
        record[start:end] = iostream.read(end - start)[0][:, 0]
        bar.next()
iostream.stop()
iostream.close()
print("Playback and recording finished.")

# Plot signal and record peridiogram for debuging
# f, signal_Pxx = periodogram(signal, SAMPLE_RATE)
# f, record_Pxx = periodogram(record, SAMPLE_RATE)
# mask = (f >= 20) & (f <= 20000)
# plt.semilogx(f[mask], 10 * np.log10(signal_Pxx[mask].clip(1e-20)), label="Input Signal")
# plt.semilogx(
#     f[mask], 10 * np.log10(record_Pxx[mask].clip(1e-20)), label="Recorded Signal"
# )
# plt.title("Input and Recorded Signal Periodogram")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Power/Frequency [dB/Hz]")
# plt.grid(True, which="both", ls="--")
# plt.show()

# Build Farina inverse filter
inv_filter = signal[PADDING:PADDING+len(t)][::-1]
inv_envelope = np.exp(t / k)
inv_filter *= inv_envelope

# Normalize record and filter
# record /= np.max(np.abs(record))
# inv_filter /= np.max(np.abs(inv_filter))

# Convolve recorded signal with inverse filter
h = fftconvolve(record, inv_filter, mode="full")
# h = h[::-1]  # Reverse impulse response
# Plot the impulse response
t_h = np.arange(len(h)) / SAMPLE_RATE
plt.plot(t_h, h)
plt.title("Impulse Response")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Split harmonics and calculate fft of each harmonic

for i in range(1, 5):
    window = int(2.0 * SAMPLE_RATE)
    t1 = TIME + k * np.log(i) + PADDING / SAMPLE_RATE
    t2 = TIME + k * np.log(i + 1) + PADDING / SAMPLE_RATE
    print(t1, t2)
    start = int(t1 * SAMPLE_RATE)
    end = min(int(t2 * SAMPLE_RATE), len(h))
    f, Pxx = periodogram(h[start:end], fs=SAMPLE_RATE, scaling="spectrum")
    Pxx *= f * f * f
    plt.semilogx(f, 10 * np.log10(Pxx.clip(1e-20)), label=f"{i}x Harmonic")
plt.title("Harmonic Spectra")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power/Frequency [dB/Hz]")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()
