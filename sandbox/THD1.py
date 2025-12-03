import sounddevice as sd
import numpy as np
from scipy.fft import rfftfreq
from scipy.signal import periodogram
import matplotlib.pyplot as plt
from progress.bar import ChargingBar

SAMPLE_RATE = 96000  # Hz
PADDING = int(0.5 * SAMPLE_RATE)
BUFFER_SIZE = 4096  # samples
STEP_DURATION = 30  # cycles

f_centers = np.geomspace(20, 20000, 500)
step_lengths = np.zeros_like(f_centers, dtype=int)
num_periods = np.zeros_like(f_centers, dtype=int)

# Initial calculation of step lengths based on desired center frequencies
for i, fc in enumerate(f_centers):
    period_length = int(SAMPLE_RATE / fc)
    num_periods[i] = max(STEP_DURATION, 1024 // period_length)


# Adjust center frequencies to rfft bin centers in iterative manner
for ii in range(3):
    for i, (fc, n) in enumerate(zip(f_centers, num_periods)):
        length = n * int(SAMPLE_RATE / fc)
        freqs = rfftfreq(length, 1 / SAMPLE_RATE)
        fc_idx = np.argmin(np.abs(freqs - fc))
        f_centers[i] = freqs[fc_idx]
        step_lengths[i] = int(n * SAMPLE_RATE / freqs[fc_idx])

# Calculate total signal length and build input and output arrays
signal_length = np.sum(step_lengths) + 2 * PADDING
signal = np.zeros(signal_length, dtype=np.float32)
record = np.zeros(signal_length, dtype=np.float32)
print(f"Total signal length is: {(signal_length / SAMPLE_RATE):.2f} s")

# Generate the stepped sine wave signal
for i, (fc, length) in enumerate(zip(f_centers, step_lengths)):
    t = np.arange(length) / SAMPLE_RATE
    start_idx = PADDING + np.sum(step_lengths[:i])
    signal[start_idx : start_idx + length] = 0.5 * np.sin(2 * np.pi * fc * t)

# Add fade in and fade out
fade_length = int(0.1 * SAMPLE_RATE)
fade_in = np.linspace(0, 1, fade_length)
fade_out = np.linspace(1, 0, fade_length)
signal[PADDING : fade_length + PADDING] *= fade_in
signal[-fade_length - PADDING : -PADDING] *= fade_out

# Play and record the signal
print("Starting playback and recording...")
iostream = sd.Stream(
    samplerate=SAMPLE_RATE, channels=2, device=(19, 17), blocksize=BUFFER_SIZE
)
iostream.start()
with ChargingBar("Processing", max=signal_length // BUFFER_SIZE) as bar:
    for start in range(0, signal_length, BUFFER_SIZE):
        end = min(start + BUFFER_SIZE, signal_length)
        iostream.write(np.column_stack((signal[start:end], signal[start:end])))
        record[start:end] = iostream.read(end - start)[0][:, 0]
        bar.next()
iostream.stop()
iostream.close()
print("Playback and recording finished.")

# Roll the recorded signal to remove delay
delay = np.argmax(record > 0.01) - np.argmax(signal > 0.01)
record = np.roll(record, -delay)

# Plot the output and recorded signals
# ts = np.arange(signal_length) / SAMPLE_RATE
# plt.plot(ts, signal, label="Output Signal")
# plt.plot(ts, record, label="Recorded Signal")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.show()

# Plot peridiogram of the recorded signal
# fs, recPxx = periodogram(record, fs=SAMPLE_RATE, scaling="spectrum")
# _, signalPxx = periodogram(signal, fs=SAMPLE_RATE, scaling="spectrum")
# mask = (fs >= 20) & (fs <= 20000)
# plt.semilogx(fs[mask], 10 * np.log10(recPxx[mask].clip(1e-20)), label="Recorded Signal")
# plt.semilogx(
#     fs[mask], 10 * np.log10(signalPxx[mask].clip(1e-20)), label="Output Signal"
# )
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Amplitude")
# plt.title("Periodogram of Output and Recorded Signals")
# plt.grid(True, which="both", ls="--")
# plt.legend()
# plt.show()

# Calculate starts end ends of each step in the recorded signal
step_starts = np.cumsum(step_lengths) + PADDING
step_ends = step_starts + step_lengths

THDs = np.zeros_like(f_centers)
for i, (start, end, fc) in enumerate(zip(step_starts, step_ends, f_centers)):
    fs, Pxx = periodogram(
        record[start:end], fs=SAMPLE_RATE, scaling="spectrum", window="hann"
    )

    # Remove DC component
    Pxx[0] = 0

    # Find fundamental frequency bin
    fc_idx = np.argmin(np.abs(fs - fc))

    # Build band mask
    band_mask = (fs >= 20) & (fs <= 20000)

    # Build the fundamental mask
    fund_mask = (fs >= fc / 1.5) & (fs <= fc * 1.5)

    # Calculate sum of all other bins
    fund_Pxx = Pxx[fund_mask].sum()
    distortion_Pxx = np.sum(Pxx[band_mask]) - fund_Pxx

    THDs[i] = np.sqrt(distortion_Pxx / fund_Pxx)

    # if True:  # Plot spectra for debugging
    #     # Plot the spectrum for the 10th frequency step
    #     harm_mask = band_mask & ~fund_mask
    #     plt.semilogx(
    #         fs[fund_mask],
    #         np.sqrt(Pxx[fund_mask]),
    #         label="Fundamental",
    #         color="r",
    #     )
    #     plt.semilogx(
    #         fs[harm_mask],
    #         np.sqrt(Pxx[harm_mask]),
    #         label="Harmonics",
    #         color="b",
    #     )
    #     plt.xlabel("Frequency [Hz]")
    #     plt.ylabel("Amplitude")
    #     plt.title(f"Spectrum at {fc:.2f} Hz")
    #     plt.grid(True, which="both", ls="--")
    #     plt.show()

# Plot THD vs frequency
plt.semilogx(f_centers, THDs * 100)
plt.xlabel("Frequency [Hz]")
plt.ylabel("THD [%]")
plt.title("Total Harmonic Distortion vs Frequency")
plt.grid(True, which="both", ls="--")
plt.show()
