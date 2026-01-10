# Lets test ferina inverse filter method

import numpy as np
from scipy.signal import fftconvolve, periodogram
import matplotlib.pyplot as plt

# Import log_filter2 from utils.windows that is not in serarch path
import sys

sys.path.append("C:\\Files\\Code\\Spectrum\\")
from utils.windows import log_filter2, grid_filter

# Length of the sweep
T = 30.0  # seconds

# Sampling frequency
fs = 96000  # Hz

# Number of harmonics
num_harmonics = 10

# Time vector
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Initialize signal and response
signal = np.zeros_like(t)
response = np.zeros_like(t)

# bandwidth for the sweep
f_start = 20  # Hz
f_end = 40000  # Hz

# Generate signal and response
for i in range(num_harmonics):
    # Generate logarithmic sweep
    f_start_i = f_start * (2**i)
    f_end_i = f_end * (2**i)
    Ki = T / np.log(f_end_i / f_start_i)
    sweep = np.sin(2 * np.pi * f_start_i * Ki * (np.exp(t / Ki) - 1))
    # # Add in and out fading to each sweep to reduce clicks
    fade_length = int(1.0 * fs)
    sweep[:fade_length] *= np.linspace(0, 1, fade_length)
    sweep[-fade_length:] *= np.linspace(1, 0, fade_length)

    if i == 0:
        signal = sweep

    response += sweep * (0.1**i)  # Decrease amplitude for higher harmonics


# Add some noise to the response
# response += 0.001 * np.random.normal(size=response.shape)

# Attenuate response to avoid clipping
# response/=10

# Simulate response nonlinearity
mask = (10 < t) & (t < 20)
multiplier = np.ones(sum(mask), dtype=float) * 0.1
fade_length = int(0.5 * fs)
multiplier[:fade_length] = np.linspace(1, 0.1, fade_length)
multiplier[-fade_length:] = np.linspace(0.1, 1, fade_length)
response[mask] *= multiplier

# Compute Farina inverse filter
K = T / np.log(f_end / f_start)
env = np.exp(t / K)
inv_filter = signal[::-1] / env

# Normilise inverse filter and response
# inv_filter /= np.max(np.abs(inv_filter))
# response /= np.max(np.abs(response))

# Convolve sweep with its inverse filter to get impulse response
print("Computing impulse response...")
impulse_response = fftconvolve(response, inv_filter, mode="full")
ir_time = np.arange(impulse_response.size) / fs

# Normalize impulse response
# impulse_response /= np.max(np.abs(impulse_response))

# Make a plot with 4 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes
ax4: plt.Axes
# Plot the impulse response
ax1.plot(ir_time, impulse_response)

# Calculate predicted time for each harmonic IR and add vertical lines to the plot
for i in range(num_harmonics):
    n_ir_time = T - K * i * np.log(2)
    ax1.axvline(
        n_ir_time,
        color="r",
        linestyle="--",
        label=f"Harmonic {i+1} Time" if i == 0 else "",
    )


ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.grid()

# Plot harmonic spectra
log_f = np.geomspace(20, 20000, 1024)


# Calculate signal spectra
f, signal_Pxx = periodogram(signal, fs=fs)
signal_Pxx *= f  # Weight by frequency to match Farina method
log_signal_Pxx = grid_filter(f, signal_Pxx, log_f, w=1 / 10)
log_signal_Pxx /= np.max(log_signal_Pxx)

w = K * np.log(2) / 2  # Half window width in seconds
h0 = np.zeros_like(log_f)
thd = np.zeros_like(log_f)
for i in range(num_harmonics):
    n_ir_time = T - K * i * np.log(2)
    st = n_ir_time - w
    et = n_ir_time + w
    mask = (ir_time >= st) & (ir_time < et)
    f, Pxx = periodogram(impulse_response[mask], fs=fs, window="hann")
    # Zero Pxx for f<f_start*(2**i)
    # Pxx[f < f_start * (2**i)] = 1e-20
    log_Pxx = grid_filter(f / (2**i), Pxx, log_f, w=1 / 10)
    # log_Pxx[log_f < f_start * (2**i)] = 1e-20
    # log_Pxx /= log_signal_Pxx  # Normalize by signal spectrum
    if i == 0:
        log_Pxx /= log_signal_Pxx
        h0 = log_Pxx
    else:
        # log_Pxx /= log_signal_Pxx
        thd += log_Pxx
    ax2.semilogx(log_f, 10 * np.log10(log_Pxx.clip(1e-20)), label=f"{i+1}x Harmonic")

ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Power/Frequency [dB/Hz]")
ax2.grid(True, which="both", ls="--")
ax2.set_title("Harmonic Spectra")

# Calculate and plot THD+n
thd = np.sqrt(thd / h0)
#Apply smoothing to THD+n
window_size = 5
thd = np.convolve(thd, np.ones(window_size) / window_size, mode="same")

ax3.semilogx(log_f, thd * 100)
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("THD+n [%]")
ax3.grid(True, which="both", ls="--")
ax3.set_title("THD+n")


plt.show()
