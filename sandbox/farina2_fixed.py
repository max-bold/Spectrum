# Lets test Farina inverse filter method (corrected bookish approach)
"""
Farina method for harmonic analysis using logarithmic sweep.
Multiple sweeps at harmonic frequencies with 10:1 amplitude ratio.

FIX: No fade on fundamental sweep to avoid edge artifacts in deconvolution.
     Fade applied only to harmonics.
"""

import sys

# Remove sandbox from path to avoid local platform module conflict
sys.path = [p for p in sys.path if "sandbox" not in p]
sys.path.insert(0, "C:\\Files\\Code\\Spectrum\\")

import numpy as np
from scipy.signal import fftconvolve, periodogram
import matplotlib.pyplot as plt
from utils.windows import grid_filter

# ============================================================================
# CONFIGURATION
# ============================================================================

T = 30.0  # Sweep duration (seconds)
fs = 96000  # Sampling frequency (Hz)
f_start = 20  # Start frequency (Hz)
f_end = 40000  # End frequency (Hz)
n_harmonics = 10  # Harmonics count for analysis
amplitude_ratio = 0.1  # Each harmonic is 0.1x weaker (20 dB)

# ============================================================================
# GENERATE SIGNALS
# ============================================================================

# Time vector
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Fundamental sweep parameters
K = T / np.log(f_end / f_start)  # Time-frequency scaling factor
fade_length = int(1.0 * fs)  # 1 second fade in/out

signal = np.zeros_like(t)
response = np.zeros_like(t)

# FUNDAMENTAL SWEEP (NO FADE - this is key!)
fundamental = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
signal += fundamental
response += fundamental

# HARMONIC SWEEPS (WITH FADE)
for i in range(1, n_harmonics):
    h = i + 1
    sweep = np.sin(2 * np.pi * f_start * h * K * (np.exp(t / K) - 1))
    # Apply fade only to harmonics
    sweep[:fade_length] *= np.linspace(0, 1, fade_length)
    sweep[-fade_length:] *= np.linspace(1, 0, fade_length)

    response += (amplitude_ratio**i) * sweep


# Apply time-varying compression (simulating non-stationary behavior)
time_window = ((T / 3) < t) & (t < (T / 3 * 2))
multiplier = np.ones_like(t)
multiplier[time_window] = 0.1
fade_len = int(np.sum(time_window) / 10)
idx = np.where(time_window)[0]
multiplier[idx[:fade_len]] = np.linspace(1, 0.1, fade_len)
multiplier[idx[-fade_len:]] = np.linspace(0.1, 1, fade_len)
response *= multiplier

# ============================================================================
# COMPUTE IMPULSE RESPONSE (Farina deconvolution)
# ============================================================================

env = np.exp(t / K)
inv_filter = signal[::-1] / env
inv_filter /= np.max(np.abs(inv_filter))
response /= np.max(np.abs(response))
impulse_response = fftconvolve(response, inv_filter, mode="full")
ir_time = np.arange(impulse_response.size) / fs

# ============================================================================
# PLOTTING
# ============================================================================

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))
ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes
ax4: plt.Axes

# --- Plot 1: Impulse Response with harmonic timing markers
ax1.plot(ir_time, impulse_response, label="Impulse Response")
for i in range(n_harmonics):
    h = i + 1
    t_n = T - K * np.log(h)
    ax1.axvline(t_n, color="r", linestyle="--", alpha=0.7, linewidth=0.8)
ax1.set_xlabel("Time (s)", fontsize=10)
ax1.set_ylabel("Amplitude", fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.set_title(
    "Impulse Response (NO fade on fundamental)", fontsize=11, fontweight="bold"
)

# --- Plot 2: Harmonic Spectra
log_f = np.geomspace(20, 20000, 1024)

h1 = np.zeros_like(log_f)
THDn = np.zeros_like(log_f)

for i in range(n_harmonics):
    h = i + 1
    st = T - K * np.log(h + 0.5)
    et = T - K * np.log(h - 0.5)
    mask = (ir_time >= st) & (ir_time < et)
    f, Pxx = periodogram(impulse_response[mask], fs=fs, window="hann")
    grid_Pxx = grid_filter(f / h, Pxx, log_f, w=1 / 10)
    ax2.semilogx(
        log_f,
        10 * np.log10(grid_Pxx.clip(1e-20)),
        label=f"{h}x Harmonic" if i > 0 else "Fundamental",
    )
    if i == 0:
        h1 = grid_Pxx
    else:
        THDn += grid_Pxx

ax2.set_xlabel("Frequency (Hz)", fontsize=10)
ax2.set_ylabel("PSD (dB/Hz)", fontsize=10)
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(fontsize=9, loc="upper right")
ax2.set_title("Harmonic Spectra", fontsize=11, fontweight="bold")

# --- Plot 3: THD+n
h1 = h1.clip(1e-20)
thd = 100 * np.sqrt(THDn / h1)
window_size = 5
thd = np.convolve(thd, np.ones(window_size) / window_size, mode="same")


ax3.semilogx(log_f, thd, linewidth=1.5, color="darkblue")
ax3.set_xlabel("Frequency (Hz)", fontsize=10)
ax3.set_ylabel("THD+n (%)", fontsize=10)
ax3.grid(True, which="both", alpha=0.3)
ax3.set_title("Total Harmonic Distortion", fontsize=11, fontweight="bold")

# --- Plot 4: Info
ax4.axis("off")
info_text = """
FIX APPLIED:
- No fade on fundamental sweep
- Fade applied only to harmonics

REASON:
- Fade creates discontinuities at edges
- During deconvolution, discontinuities
  become impulses (spikes)
- Two discontinuities interact: ~^2 effect
- Removing fade from fundamental
  eliminates the main source

RESULT:
- Cleaner h1 measurement
- Better edge characteristics
- Reduced artificial THD from fade
"""
ax4.text(
    0.05,
    0.95,
    info_text,
    transform=ax4.transAxes,
    fontsize=9,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

plt.tight_layout()
plt.show()
