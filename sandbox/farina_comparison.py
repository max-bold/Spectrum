# Сравнение: Original vs Fixed Farina implementation
"""
Side-by-side comparison of:
1. Original (с fade на fundamental) - проблемное
2. Fixed (без fade на fundamental) - исправленное
"""

import sys

sys.path = [p for p in sys.path if "sandbox" not in p]
sys.path.insert(0, "C:\\Files\\Code\\Spectrum\\")

import numpy as np
from scipy.signal import fftconvolve, periodogram
import matplotlib.pyplot as plt
from utils.windows import grid_filter

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

T = 10.0  # Shorter for faster analysis
fs = 48000
f_start = 20
f_end = 20000
n_harmonics = 5
amplitude_ratio = 0.1
fade_length = int(1.0 * fs)

t = np.linspace(0, T, int(fs * T), endpoint=False)
K = T / np.log(f_end / f_start)
env = np.exp(t / K)

# ============================================================================
# ORIGINAL IMPLEMENTATION (PROBLEMATIC)
# ============================================================================

print("Generating Original (problematic) version...")

signal_orig = np.zeros_like(t)
response_orig = np.zeros_like(t)

for i in range(n_harmonics):
    h = i + 1
    sweep = np.sin(2 * np.pi * f_start * h * K * (np.exp(t / K) - 1))
    sweep[:fade_length] *= np.linspace(0, 1, fade_length)  # Fade ON fundamental
    sweep[-fade_length:] *= np.linspace(1, 0, fade_length)

    if i == 0:
        signal_orig += sweep
    response_orig += (amplitude_ratio**i) * sweep

# Simulate non-stationary behavior
time_window = (3 < t) & (t < 7)
multiplier = np.ones_like(t)
multiplier[time_window] = 0.1
fade_len = int(np.sum(time_window) / 10)
idx = np.where(time_window)[0]
if len(idx) > 2 * fade_len:
    multiplier[idx[:fade_len]] = np.linspace(1, 0.1, fade_len)
    multiplier[idx[-fade_len:]] = np.linspace(0.1, 1, fade_len)
response_orig *= multiplier

# Deconvolution
inv_filter_orig = signal_orig[::-1] / env
inv_filter_orig /= np.max(np.abs(inv_filter_orig))
response_orig_norm = response_orig / np.max(np.abs(response_orig))

ir_orig = fftconvolve(response_orig_norm, inv_filter_orig, mode="full")
ir_time_orig = np.arange(ir_orig.size) / fs

# ============================================================================
# FIXED IMPLEMENTATION (NO FADE ON FUNDAMENTAL)
# ============================================================================

print("Generating Fixed version...")

signal_fix = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
# NO FADE on fundamental!

response_fix = signal_fix.copy()

for i in range(1, n_harmonics):
    h = i + 1
    sweep = np.sin(2 * np.pi * f_start * h * K * (np.exp(t / K) - 1))
    sweep[:fade_length] *= np.linspace(0, 1, fade_length)  # Fade OK on harmonics
    sweep[-fade_length:] *= np.linspace(1, 0, fade_length)
    response_fix += (amplitude_ratio**i) * sweep

# Simulate non-stationary behavior (same as original)
time_window = (3 < t) & (t < 7)
multiplier = np.ones_like(t)
multiplier[time_window] = 0.1
fade_len = int(np.sum(time_window) / 10)
idx = np.where(time_window)[0]
if len(idx) > 2 * fade_len:
    multiplier[idx[:fade_len]] = np.linspace(1, 0.1, fade_len)
    multiplier[idx[-fade_len:]] = np.linspace(0.1, 1, fade_len)
response_fix *= multiplier

# Deconvolution
inv_filter_fix = signal_fix[::-1] / env
inv_filter_fix /= np.max(np.abs(inv_filter_fix))
response_fix_norm = response_fix / np.max(np.abs(response_fix))

ir_fix = fftconvolve(response_fix_norm, inv_filter_fix, mode="full")
ir_time_fix = np.arange(ir_fix.size) / fs

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("COMPARATIVE ANALYSIS")
print("=" * 70)

peak_orig = np.max(np.abs(ir_orig))
peak_fix = np.max(np.abs(ir_fix))

print(f"\nOriginal (with fade on fundamental):")
print(f"  Peak: {peak_orig:.4f}")
print(f"  Energy: {np.sum(ir_orig**2):.4e}")

print(f"\nFixed (no fade on fundamental):")
print(f"  Peak: {peak_fix:.4f}")
print(f"  Energy: {np.sum(ir_fix**2):.4e}")

print(f"\nImprovement:")
print(f"  Peak reduction: {(1 - peak_fix/peak_orig)*100:.1f}%")
print(f"  Energy ratio: {np.sum(ir_orig**2) / np.sum(ir_fix**2):.3f}x")

# ============================================================================
# PLOTTING
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Row 1: Input signals
ax1 = fig.add_subplot(gs[0, 0])
t_short = t[: min(8000, len(t))]
ax1.plot(t_short, signal_orig[: len(t_short)], linewidth=1.5)
ax1.set_title("Original: Signal (with fade)", fontsize=10, fontweight="bold")
ax1.set_ylabel("Amplitude")
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t_short, signal_fix[: len(t_short)], linewidth=1.5, color="orange")
ax2.set_title("Fixed: Signal (NO fade)", fontsize=10, fontweight="bold")
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(
    t_short,
    signal_orig[: len(t_short)],
    linewidth=1.5,
    label="Original (with fade)",
    alpha=0.7,
)
ax3.plot(
    t_short,
    signal_fix[: len(t_short)],
    linewidth=1.5,
    label="Fixed (no fade)",
    alpha=0.7,
    color="orange",
)
ax3.set_title("Signal Comparison", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Row 2: Impulse responses (full)
ax4 = fig.add_subplot(gs[1, 0])
t_ir = np.arange(min(100000, len(ir_orig))) / fs
ax4.plot(t_ir, ir_orig[: len(t_ir)], linewidth=0.8)
ax4.set_title(
    f"Original: IR (Peak={peak_orig:.0f})", fontsize=10, fontweight="bold", color="red"
)
ax4.set_ylabel("Amplitude")
ax4.set_xlabel("Time (s)")
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(t_ir, ir_fix[: len(t_ir)], linewidth=0.8, color="orange")
ax5.set_title(
    f"Fixed: IR (Peak={peak_fix:.0f})", fontsize=10, fontweight="bold", color="green"
)
ax5.set_xlabel("Time (s)")
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(
    t_ir,
    ir_orig[: len(t_ir)],
    linewidth=0.8,
    label=f"Original ({peak_orig:.0f})",
    alpha=0.7,
)
ax6.plot(
    t_ir,
    ir_fix[: len(t_ir)],
    linewidth=0.8,
    label=f"Fixed ({peak_fix:.0f})",
    alpha=0.7,
    color="orange",
)
ax6.set_title("IR Comparison", fontsize=10, fontweight="bold")
ax6.set_ylabel("Amplitude")
ax6.set_xlabel("Time (s)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# Row 3: Edge artifacts zoom
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(ir_orig[len(signal_orig) : len(signal_orig) + int(2 * fs)], linewidth=1)
ax7.set_title(
    "Original: Edge artifact (1-3s region)", fontsize=10, fontweight="bold", color="red"
)
ax7.set_ylabel("Amplitude")
ax7.set_xlabel("Sample")
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(
    ir_fix[len(signal_fix) : len(signal_fix) + int(2 * fs)], linewidth=1, color="orange"
)
ax8.set_title(
    "Fixed: Edge artifact (1-3s region)", fontsize=10, fontweight="bold", color="green"
)
ax8.set_ylabel("Amplitude")
ax8.set_xlabel("Sample")
ax8.grid(True, alpha=0.3)

ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
text_summary = f"""
IMPROVEMENT SUMMARY

Original (with fade on fundamental):
  Peak: {peak_orig:.0f}
  Energy: {np.sum(ir_orig**2):.2e}

Fixed (no fade on fundamental):
  Peak: {peak_fix:.0f}
  Energy: {np.sum(ir_fix**2):.2e}

Results:
  Peak reduced: {(peak_orig-peak_fix):.0f} ({(1-peak_fix/peak_orig)*100:.1f}%)
  Energy ratio: {np.sum(ir_orig**2)/np.sum(ir_fix**2):.2f}x

Conclusion:
Removing fade from fundamental
eliminates the main source of
deconvolution artifacts!
"""
ax9.text(
    0.05,
    0.95,
    text_summary,
    transform=ax9.transAxes,
    fontsize=9,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
)

plt.suptitle(
    "Farina Deconvolution: Original vs Fixed (No Fade on Fundamental)",
    fontsize=12,
    fontweight="bold",
    y=0.995,
)

plt.show()

print("\nComparison complete!")
