# Глубокий анализ: Почему энергия увеличивается на ^2?
"""
КЛЮЧЕВАЯ ПРОБЛЕМА ОБНАРУЖЕНА!

При Farina deconvolution с fade:
- Signal имеет fade in/out
- Response также имеет fade in/out
- Но при свертке они ВЗАИМОДЕЙСТВУЮТ

Это НЕ противоречит теории, это СЛЕДСТВИЕ теории!

Когда свертывали сигнал ЗА период измерения:
  response = (сигнал с fade) * channel

Когда деконволютивали:
  h = response * inverse_filter
  h = (signal_fade * channel) * (signal_fade[::-1] / envelope)

При несовершенной инверсии (из-за fade и граничных условий):
  → энергия УВЕЛИЧИВАЕТСЯ

РЕШЕНИЕ: Нужно обрезать результат деконволюции!
"""

import sys

sys.path = [p for p in sys.path if "sandbox" not in p]
sys.path.insert(0, "C:\\Files\\Code\\Spectrum\\")

import numpy as np
from scipy.signal import fftconvolve, periodogram, windows
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

T = 10.0  # Sweep duration
fs = 48000
f_start = 100
f_end = 10000
fade_length = int(1.0 * fs)

t = np.linspace(0, T, int(fs * T), endpoint=False)
K = T / np.log(f_end / f_start)

# ============================================================================
# CREATE TEST SIGNALS
# ============================================================================

# Pure sweep
sweep = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))

# With fade
signal_fade = sweep.copy()
signal_fade[:fade_length] *= np.linspace(0, 1, fade_length)
signal_fade[-fade_length:] *= np.linspace(1, 0, fade_length)

# Simulated response = signal (simple system with unity gain)
response_fade = signal_fade.copy()

# ============================================================================
# STANDARD DECONVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("STANDARD DECONVOLUTION ANALYSIS")
print("=" * 70)

env = np.exp(t / K)
inv_filter = signal_fade[::-1] / env
inv_filter /= np.max(np.abs(inv_filter))

# Full convolution includes boundary artifacts!
ir_full = fftconvolve(response_fade, inv_filter, mode="full")
ir_time_full = np.arange(ir_full.size) / fs

print(f"Signal length: {len(signal_fade)} samples ({T} s)")
print(f"Convolution result length: {len(ir_full)} samples ({len(ir_full)/fs:.3f} s)")
print(f"Expected valid region: ~{len(signal_fade)} samples")

# ============================================================================
# KEY ANALYSIS: Where is the peak?
# ============================================================================

print("\n" + "=" * 70)
print("PEAK ANALYSIS")
print("=" * 70)

peak_idx = np.argmax(np.abs(ir_full))
peak_time = ir_time_full[peak_idx]
peak_value = ir_full[peak_idx]

print(f"Peak location: index={peak_idx}, time={peak_time:.4f} s")
print(f"Peak value: {peak_value:.6f}")
print(f"Expected peak at: ~{T} s (or earlier due to fade)")

# Look at energy distribution
energy_by_region = {
    "First 1s (fade)": np.sum(ir_full[: int(1 * fs)] ** 2),
    "1s-2s": np.sum(ir_full[int(1 * fs) : int(2 * fs)] ** 2),
    "2s-5s": np.sum(ir_full[int(2 * fs) : int(5 * fs)] ** 2),
    "5s-10s": np.sum(ir_full[int(5 * fs) : int(10 * fs)] ** 2),
    "After 10s (fade+overflow)": np.sum(ir_full[int(10 * fs) :] ** 2),
}

print(f"\nEnergy distribution:")
total_energy = np.sum(ir_full**2)
for region, energy in energy_by_region.items():
    print(f"  {region:30s}: {energy:.2e} ({100*energy/total_energy:.1f}%)")

# ============================================================================
# SOLUTION: Windowing/truncation strategies
# ============================================================================

print("\n" + "=" * 70)
print("WINDOWING STRATEGIES")
print("=" * 70)

# Strategy 1: Simple truncation
ir_truncated = ir_full[: len(signal_fade)]
peak_trunc = np.max(np.abs(ir_truncated))
print(f"Strategy 1 - Truncate to signal length:")
print(f"  Length: {len(ir_truncated)} samples")
print(f"  Peak: {peak_trunc:.6f}")

# Strategy 2: Remove fade regions
fade_region = int(fade_length * 1.5)
ir_windowed = ir_full[fade_region : len(signal_fade) + fade_region]
peak_wind = np.max(np.abs(ir_windowed))
print(f"\nStrategy 2 - Remove fade regions ({fade_region} samples):")
print(f"  Length: {len(ir_windowed)} samples")
print(f"  Peak: {peak_wind:.6f}")

# Strategy 3: Use 'same' mode instead of 'full'
ir_same = fftconvolve(response_fade, inv_filter, mode="same")
peak_same = np.max(np.abs(ir_same))
print(f"\nStrategy 3 - Use 'same' mode:")
print(f"  Length: {len(ir_same)} samples")
print(f"  Peak: {peak_same:.6f}")

# Strategy 4: Use 'valid' mode
ir_valid = fftconvolve(response_fade, inv_filter, mode="valid")
peak_valid = np.max(np.abs(ir_valid)) if len(ir_valid) > 0 else 0
print(f"\nStrategy 4 - Use 'valid' mode:")
print(f"  Length: {len(ir_valid)} samples")
print(f"  Peak: {peak_valid:.6f}")

# Strategy 5: Apply window to signals BEFORE convolution
window_func = windows.hann(len(signal_fade))
response_windowed = response_fade * window_func
inv_filter_windowed = inv_filter * window_func[::-1]
inv_filter_windowed /= np.max(np.abs(inv_filter_windowed))

ir_pre_window = fftconvolve(response_windowed, inv_filter_windowed, mode="full")
peak_pre_window = np.max(np.abs(ir_pre_window))
print(f"\nStrategy 5 - Pre-multiply with Hann window:")
print(f"  Length: {len(ir_pre_window)} samples")
print(f"  Peak: {peak_pre_window:.6f}")

# ============================================================================
# THE REAL PROBLEM: Boundary Effects
# ============================================================================

print("\n" + "=" * 70)
print("THE REAL PROBLEM: Periodicity Assumptions")
print("=" * 70)

print(
    """
When using fftconvolve(mode='full'):
1. fftconvolve assumes CIRCULAR convolution with zero-padding
2. This means the signal is implicitly extended with zeros
3. The fade regions interact with this zero-padding

Result: At the boundaries where signal meets padding, we get
DISCONTINUITIES that appear as impulses in the deconvolved result!

Mathematical explanation:
- Signal ends with: fade_out -> 0 (discontinuity!)
- Response ends with: fade_out -> 0 (discontinuity!)
- Inverse filter has: signal[::-1] / envelope
  
At boundaries:
- signal[::-1] ends with zeros (from fade start)
- envelope is HUGE at those points
- So inverse filter ~= 0
- But the discontinuity 0->something creates a spike during convolution!

CONCLUSION: This is NOT a bug, it's a fundamental issue with:
a) Using finite signals with deconvolution
b) Having discontinuities (fade) at the edges
c) The non-stationary nature of the sweep
"""
)

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Plot 1: Full IR
ax = axes[0, 0]
t_plot = np.linspace(0, len(ir_full) / fs, len(ir_full))
ax.plot(t_plot, ir_full)
ax.axvline(T, color="r", linestyle="--", alpha=0.5, label=f"T={T}s")
ax.axvline(
    fade_length / fs,
    color="g",
    linestyle="--",
    alpha=0.5,
    label=f"Fade length={fade_length/fs}s",
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Full IR (mode='full')")
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: IR - first part
ax = axes[0, 1]
ax.plot(ir_full[: int(5 * fs)])
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("IR - First 5 seconds")
ax.grid(True, alpha=0.3)

# Plot 3: IR - last part (overflow)
ax = axes[0, 2]
ax.plot(ir_full[int(10 * fs) :])
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("IR - After 10s (boundary overflow)")
ax.grid(True, alpha=0.3)

# Plot 4: Truncated IR
ax = axes[1, 0]
ax.plot(ir_truncated)
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title(f"Truncated IR (strategy 1)")
ax.grid(True, alpha=0.3)

# Plot 5: Windowed IR
ax = axes[1, 1]
ax.plot(ir_windowed)
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title(f"Windowed IR (strategy 2)")
ax.grid(True, alpha=0.3)

# Plot 6: Comparison
ax = axes[1, 2]
strategies = [
    ("Full\n(current)", peak_idx / fs, np.max(np.abs(ir_full))),
    ("Truncated", 0, peak_trunc),
    ("Windowed", 0, peak_wind),
    ("'same'", 0, peak_same),
]
names = [s[0] for s in strategies]
peaks = [s[2] for s in strategies]
colors = ["red" if i == 0 else "blue" for i in range(len(strategies))]
bars = ax.bar(range(len(strategies)), peaks, color=colors, alpha=0.7)
ax.set_ylabel("Peak Value")
ax.set_title("Peak Comparison: Different Strategies")
ax.set_xticks(range(len(strategies)))
ax.set_xticklabels(names)
for i, (bar, peak) in enumerate(zip(bars, peaks)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        peak,
        f"{peak:.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)
print(
    """
BEST PRACTICE for Farina with fade:
1. Apply fade to signal (for smooth edges in measurement)
2. After deconvolution, TRUNCATE to meaningful region
3. Remove ~1-2 seconds from both ends to avoid edge artifacts
4. Use the middle portion for analysis (response time, THD, etc.)

ALTERNATIVELY:
- Don't apply fade to the fundamental sweep
- Apply fade only to the harmonic content
- This reduces boundary artifacts while keeping measurement smooth
"""
)
