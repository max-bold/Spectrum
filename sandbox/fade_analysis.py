# Анализ влияния fade in/out на h1 при деконволюции
"""
Проблема: fade in/out увеличивают артефакты в h1 при деконволюции.

Теория:
- Signal: [FADE_IN | SWEEP | FADE_OUT]
- Response: [FADE_IN | SWEEP | FADE_OUT]  (+ harmonic content)
- Inverse filter: signal[::-1] / envelope

При свертке: response * inv_filter_reversed

Проблема 1: ЭНЕРГИЯ УВЕЛИЧИВАЕТСЯ
=============================
Fade in/out создает РАЗРЫВЫ в сигнале. При деконволюции эти разрывы
преобразуются в импульсы.

Например:
- Signal:    [0.0, 0.1, 0.2, ..., 1.0 (middle) ..., 0.2, 0.1, 0.0]
- Inv filter должен быть: [0.0 ... reversed ... 0.0]

Но при делении на envelope: [нечеткие границы в обе стороны]

Свертка КВАДРАТНОГО РАЗРЕЗА (~^2) - это особенность деконволюции!

Проблема 2: ФАЗОВЫЙ СДВИГ
==========================
Fade in влияет на ФАЗУ сигнала в начале.
Fade out влияет на ФАЗУ сигнала в конце.
При реверсировании для inverse filter эти фазовые сдвиги
создают ИНТЕРФЕРИРУЮЩИЕ компоненты.

Проблема 3: ВРЕМЕННОЕ СМЕЩЕНИЕ
================================
Inverse filter = signal[::-1] / exp(t/K)
Но envelope = exp(t/K) монотонно растет!
При реверсировании она становится монотонно ПАДАЮЩЕЙ в обратном времени.
Это означает, что начало сигнала (с малой амплитудой из fade)
получает ОГРОМНОЕ усиление (деление на маленькие числа).
"""

import sys

sys.path = [p for p in sys.path if "sandbox" not in p]
sys.path.insert(0, "C:\\Files\\Code\\Spectrum\\")

import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

T = 10.0  # Sweep duration (seconds)
fs = 48000  # Sampling frequency (Hz)
f_start = 100  # Start frequency (Hz)
f_end = 10000  # End frequency (Hz)
fade_length = int(1.0 * fs)  # 1 second fade in/out

# ============================================================================
# TEST 1: Простой sweep БЕЗ fade
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1: Sweep БЕЗ fade в/о")
print("=" * 70)

t = np.linspace(0, T, int(fs * T), endpoint=False)
K = T / np.log(f_end / f_start)

# Pure sweep без fade
signal_no_fade = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))

# Response = signal (простой случай)
response_no_fade = signal_no_fade.copy()

# Deconvolution
env = np.exp(t / K)
inv_filter_no_fade = signal_no_fade[::-1] / env
inv_filter_no_fade /= np.max(np.abs(inv_filter_no_fade))

ir_no_fade = fftconvolve(response_no_fade, inv_filter_no_fade, mode="full")
ir_time_no_fade = np.arange(ir_no_fade.size) / fs

peak_no_fade = np.max(np.abs(ir_no_fade))
print(f"Peak в h1 (без fade): {peak_no_fade:.6f}")

# ============================================================================
# TEST 2: Sweep С fade
# ============================================================================

print("\n" + "=" * 70)
print("TEST 2: Sweep С fade в/о")
print("=" * 70)

# Sweep с fade
signal_with_fade = signal_no_fade.copy()
signal_with_fade[:fade_length] *= np.linspace(0, 1, fade_length)
signal_with_fade[-fade_length:] *= np.linspace(1, 0, fade_length)

# Response = signal (простой случай)
response_with_fade = signal_with_fade.copy()

# Deconvolution
inv_filter_with_fade = signal_with_fade[::-1] / env
inv_filter_with_fade /= np.max(np.abs(inv_filter_with_fade))

ir_with_fade = fftconvolve(response_with_fade, inv_filter_with_fade, mode="full")
ir_time_with_fade = np.arange(ir_with_fade.size) / fs

peak_with_fade = np.max(np.abs(ir_with_fade))
print(f"Peak в h1 (с fade): {peak_with_fade:.6f}")
print(f"Увеличение: {peak_with_fade / peak_no_fade:.3f}x (~^2 = {2**2})")

# ============================================================================
# TEST 3: Анализ inverse filter
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3: Анализ inverse filter")
print("=" * 70)

# Посмотрим на энергию
energy_inv_no_fade = np.sum(inv_filter_no_fade**2)
energy_inv_with_fade = np.sum(inv_filter_with_fade**2)

print(f"Энергия inv_filter (без fade): {energy_inv_no_fade:.6e}")
print(f"Энергия inv_filter (с fade): {energy_inv_with_fade:.6e}")
print(f"Увеличение энергии: {energy_inv_with_fade / energy_inv_no_fade:.3f}x")

# Посмотрим на пики
peak_inv_no_fade = np.max(np.abs(inv_filter_no_fade))
peak_inv_with_fade = np.max(np.abs(inv_filter_with_fade))

print(f"\nPeak inv_filter (без fade): {peak_inv_no_fade:.6f}")
print(f"Peak inv_filter (с fade): {peak_inv_with_fade:.6f}")
print(f"Увеличение пика: {peak_inv_with_fade / peak_inv_no_fade:.3f}x")

# ============================================================================
# TEST 4: Анализ граничных условий (THE KEY!)
# ============================================================================

print("\n" + "=" * 70)
print("TEST 4: Анализ граничных условий - ГДЕ ПРОБЛЕМА!")
print("=" * 70)

# Посмотрим что происходит в начале/конце signal
print(f"\nSignal без fade:")
print(f"  Начало: {signal_no_fade[:5]}")
print(f"  Конец: {signal_no_fade[-5:]}")

print(f"\nSignal с fade:")
print(f"  Начало (fade in): {signal_with_fade[:5]}")
print(f"  Конец (fade out): {signal_with_fade[-5:]}")

# Проблема в envelope!
print(f"\nEnvelope (exp(t/K)):")
print(f"  Начало: {env[:5]}")
print(f"  Конец: {env[-5:]}")

# Inverse filter = signal[::-1] / envelope
print(f"\nInverse filter (без fade) - в начале (=сигнал[::-1][-5:]/env[:5]):")
inv_no_fade_start = (signal_no_fade[::-1][-5:]) / env[:5]
print(f"  {inv_no_fade_start}")

print(f"\nInverse filter (с fade) - в начале (=сигнал[::-1][-5:]/env[:5]):")
inv_with_fade_start = (signal_with_fade[::-1][-5:]) / env[:5]
print(f"  {inv_with_fade_start}")

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Plot 1: Сигналы
ax = axes[0, 0]
t_plot = np.linspace(0, T, min(5000, len(t)))
ax.plot(t_plot, signal_no_fade[: len(t_plot)], label="Signal (no fade)", alpha=0.7)
ax.plot(t_plot, signal_with_fade[: len(t_plot)], label="Signal (with fade)", alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Signals: с fade и без fade")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Envelope
ax = axes[0, 1]
ax.plot(t_plot, env[: len(t_plot)], label="Envelope exp(t/K)", linewidth=2)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Envelope")
ax.set_title("Envelope (монотонно растет)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Inverse filter
ax = axes[1, 0]
t_inv_plot = np.linspace(0, T, min(5000, len(inv_filter_no_fade)))
idx_plot = np.arange(0, min(5000, len(inv_filter_no_fade)))
ax.plot(
    t_inv_plot, inv_filter_no_fade[idx_plot], label="Inv filter (no fade)", alpha=0.7
)
ax.plot(
    t_inv_plot,
    inv_filter_with_fade[idx_plot],
    label="Inv filter (with fade)",
    alpha=0.7,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title("Inverse Filter (signal[::-1] / envelope)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Inverse filter - начало (PROBLEM AREA!)
ax = axes[1, 1]
idx_start = slice(0, 2000)
ax.plot(
    inv_filter_no_fade[idx_start],
    label="Inv filter (no fade)",
    alpha=0.7,
    linewidth=1.5,
)
ax.plot(
    inv_filter_with_fade[idx_start],
    label="Inv filter (with fade)",
    alpha=0.7,
    linewidth=1.5,
)
ax.set_xlabel("Sample")
ax.set_ylabel("Amplitude")
ax.set_title("Inverse Filter - НАЧАЛО (где большие пики!)")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Impulse Response
ax = axes[2, 0]
ir_time_plot_no_fade = np.arange(min(10000, len(ir_no_fade))) / fs
ir_time_plot_with_fade = np.arange(min(10000, len(ir_with_fade))) / fs
ax.plot(
    ir_time_plot_no_fade,
    ir_no_fade[: min(10000, len(ir_no_fade))],
    label="h1 (no fade)",
    alpha=0.7,
)
ax.plot(
    ir_time_plot_with_fade,
    ir_with_fade[: min(10000, len(ir_with_fade))],
    label="h1 (with fade)",
    alpha=0.7,
)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(
    f"Impulse Response (Peak: no_fade={peak_no_fade:.4f}, with_fade={peak_with_fade:.4f})"
)
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Summary
ax = axes[2, 1]
ax.axis("off")
summary_text = f"""
ВЫВОДЫ:

1. ПРОБЛЕМА: fade in/out увеличивают h1 на ~^2

2. ПРИЧИНА:
   - Fade создает РАЗРЫВЫ в сигнале (0 → плавное rise → значение)
   - При делении на envelope мы получаем УСИЛЕНИЕ
   - Envelope в начале ОЧЕНЬ МАЛЕНЬКАЯ (e^0 ≈ 1)
   - Но в конце ОЧЕНЬ БОЛЬШАЯ
   
3. ЭФФЕКТ:
   - Начало сигнала (с fade) / маленькая envelope = БОЛЬШОЕ значение
   - Конец сигнала (с fade) / большая envelope = маленькое значение
   
4. ПРИ СВЕРТКЕ:
   - Разрывы в signal и response создают КОНВОЛЮЦИЮ разрывов
   - Результат: энергия увеличивается примерно на ^2

5. РЕШЕНИЕ:
   - Применить fade ПОСЛЕ деконволюции? НЕТ
   - Компенсировать fade на обоих концах? МОЖЕТ
   - Не применять fade к signal/response? ЛУЧШЕ
"""
ax.text(
    0.05,
    0.95,
    summary_text,
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 70)
