import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy.typing import NDArray

fs = 44100  # частота дискретизации

# N_fir = 11  # длина FIR фильтра
# f_min = 20  # минимальная частота для аппроксимации 1/f
# f_max = fs / 2  # максимальная частота

# 1️⃣ Массив частот для FIR
# freqs = np.linspace(0, f_max, N_fir)

# # 2️⃣ Амплитудная характеристика 1/f
# # избегаем деления на ноль
# amps = np.ones_like(freqs)
# amps[freqs >= f_min] = 1 / freqs[freqs >= f_min]
# amps[-1] = 0
# amps /= np.max(amps)  # нормируем в 1

# # 3️⃣ Проектируем FIR фильтр
# fir_coeff = signal.firwin2(N_fir, freqs, amps, fs=fs)

# # 4️⃣ Конвертируем в SOS через tf2sos
# # сначала b,a для FIR
# b = fir_coeff
# a = [1.0]

PINKING_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINKING_A = [1, -2.494956002, 2.017265875, -0.522189400]

sos_pinking = np.array(signal.tf2sos(PINKING_B, PINKING_A), np.float64)
print(sos_pinking)
# 5️⃣ Проектируем 4-го порядка полосовой фильтр
lowcut = 100.0
highcut = 2e3
sos_band = np.array(
    signal.butter(4, (lowcut, highcut), btype="band", fs=fs, output="sos"), np.float64
)

# 6️⃣ Объединяем SOS фильтры
combined_sos = np.vstack([sos_pinking, sos_band])

# Проверка: фильтруем белый шум
white = np.random.uniform(-1, 1, fs * 30)  # 30 секунд
pink_band = signal.sosfilt(combined_sos, white)

# Спектр
f_pink, Pxx_pink = signal.welch(pink_band, fs=fs, nperseg=fs/2)
Pxx_pink *= f_pink*3000
plt.figure(figsize=(8, 4))
plt.semilogx(f_pink, 10 * np.log10(Pxx_pink.clip(1e-20)))
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.title("Combined Pink + Bandpass Filter")
plt.grid(True, which="both", ls="--")
plt.xlim(20, fs / 2)
plt.ylim(-60,10)
plt.show()
