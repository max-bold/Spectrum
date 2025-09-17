import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Частота дискретизации
fs = 96000  # Гц

# Коэффициенты фильтра
PINKING_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINKING_A = [1, -2.494956002, 2.017265875, -0.522189400]

# Преобразуем в SOS
sos = signal.tf2sos(PINKING_B, PINKING_A)
print(sos)

# Частотный отклик в Гц
w1, h1 = signal.freqz(PINKING_B, PINKING_A, worN=1024, fs=fs)
w2, h2 = signal.sosfreqz(sos, worN=1024, fs=fs)

# Построим АЧХ в дБ
plt.figure(figsize=(8,4))
plt.semilogx(w1, 20*np.log10(abs(h1)), label='Original (b, a)')
plt.semilogx(w2, 20*np.log10(abs(h2)), '--', label='SOS')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude [dB]')
plt.title('Comparison of Original and SOS Filter')
plt.legend()
plt.grid(True, which='both', ls='--')
plt.show()
