import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- Параметры свипа ----
rate: int = 96000
chunksize: int = 4096
band = np.asarray((100, 10000), np.float64) / rate
t_end = 3
length = int(t_end * rate)
nk = np.sqrt(length * rate / np.log10(band[1] / band[0]) / 10) * 10 ** (0.35726 / 20)
# print("speed:", speed)

# ---- Генерация свипа ----
k = length * band[0] / np.log(band[1] / band[0])
l = length / np.log(band[1] / band[0])
t = np.arange(0, length, dtype=np.float64)
sweep = np.sin(2 * np.pi * k * (np.exp(t / l) - 1))

# ---- Фигура/оси ----
fig, ax = plt.subplots(figsize=(7, 4))
ax: Axes = ax
ax.grid(True, which="both")
ax.set_xscale("log")
(line1,) = ax.plot([], [], lw=1, label="2/len(chunk) norm")  # будет как ax.semilogx
(line2,) = ax.plot([], [], lw=1, label="1/nk norm")
line1: Line2D = line1
line2: Line2D = line2
ax.set_xlim(20, 20e3)
ax.set_ylim(-60, 20)
ax.set_xlabel("Frequency, Hz")
ax.set_ylabel("Amplitude, dB")
ax.set_title("FFT of growing sweep (two normalizations)")
ax.legend()

# ---- Кадры для анимации (индексы конца окна) ----
frame_indices = list(range(chunksize, length + 1, chunksize))
if frame_indices[-1] != length:
    frame_indices.append(length)


def update(end_idx: int):
    """Обновление данных для кадра: берём sweep[:end_idx] и пересчитываем FFT."""
    chunk = sweep[:end_idx]

    xf = np.fft.rfftfreq(len(chunk), 1 / rate)
    yf = np.abs(np.fft.rfft(chunk))

    # Вариант 1 — 2 * |FFT| / N (с pink-тилингом sqrt(f))
    yf1 = 2 * yf / len(chunk)
    pinked_yf1 = yf1 * np.sqrt(np.clip(xf, 1e-12, None))
    log_yf1 = 20 * np.log10(np.clip(pinked_yf1, 1e-12, None))
    line1.set_data(xf, log_yf1)

    # Вариант 2 — твоя "стабильная" нормировка (масштаб по speed) + pink
    yf2 = yf / nk
    pinked_yf2 = yf2 * np.sqrt(np.clip(xf, 1e-12, None))
    log_yf2 = 20 * np.log10(np.clip(pinked_yf2, 1e-12, None))
    line2.set_data(xf, log_yf2)

    # Возвращаем артисты, чтобы FuncAnimation знал, что перерисовывать
    return line1, line2


# ---- Создание анимации ----
ani = FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=100,  # мс между кадрами (визуальная скорость)
    blit=False,  # у нас меняется длина массивов — blit лучше выключить
)

# ---- Сохранение в GIF ----
# Требуется Pillow: pip install pillow
writer = PillowWriter(fps=10)  # кадры в секунду в результирующем GIF
ani.save(r"sandbox\sweep_fft.gif", writer=writer, dpi=120)

plt.close(fig)  # чтобы окно не всплывало при сохранении из скрипта
print("Готово: sweep_fft.gif")
