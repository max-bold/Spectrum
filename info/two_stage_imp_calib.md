Нужно изменить proof-of-concept модуль измерения импеданса через звуковую карту.

Текущая схема измерения:

audio_out ── Rr ── p1 ── Zl ── gnd

CH1 измеряет напряжение в точке audio_out относительно gnd.
CH2 измеряет напряжение в точке p1 относительно gnd.
Rr — последовательный reference resistor.
Zl — измеряемая нагрузка.

Импеданс считается по формуле:

H = V2 / V1

Z = Rr * H / (1 - H)

Где V1 и V2 — комплексные FFT-спектры CH1 и CH2.

Нужно доработать калибровку. Вместо одностадийной калибровки по известному резистору сделать двухэтапную калибровку:

1. Калибровка относительного усиления и фазы каналов CH1/CH2.
2. Оценка фактического значения Rr по известному резистору Rcal.

Важно:
- Использовать комплексный FFT.
- Не использовать abs(FFT) до расчёта Z.
- Не использовать scipy.signal.welch.
- Не использовать scipy.signal.csd.
- Не использовать scipy.signal.periodogram.
- Не использовать отношение спектров мощности.
- Все отношения считать по комплексным данным.
- Сглаживание делать через utils.windows.log_filter2.
- utils.windows.log_filter2 должна поддерживать complex dtype.
- Не сохранять калибровку.
- Не загружать калибровку.
- Калибровку выполнять непосредственно перед измерением.
- Если есть clipping, выдавать warning или ValueError.
- Если сигнал достиг abs(signal) >= 1.0, при raise_on_clipping=True выдавать ValueError.

1. Доработать utils.windows.log_filter2

Сейчас функция может терять imaginary part, если выходной массив создаётся через zeros_like(log_f).

Нужно сделать поддержку complex dtype.

Если вход real:
- поведение должно остаться прежним.

Если вход complex:
- сглаживать real и imag отдельно;
- возвращать complex массив;
- не сглаживать abs(x);
- не терять imaginary part.

Желательная структура:

def _log_filter2_real(...):
    # существующая real-valued реализация

def log_filter2(...):
    if np.iscomplexobj(Pxx):
        log_f, real_filtered = _log_filter2_real(f, np.real(Pxx), ...)
        _, imag_filtered = _log_filter2_real(f, np.imag(Pxx), ...)
        return log_f, real_filtered + 1j * imag_filtered

    return _log_filter2_real(f, Pxx, ...)

Также поправить dtype выходного массива в real-реализации:

out_dtype = np.result_type(Pxx.dtype, np.float64)
log_Pxx = np.full(log_f.shape, np.nan, dtype=out_dtype)

И использовать:

df = f[1] - f[0]

а не:

df = f[1]

Добавить простую проверку/тест:
- complex input возвращает complex output;
- imaginary part не обнуляется;
- результат совпадает с log_filter2(real(x)) + 1j * log_filter2(imag(x)).

2. Калибровка каналов

Добавить функцию:

calculate_channel_correction(
    ch1_same: np.ndarray,
    ch2_same: np.ndarray,
    fs: float,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]

Назначение:
Оценить комплексную поправку относительного усиления и фазы CH2 относительно CH1.

Схема подключения для этой калибровки:
- CH1 и CH2 подключены к одной и той же точке audio_out относительно gnd.
- Физически оба входа должны видеть одинаковый сигнал.

Алгоритм:
1. Посчитать комплексные FFT-спектры:

   freq, V1, V2 = calculate_fft_spectra(ch1_same, ch2_same, fs)

2. Если smoothing=True:
   - сгладить V1 через utils.windows.log_filter2;
   - сгладить V2 через utils.windows.log_filter2.

3. Посчитать:

   Kch = V2 / V1

4. Применить eps-защиту:
   - если abs(V1) < eps, Kch в этой точке = np.nan + 1j*np.nan.

5. Ограничить диапазон f_min/f_max.

6. Вернуть:

   freq, Kch

Дальше во всех измерениях CH2 корректируется так:

V2_corrected = V2_measured / Kch

3. Оценка фактического значения Rr

Добавить функцию:

estimate_reference_resistor(
    ch1_cal: np.ndarray,
    ch2_cal: np.ndarray,
    fs: float,
    channel_correction: np.ndarray,
    calibration_resistor: float,
    reference_resistor_nominal: float | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, dict]

Назначение:
Оценить фактическое значение последовательного резистора Rr в текущем состоянии, с учётом температуры и реального сопротивления.

Схема подключения:

audio_out ── Rr ── p1 ── Rcal ── gnd

Где:
- Rcal — точный известный резистор;
- CH1 измеряет audio_out;
- CH2 измеряет p1.

Алгоритм:
1. Проверить calibration_resistor > 0.
2. Если reference_resistor_nominal передан, проверить reference_resistor_nominal > 0.
3. Посчитать комплексные FFT-спектры:

   freq, V1, V2 = calculate_fft_spectra(ch1_cal, ch2_cal, fs)

4. Если smoothing=True:
   - сгладить V1 через utils.windows.log_filter2;
   - сгладить V2 через utils.windows.log_filter2.

5. Проверить, что размер channel_correction совпадает с размером V2 после всех ограничений/сглаживаний.
   Если не совпадает, выдать ValueError.

6. Скорректировать CH2:

   V2_corr = V2 / channel_correction

7. Посчитать:

   Hcal = V2_corr / V1

8. Посчитать частотную оценку Rr:

   Rr_by_freq = calibration_resistor * (1 - Hcal) / Hcal

9. В рабочем диапазоне частот оценить одно итоговое значение:

   Rr_estimated = median(real(Rr_by_freq))

   При расчёте median игнорировать NaN/Inf и явно плохие точки.

10. Сформировать diagnostics dict.

Diagnostics должен содержать хотя бы:
- rr_estimated;
- rr_nominal;
- rr_nominal_error_rel, если reference_resistor_nominal передан;
- rr_real_median;
- rr_real_mean;
- rr_real_std;
- rr_real_cv;
- rr_imag_abs_median;
- rr_imag_to_real_ratio;
- valid_points_count;
- warnings: list[str].

Рекомендуемые warning-условия:
- Rr_estimated <= 0;
- valid_points_count слишком маленький;
- rr_imag_to_real_ratio > 0.05;
- rr_real_cv > 0.03;
- если reference_resistor_nominal передан и abs(Rr_estimated - reference_resistor_nominal) / reference_resistor_nominal > 0.05.

Если данные подозрительные, не обязательно падать с ошибкой, но нужно выдать warning через warnings.warn и добавить текст в diagnostics["warnings"].

Вернуть:

freq, Rr_by_freq, Rr_estimated, diagnostics

4. Изменить calculate_impedance

Текущую функцию calculate_impedance изменить так, чтобы она принимала channel_correction и фактическое reference_resistor.

Сигнатура:

calculate_impedance(
    ch1: np.ndarray,
    ch2: np.ndarray,
    fs: float,
    reference_resistor: float,
    channel_correction: np.ndarray | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]

Алгоритм:
1. Проверить reference_resistor > 0.
2. Посчитать:

   freq, V1, V2 = calculate_fft_spectra(ch1, ch2, fs)

3. Если smoothing=True:
   - сгладить V1 через utils.windows.log_filter2;
   - сгладить V2 через utils.windows.log_filter2.

4. Если channel_correction передан:
   - проверить совпадение размеров;
   - посчитать V2_corr = V2 / channel_correction.

   Если channel_correction не передан:
   - V2_corr = V2.

5. Посчитать:

   H = V2_corr / V1

6. Посчитать:

   Z = reference_resistor * H / (1 - H)

7. Применить eps-защиту:
   - если abs(V1) < eps, Z = np.nan + 1j*np.nan;
   - если channel_correction передан и abs(channel_correction) < eps, Z = np.nan + 1j*np.nan;
   - если abs(1 - H) < eps, Z = np.nan + 1j*np.nan.

8. Ограничить диапазон f_min/f_max.

9. Вернуть freq, Z.

Важно:
- reference_resistor здесь должен быть уже Rr_estimated, а не обязательно номинал.
- Не использовать старую calibration = H_measured_cal / H_expected.
- Теперь correction и Rr_estimated считаются отдельно.

5. Изменить полный цикл измерения

Заменить старую функцию measure_impedance_with_inline_calibration или изменить её логику.

Новая логика:

measure_impedance_with_inline_calibration(
    fs: float,
    duration: float,
    reference_resistor_nominal: float,
    calibration_resistor: float,
    f_start: float = 20.0,
    f_end: float = 20000.0,
    amplitude: float = 0.2,
    input_device=None,
    output_device=None,
    input_channels: int = 2,
    output_channels: int = 2,
    f_min: float | None = None,
    f_max: float | None = None,
    smoothing: bool = True,
    raise_on_clipping: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]

Алгоритм:
1. Сгенерировать chirp.
2. Попросить пользователя соединить CH1 и CH2 с audio_out.
3. Дождаться Enter.
4. Проиграть chirp и записать CH1/CH2.
5. Обрезать запись.
6. Проверить уровни и clipping.
7. Посчитать channel_correction = Kch.
8. Попросить пользователя собрать схему:

   audio_out ── Rr ── p1 ── Rcal ── gnd

9. Дождаться Enter.
10. Проиграть chirp и записать CH1/CH2.
11. Обрезать запись.
12. Проверить уровни и clipping.
13. Посчитать Rr_by_freq, Rr_estimated, rr_diagnostics через estimate_reference_resistor.
14. Если diagnostics содержит warning, вывести их пользователю.
15. Попросить пользователя подключить измеряемую нагрузку:

   audio_out ── Rr ── p1 ── Zl ── gnd

16. Дождаться Enter.
17. Проиграть chirp и записать CH1/CH2.
18. Обрезать запись.
19. Проверить уровни и clipping.
20. Посчитать impedance через calculate_impedance, передав:
   - reference_resistor=Rr_estimated;
   - channel_correction=Kch.
21. Вернуть:

   freq, impedance, diagnostics

Diagnostics итоговой функции должен включать:
- channel_correction;
- rr_by_freq;
- rr_estimated;
- rr_diagnostics;
- level diagnostics для трёх записей.

6. Проверка уровней

Оставить/добавить analyze_recording_levels.

Требования:
- считать peak, rms, has_clipping, is_too_quiet по каждому каналу;
- если abs(signal) >= 1.0 хотя бы в одном отсчёте, это clipping;
- если raise_on_clipping=True, выдавать ValueError;
- если сигнал близок к clipping threshold, выдавать warning;
- сигнал не нормализовать и не менять.

7. Воспроизведение

play_and_record должен поддерживать stereo output.

Если output_channels == 2, chirp дублируется в оба выхода:

playback = np.column_stack([chirp, chirp])

argparse не нужен.

8. main

В конце файла оставить запуск:

if __name__ == "__main__":
    main()

argparse не использовать.
Параметры задаются руками внутри main().

main() должен:
1. Задать параметры в коде.
2. Запустить measure_impedance_with_inline_calibration.
3. Получить freq, impedance, diagnostics.
4. Напечатать Rr_estimated и основные diagnostics.
5. Построить график abs(impedance).
6. Вызвать plt.show().

Пример параметров внутри main:

fs = 48000
duration = 5.0
f_start = 20.0
f_end = 20000.0
amplitude = 0.2

reference_resistor_nominal = 10.0
calibration_resistor = 8.2

input_device = None
output_device = None
input_channels = 2
output_channels = 2

f_min = 20.0
f_max = 20000.0

smoothing = True
raise_on_clipping = True

9. График

Оставить простой график модуля импеданса:

plot_impedance_magnitude(freq, impedance)

Строить:
- abs(impedance) от freq;
- X axis log;
- grid;
- labels.

Фазу отдельным графиком пока не выводить.

10. Что убрать / не делать

Не делать:
- сохранение калибровки;
- загрузку калибровки;
- argparse;
- Welch;
- CSD;
- periodogram;
- abs(FFT) в расчёте;
- отношение спектров мощности;
- независимую нормализацию каналов;
- автоматическое исправление clipping;
- старую одностадийную calibration = H_measured_cal / H_expected.

11. Формулы

Этап 1. Калибровка каналов:

Kch = V2_same / V1_same

V2_corr = V2 / Kch

Этап 2. Оценка Rr:

Hcal = V2_cal_corr / V1_cal

Rr_by_freq = Rcal * (1 - Hcal) / Hcal

Rr_estimated = median(real(Rr_by_freq))

Этап 3. Измерение Z:

H = V2_meas_corr / V1_meas

Z = Rr_estimated * H / (1 - H)

12. Основной смысл изменения

Нужно разделить калибровку на две независимые части:
- channel_correction Kch компенсирует отличие CH2 от CH1;
- Rr_estimated оценивает фактическое сопротивление reference resistor в текущем состоянии.

Если Rr_estimated выглядит подозрительно, например имеет заметную мнимую часть, сильно гуляет по частоте или сильно отличается от номинала, нужно выдать warning.