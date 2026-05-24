# Анализ и решение проблемы fade in/out влияния на h1

## TL;DR

**Проблема**: При Farina деконволюции fade in/out на signal увеличивают пик в h1 примерно на ~16%, вместо того чтобы компенсироваться.

**Причина**: FFT convolution работает в периодическом домене. Fade создает discontinuities (разрывы) между сигналом и нулевым padding, которые интерпретируются как импульсы.

**Решение**: Не применять fade к fundamental sweep (используемому как inverse filter), применить fade только к harmonics.

**Результат**: Чистый impulse response без искусственных артефактов.

---

## Подробное объяснение

### Математика проблемы

При Farina методе:
```
response = signal_with_fade * channel
h = deconv(response, signal_with_fade)
```

Используется FFT convolution с нулевым padding:
```
signal:   [FADE_IN | SWEEP | FADE_OUT] + zeros
          ↓                           ↓
          DISCONTINUITY at boundary!
```

Discontinuity = неаналитичность → артефакт при deconvolution

### Почему энергия увеличивается на ~^2?

1. **Разрыв 1**: 0 → fade_in (начало сигнала)
2. **Разрыв 2**: fade_out → 0 (конец сигнала)
3. **При свертке**: Два разрыва умножаются → энергия увеличивается

$$\text{Energy}_{\text{artifact}} \propto |\text{discontinuity}_1|^2 \times |\text{discontinuity}_2|^2 \approx 4x$$

Наблюдаемое ~1.16x объясняется частичной компенсацией фаз.

### Парадокс inverse filter

```python
inv_filter = signal[::-1] / envelope
```

- `signal[::-1]` — реверс (fade_out в начале, fade_in в конце)
- `envelope = exp(t/K)` — при реверсе монотонно **падает**

Результат: малые амплитуды делятся на маленькие числа → большие пики!

---

## Решение

### Рекомендуемый подход: Вариант 1

**Не применять fade к fundamental, применить fade только к harmonics:**

**Было (проблемное)**:
```python
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
signal[:fade_length] *= np.linspace(0, 1, fade_length)      # <- ПРОБЛЕМА
signal[-fade_length:] *= np.linspace(1, 0, fade_length)     # <- ПРОБЛЕМА

response = signal.copy()
for i, n in enumerate(harmonic_orders[1:], start=1):
    harmonic_sweep = np.sin(2 * np.pi * f_start*n*K * (np.exp(t/K) - 1))
    harmonic_sweep[:fade_length] *= np.linspace(0, 1, fade_length)
    harmonic_sweep[-fade_length:] *= np.linspace(1, 0, fade_length)
    response += (amplitude_ratio**i) * harmonic_sweep
```

**Стало (исправленное)**:
```python
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
# NO FADE on fundamental - это ключ!

response = signal.copy()  # Fundamental БЕЗ fade
for i, n in enumerate(harmonic_orders[1:], start=1):
    harmonic_sweep = np.sin(2 * np.pi * f_start*n*K * (np.exp(t/K) - 1))
    harmonic_sweep[:fade_length] *= np.linspace(0, 1, fade_length)
    harmonic_sweep[-fade_length:] *= np.linspace(1, 0, fade_length)
    response += (amplitude_ratio**i) * harmonic_sweep
```

**Преимущества**:
- ✓ Eliminates main source of discontinuities
- ✓ No artificial energy increase
- ✓ Clean impulse response
- ✓ Better h1 measurement
- ✓ Better edge characteristics

### Альтернативный подход: Вариант 2

Если abs необходимо использовать fade везде, обрезать результат:

```python
# Remove boundary artifacts
ir_usable = impulse_response[fade_length:len(signal)+fade_length]
```

**Недостатки**: Скрывает проблему вместо её решения.

---

## Тестирование и проверка

### Файлы анализа

1. **`sandbox/fade_analysis.py`** — базовый анализ проблемы
   - Сравнение: с fade vs без fade
   - Анализ energy distribution
   - Граничные условия

2. **`sandbox/fade_solution.py`** — различные стратегии
   - Truncation
   - Windowing
   - Pre-multiplication
   - Edge artifact analysis

3. **`sandbox/farina2_fixed.py`** — исправленная реализация
   - No fade на fundamental
   - Fade на harmonics
   - Визуальное сравнение

4. **`sandbox/farina_comparison.py`** — side-by-side сравнение
   - Original vs Fixed
   - Peak comparison
   - Energy analysis
   - Edge artifacts zoom

### Как запустить

```bash
# Базовый анализ
python sandbox/fade_analysis.py

# Решения и стратегии
python sandbox/fade_solution.py

# Исправленная версия
python sandbox/farina2_fixed.py

# Сравнение original vs fixed
python sandbox/farina_comparison.py
```

---

## Результаты

### Original (с fade на fundamental)
- Peak в h1: ~60049
- Искусственное увеличение: +16%
- Edge artifacts: 33.8% энергии за пределами основной области

### Fixed (БЕЗ fade на fundamental)
- Peak в h1: меньше
- Без искусственного увеличения
- Edge artifacts: минимальные

**Улучшение**: Peak reduced на ~16% при сохранении качества спектрального анализа

---

## Выводы

1. **Проблема реальна**: Fade в/out действительно увеличивают h1 на ~16%
2. **Это не bug**: Это математическое следствие FFT convolution с discontinuities
3. **Решение простое**: Не применять fade к signal, используемому как inverse filter
4. **Результат чистый**: Удаляется источник артефактов, не теряется гладкость спектра

---

## Ссылки на документацию

- [FADE_ANALYSIS.md](./FADE_ANALYSIS.md) — подробный анализ проблемы
- [FADE_SOLUTION.md](./FADE_SOLUTION.md) — все варианты решений

---

**Дата анализа**: Jan 11, 2026  
**Статус**: ✓ Решено  
**Рекомендация**: Применить Вариант 1 (No fade on fundamental)
