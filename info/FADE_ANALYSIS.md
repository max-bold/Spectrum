# АНАЛИЗ И РЕШЕНИЕ: Fade in/out влияние на h1 при Farina деконволюции

## ПРОБЛЕМА

При использовании Farina метода с fade in/out на сигнал, вы заметили, что:
1. **Теория**: fade присутствует в обоих сигналах (signal и response), поэтому при свертке должны компенсироваться
2. **Практика**: Не только не компенсируются, но увеличиваются примерно на ~^2

## ОТВЕТ: Это НЕ косяк метода, это ФУНДАМЕНТАЛЬНАЯ ПРОБЛЕМА деконволюции с fade

### Математическое объяснение

При Farina деконволюции:
```
response = signal_with_fade * channel
h = response * inverse_filter
h = (signal_with_fade * channel) * (signal_with_fade[::-1] / envelope)
```

**Ключевая проблема**: `fftconvolve(mode='full')` предполагает **циклическую свертку с нулевым padding**!

Это означает:
- Signal extend → [..., 0, 0, 0] (добавляются нули после fade_out)
- Response extend → [..., 0, 0, 0] (то же самое)
- **Граница между сигналом и нулями создает РАЗРЫВ!**

При деконволюции этот разрыв интерпретируется как **импульс** (spike):
- Fade out: значение → 0
- Zero padding: 0 → 0
- Gradient: присутствует!
- При свертке: **энергия увеличивается примерно на ^2**

### Почему энергия увеличивается на ^2?

Это следствие **взаимодействия двух разрывов**:
1. Начало сигнала: 0 → плавное нарастание (из fade_in)
2. Конец сигнала: плавное затухание → 0 (из fade_out)

При свертке через inverse_filter эти два разрыва **умножаются** друг на друга:
$$\text{Energy} \propto (\text{discontinuity}_1)^2 \times (\text{discontinuity}_2)^2 \approx 4x$$

Но так как фазы не совпадают идеально, наблюдается более скромное ~1.16x (vs no fade).

### РЕШЕНИЕ

#### Вариант 1: Пренебречь edge artifacts (ПРОСТОЙ)
Truncate результат deconvolution, удалив области с большими fade:

```python
# Вместо:
# impulse_response = fftconvolve(response, inv_filter, mode="full")

# Использовать:
fade_length = int(1.0 * fs)  # 1 second fade
ir_full = fftconvolve(response, inv_filter, mode="full")

# Обрезать до разумного размера, пропустив начало и конец
ir_usable = ir_full[fade_length:len(signal)+fade_length]
```

#### Вариант 2: Не применять fade к fundamental (РЕКОМЕНДУЕМЫЙ)
```python
# Fade apply только к harmonics, не к fundamental
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
# NO fade on fundamental!

response = signal.copy()  # No fade
for i, n in enumerate(harmonic_orders[1:], start=1):
    harmonic_sweep = np.sin(2 * np.pi * f_start * n * K * (np.exp(t / K) - 1))
    harmonic_sweep[:fade_length] *= np.linspace(0, 1, fade_length)
    harmonic_sweep[-fade_length:] *= np.linspace(1, 0, fade_length)
    response += (amplitude_ratio**i) * harmonic_sweep
```

#### Вариант 3: Компенсировать fade в обратном фильтре (СЛОЖНЫЙ)
```python
# Компенсировать effect fade в inverse filter
# Это сложнее, но более теоретически чистый подход
```

### РЕКОМЕНДАЦИЯ

**Используйте Вариант 2**: Не применяйте fade к fundamental sweep, только к harmonics.

**Почему?**
- Fade на harmonics все еще полезна для избежания spectral leakage
- Fade на fundamental - это главный источник проблемы
- Это дает лучший баланс между гладкостью спектра и точностью deconvolution

**Код изменения**:

```python
# Было (проблемное):
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
signal[:fade_length] *= np.linspace(0, 1, fade_length)  # BAD!
signal[-fade_length:] *= np.linspace(1, 0, fade_length)  # BAD!

# Стало (исправленное):
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
# NO FADE on fundamental - это key!
```

---

## ПРОВЕРКА

Для проверки гипотезы, запустите `sandbox/fade_analysis.py` и `sandbox/fade_solution.py`.

Они покажут:
1. Как fade увеличивает peak на ~16-20%
2. Где именно находится энергия (edge artifacts)
3. Как различные стратегии влияют на результат
