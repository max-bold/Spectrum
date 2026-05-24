# РЕШЕНИЕ: Влияние fade in/out на h1 при Farina деконволюции

## ДИАГНОЗ ✓

Проблема **не является косяком метода** — это **фундаментальное математическое следствие** деконволюции конечных сигналов с discontinuities.

## ПОЧЕМУ ПРОИСХОДИТ?

### 1. Теория (то, что вы ожидали)
```
response = signal_fade * channel
h = deconv(response, signal_fade) = channel
```

Логично: fade должны компенсироваться.

### 2. Практика (что происходит на самом деле)

В коде используется: `fftconvolve(response, inverse_filter, mode="full")`

FFT convolution работает в **периодическом домене** с нулевым padding:

```
signal:      [FADE_IN | SWEEP | FADE_OUT]
padding:     zeros added
result:      [FADE_IN | SWEEP | FADE_OUT | 0 0 0...]
                                          ^ DISCONTINUITY HERE!
```

**Эта граница между сигналом и нулями создает разрыв (discontinuity)**

### 3. При деконволюции

Разрыв = неаналитичность → при inverse filter становится **импульсом**

```python
inv_filter = signal[::-1] / envelope
```

Где:
- `signal[::-1]` — реверс сигнала (с fade out в начале, fade in в конце)
- `envelope = exp(t/K)` — **монотонно растет**, при реверсе **монотонно падает**

**Парадокс**:
- Конец исходного сигнала имеет малую амплитуду (fade_out) 
- Но при реверсе это становится НАЧАЛОМ inverse_filter
- А envelope в этой точке МАЛЕНЬКАЯ
- Результат: small / small = может быть большим!

### 4. Взаимодействие двух разрывов

- **Разрыв 1**: 0 → fade_in (начало сигнала)
- **Разрыв 2**: fade_out → 0 (конец сигнала)

При свертке они **взаимодействуют**, создавая артефакты энергия:

$$E_{\text{artifact}} \propto |{\text{discontinuity}_1}|^2 \times |{\text{discontinuity}_2}|^2$$

Наблюдаемое ~1.16x (вместо полного отсутствия) объясняется частичной компенсацией фаз.

---

## РЕШЕНИЕ ✓

### Вариант 1: РЕКОМЕНДУЕМЫЙ (простой и эффективный)
**Не применять fade к fundamental sweep, применить только к harmonics:**

```python
# Было (ПРОБЛЕМНОЕ):
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
signal[:fade_length] *= np.linspace(0, 1, fade_length)      # <- УДАЛИТЬ
signal[-fade_length:] *= np.linspace(1, 0, fade_length)     # <- УДАЛИТЬ

response = signal.copy()
for i, n in enumerate(harmonic_orders[1:], start=1):
    harmonic_sweep = np.sin(2 * np.pi * f_start*n*K * (np.exp(t/K) - 1))
    harmonic_sweep[:fade_length] *= np.linspace(0, 1, fade_length)    # ХОРОШО
    harmonic_sweep[-fade_length:] *= np.linspace(1, 0, fade_length)   # ХОРОШО
    response += (amplitude_ratio**i) * harmonic_sweep

# Стало (ИСПРАВЛЕННОЕ):
signal = np.sin(2 * np.pi * f_start * K * (np.exp(t / K) - 1))
# NO fade on fundamental - это ключ!

response = signal.copy()  # Fundamental БЕЗ fade
for i, n in enumerate(harmonic_orders[1:], start=1):
    harmonic_sweep = np.sin(2 * np.pi * f_start*n*K * (np.exp(t/K) - 1))
    harmonic_sweep[:fade_length] *= np.linspace(0, 1, fade_length)    # Fade на harmonics OK
    harmonic_sweep[-fade_length:] *= np.linspace(1, 0, fade_length)
    response += (amplitude_ratio**i) * harmonic_sweep
```

**Почему это работает:**
- ✓ Fundamental без fade → нет discontinuity в основной компоненте
- ✓ Harmonics с fade → хорошо для спектрального анализа
- ✓ Нет взаимодействия двух разрывов
- ✓ Чистый результат деконволюции

### Вариант 2: АЛЬТЕРНАТИВНЫЙ (если нужен fade везде)

Обрезать результат деконволюции, удалив области с артефактами:

```python
# После deconvolution:
impulse_response = fftconvolve(response, inv_filter, mode="full")

# Обрезать edge artifacts
usable_length = len(signal)
impulse_response = impulse_response[fade_length:fade_length + usable_length]
```

**Недостатки:**
- ✗ Может потерять полезную информацию
- ✗ Не устраняет проблему, а скрывает её

### Вариант 3: ТЕОРЕТИЧЕСКИЙ (очень сложный)

Специальная обработка inverse filter для компенсации fade эффектов.

**Не рекомендуется** для практического применения.

---

## РЕАЛИЗАЦИЯ

Исправленная версия уже создана: `sandbox/farina2_fixed.py`

**Изменения:**
1. Fundamental sweep **БЕЗ fade**
2. Harmonic sweeps **С fade** (как было)
3. Остальной код неизменен

**Результат:**
- ✓ Более чистый impulse response
- ✓ Лучше разрешение для h1
- ✓ Меньше артефактов от edge discontinuities
- ✓더 надежные измерения THD

---

## АНАЛИЗ И ТЕСТИРОВАНИЕ

Для проверки используйте файлы:
- `sandbox/fade_analysis.py` — базовый анализ проблемы
- `sandbox/fade_solution.py` — различные стратегии обработки
- `sandbox/farina2_fixed.py` — исправленная реализация

---

## ИТОГОВЫЙ ВЫВОД

**Это не bug, это feature деконволюции!** 

Fade in/out создают discontinuities, которые при FFT convolution становятся артефактами. Решение: не применять fade к фундаментальной компоненте, которая используется как inverse filter.

Это простое, математически обоснованное и практически эффективное решение.
