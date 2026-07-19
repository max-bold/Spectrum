# Задача для Codex: исправить запрос доступа к микрофону на macOS Tahoe arm64

## Контекст

В проекте используется Python-библиотека `sounddevice`, которая работает через PortAudio/CoreAudio.

Проблема: на **macOS Tahoe arm64** приложение не вызывает системный prompt на доступ к микрофону. На более ранних версиях macOS всё работает. Нужно проверить и исправить поведение.

Предположение: проблема связана не с самим `sounddevice`, а с macOS TCC permissions. На новых версиях macOS приложение должно явно запросить доступ к микрофону через AVFoundation, а для app bundle нужны корректные `Info.plist`, entitlement и подпись. Developer ID у нас нет, поэтому нужна локальная проверка через **ad-hoc signing**.

## Цель

Сделать минимальный и понятный фикс, который:

1. На macOS явно запрашивает доступ к микрофону до первого обращения к `sounddevice`.
2. Не ломает запуск на Windows/Linux.
3. Даёт понятную ошибку, если доступ запрещён.
4. Позволяет собрать и локально подписать `.app` без Developer ID.
5. Позволяет проверить поведение на macOS Tahoe arm64.

## Что нужно реализовать

### 1. Добавить модуль проверки прав микрофона

Создать модуль, например:

```text
utils/macos_microphone.py
```

или другое подходящее место в проекте.

В модуле реализовать функцию:

```python
def ensure_microphone_permission() -> bool:
    ...
```

Поведение:

* Если платформа не macOS, вернуть `True`.
* Если macOS и доступ уже разрешён, вернуть `True`.
* Если macOS и статус `notDetermined`, вызвать системный запрос доступа к микрофону через AVFoundation.
* Если пользователь разрешил доступ, вернуть `True`.
* Если доступ запрещён или ограничен, вернуть `False`.
* Если PyObjC/AVFoundation недоступны, не падать странной ошибкой, а вернуть понятное исключение или диагностическое сообщение.

Примерная логика:

```python
import sys
import threading


def ensure_microphone_permission(timeout: float = 10.0) -> bool:
    if sys.platform != "darwin":
        return True

    try:
        from AVFoundation import (
            AVCaptureDevice,
            AVMediaTypeAudio,
            AVAuthorizationStatusAuthorized,
            AVAuthorizationStatusDenied,
            AVAuthorizationStatusRestricted,
            AVAuthorizationStatusNotDetermined,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyObjC AVFoundation is required on macOS to request microphone permission. "
            "Install pyobjc-framework-AVFoundation."
        ) from exc

    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio)

    if status == AVAuthorizationStatusAuthorized:
        return True

    if status in (AVAuthorizationStatusDenied, AVAuthorizationStatusRestricted):
        return False

    if status == AVAuthorizationStatusNotDetermined:
        event = threading.Event()
        result = {"granted": False}

        def callback(granted):
            result["granted"] = bool(granted)
            event.set()

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            AVMediaTypeAudio,
            callback,
        )

        event.wait(timeout)
        return result["granted"]

    return False
```

Нужно проверить, что callback корректно отрабатывает на Tahoe.

### 2. Вызвать проверку перед первым использованием sounddevice input

Найти места, где приложение впервые открывает stream, вызывает `sd.rec()`, `sd.InputStream()`, `sd.Stream()` или аналогичный input-захват.

Перед этим добавить:

```python
from utils.macos_microphone import ensure_microphone_permission

if not ensure_microphone_permission():
    raise RuntimeError(
        "Microphone access is denied. "
        "Enable it in System Settings > Privacy & Security > Microphone."
    )
```

Важно: запрос должен происходить до первого обращения к input-устройству через `sounddevice`.

### 3. Добавить зависимость для macOS

В зависимости проекта добавить PyObjC только для macOS:

```text
pyobjc-framework-AVFoundation; sys_platform == "darwin"
```

Например, в `requirements.txt`, `pyproject.toml` или другом используемом месте.

### 4. Проверить Info.plist для app bundle

Если проект собирается в `.app`, добавить в `Info.plist`:

```xml
<key>NSMicrophoneUsageDescription</key>
<string>Microphone access is required for acoustic measurements.</string>
```

Если используется PyInstaller `.spec`, добавить это в `info_plist`, например:

```python
app = BUNDLE(
    exe,
    name="BMSpectrum.app",
    bundle_identifier="ru.boldyrev.bmspectrum",
    info_plist={
        "NSMicrophoneUsageDescription": "Microphone access is required for acoustic measurements.",
    },
)
```

Название приложения и `bundle_identifier` можно подстроить под фактический проект.

### 5. Добавить entitlement для audio input

Создать файл:

```text
entitlements.plist
```

С содержимым:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.device.audio-input</key>
    <true/>
</dict>
</plist>
```

### 6. Добавить локальную ad-hoc подпись без Developer ID

Так как Developer ID нет, нужна команда для локального тестирования:

```bash
codesign --force --deep --options runtime \
  --entitlements entitlements.plist \
  --sign - \
  dist/BMSpectrum.app
```

Важно: это не нотарификация и не нормальная подпись для публичной раздачи. Это только локальная подпись для проверки TCC permissions на машине разработчика.

Если app называется иначе, поправить путь.

### 7. Добавить диагностический тест

Добавить маленький скрипт, например:

```text
sandbox/test_macos_microphone_permission.py
```

Сценарий:

1. Вывести платформу.
2. Вызвать `ensure_microphone_permission()`.
3. Если разрешение получено, записать 1 секунду через `sounddevice`.
4. Вывести peak level.
5. Если разрешение не получено, вывести понятное сообщение.

Пример:

```python
import sys
import numpy as np
import sounddevice as sd

from utils.macos_microphone import ensure_microphone_permission


def main():
    print("platform:", sys.platform)

    granted = ensure_microphone_permission()
    print("microphone permission:", granted)

    if not granted:
        print("Microphone permission denied.")
        print("Enable it in System Settings > Privacy & Security > Microphone.")
        return

    fs = 48000
    print("devices:")
    print(sd.query_devices())

    print("recording 1 second...")
    x = sd.rec(int(fs * 1), samplerate=fs, channels=1, dtype="float32")
    sd.wait()

    print("peak:", float(np.max(np.abs(x))))


if __name__ == "__main__":
    main()
```

### 8. Добавить инструкцию для проверки на macOS Tahoe arm64

В README или отдельный файл добавить раздел:

```bash
tccutil reset Microphone
```

Потом запуск из Terminal:

```bash
python sandbox/test_macos_microphone_permission.py
```

Ожидаемое поведение:

* macOS должна показать prompt на доступ к микрофону.
* После разрешения запись должна пройти.
* В System Settings > Privacy & Security > Microphone должен появиться Terminal, Python, приложение или app bundle, в зависимости от способа запуска.

Проверка `.app`:

```bash
tccutil reset Microphone
```

Собрать app bundle.

Подписать ad-hoc:

```bash
codesign --force --deep --options runtime \
  --entitlements entitlements.plist \
  --sign - \
  dist/BMSpectrum.app
```

Запустить `.app`.

Ожидаемое поведение:

* macOS должна показать prompt на доступ к микрофону.
* Если пользователь разрешил доступ, запись через `sounddevice` должна работать.
* В настройках приватности должен появиться app bundle.

### 9. Acceptance criteria

Готово, если:

1. На macOS Tahoe arm64 при первом запуске появляется prompt на доступ к микрофону.
2. После разрешения `sounddevice` может записать 1 секунду аудио.
3. При запрете доступа приложение показывает понятное сообщение, а не падает непонятной ошибкой PortAudio.
4. На Windows/Linux поведение не изменилось.
5. Локальная `.app` сборка работает после ad-hoc signing без Developer ID.
6. В репозитории есть короткая инструкция по проверке на macOS.

## Важно

Developer ID нет. Не нужно настраивать notarization и полноценную distribution signing.

Нужно добиться именно локальной проверки:

* `.app`
* `Info.plist`
* audio input entitlement
* ad-hoc codesign
* явный AVFoundation permission request
* проверка `sounddevice` после получения разрешения
