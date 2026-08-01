# Example of target project structure for a spectrum application:

```
spectrum_app/
├── application.py
├── core/
│   ├── services.py
│   ├── signals.py
│   ├── dispatcher.py
│   ├── audio.py
│   └── tasks.py
├── modules/
│   ├── base.py
│   ├── manager.py
│   ├── spectrum/
│   │   ├── module.py
│   │   ├── state.py
│   │   ├── jobs.py
│   │   ├── view.py
│   │   └── renderers.py
│   └── impedance/
│       ├── module.py
│       ├── state.py
│       ├── jobs.py
│       ├── view.py
│       └── renderers.py
├── projects/
│   ├── model.py
│   └── storage.py
└── gui/
    ├── workspace.py
    ├── panels.py
    ├── axes.py
    └── layers.py
```