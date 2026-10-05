# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.splash import Splash


app_name = os.environ.get("APP_NAME", "BM_Spectrum")
target_arch = os.environ.get("PYINSTALLER_TARGET_ARCH") or None
is_macos = sys.platform == "darwin"
project_root = os.path.abspath(SPECPATH)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
from spectrum_app.version import APP_VERSION

if target_arch not in {None, "x86_64", "arm64", "universal2"}:
    raise ValueError(
        "PYINSTALLER_TARGET_ARCH must be one of: x86_64, arm64, universal2"
    )

hidden_imports = collect_submodules("spectrum_app.modules")
application_data = collect_data_files(
    "spectrum_app",
    includes=["gui/icons/*.png", "gui/assets/plot-logo.png", "gui/assets/splash.png",
              "gui/assets/app-icon.ico", "gui/assets/app-icon.icns"],
)

a = Analysis(
    ["run.py"],
    pathex=[project_root],
    binaries=[],
    datas=application_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Windows bootloader splash appears before one-file extraction and DSP imports.
# macOS uses the lightweight Tk helper because bootloader splash is unsupported.
splash = None
if sys.platform == "win32":
    splash = Splash(
        os.path.join(project_root, "spectrum_app/gui/assets/splash.png"),
        binaries=a.binaries, datas=a.datas, text_pos=(30, 280),
        text_size=11, text_color="#d5e6ef", text_default="Starting BM Spectrum...",
        # PyInstaller inserts this family into Tcl without quoting. A family
        # containing spaces aborts the script and leaves a blank Tk window.
        text_font="Arial", always_on_top=True,
    )

exe = EXE(
    pyz,
    a.scripts,
    *([splash, splash.binaries] if splash else []),
    [] if is_macos else a.binaries,
    [] if is_macos else a.datas,
    [],
    exclude_binaries=is_macos,
    name=app_name,
    icon=os.path.join(project_root, "spectrum_app/gui/assets/" +
                     ("app-icon.icns" if is_macos else "app-icon.ico")),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=target_arch,
    codesign_identity=None,
    entitlements_file=None,
)

if is_macos:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name,
    )
    app = BUNDLE(
        coll,
        name=f"{app_name}.app",
        icon=os.path.join(project_root, "spectrum_app/gui/assets/app-icon.icns"),
        bundle_identifier="com.bm.spectrum",
        version=APP_VERSION,
    )
