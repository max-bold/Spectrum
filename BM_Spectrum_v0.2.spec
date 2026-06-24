# -*- mode: python ; coding: utf-8 -*-

import os
import sys


app_name = os.environ.get("APP_NAME", "BM_Spectrum")
target_arch = os.environ.get("PYINSTALLER_TARGET_ARCH") or None
is_macos = sys.platform == "darwin"

if target_arch not in {None, "x86_64", "arm64", "universal2"}:
    raise ValueError(
        "PYINSTALLER_TARGET_ARCH must be one of: x86_64, arm64, universal2"
    )


a = Analysis(
    ["run.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [] if is_macos else a.binaries,
    [] if is_macos else a.datas,
    [],
    exclude_binaries=is_macos,
    name=app_name,
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
        bundle_identifier="com.bm.spectrum",
    )
