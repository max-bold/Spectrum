from __future__ import annotations

import os
import sys
from pathlib import Path

import dearpygui.dearpygui as dpg

FONT_SIZE = 16


def bind_app_font(size: int = FONT_SIZE) -> Path | None:
    font_path = find_app_font()
    if font_path is None:
        return None

    with dpg.font_registry():
        with dpg.font(str(font_path), size) as font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    dpg.bind_font(font)
    return font_path


def find_app_font() -> Path | None:
    candidates: list[Path]
    if sys.platform == "win32":
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [
            fonts / "segoeui.ttf",
            fonts / "arial.ttf",
            fonts / "tahoma.ttf",
        ]
    elif sys.platform == "darwin":
        candidates = [
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/Library/Fonts/Arial.ttf"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/opentype/noto/NotoSans-Regular.ttf"),
        ]

    return next((path for path in candidates if path.is_file()), None)
