"""Render editable icon geometry and splash typography into packaged assets."""
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectrum_app.version import APP_NAME, APP_VERSION

ASSETS = ROOT / "spectrum_app/gui/assets"


def build() -> None:
    # Original simple vector design: a spectrum whose envelope suggests a wave.
    bars = [(100, 205, 128, 307), (150, 156, 178, 356), (200, 100, 228, 412),
            (250, 128, 278, 384), (300, 180, 328, 332), (350, 218, 378, 294),
            (400, 194, 428, 318)]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">',
           '<rect x="0" y="0" width="512" height="512" rx="100" fill="#101923"/>']
    icon = Image.new("RGBA", (512, 512))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle((0, 0, 511, 511), radius=100, fill="#101923")
    for index, box in enumerate(bars):
        color = "#57daef" if index < 4 else "#579af2"
        draw.rounded_rectangle(box, radius=14, fill=color)
        x1, y1, x2, y2 = box
        svg.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" rx="14" fill="{color}"/>')
    svg.append('</svg>')
    (ASSETS / "app-icon.svg").write_text("\n".join(svg), encoding="utf-8")
    icon.save(ASSETS / "app-icon.png")
    icon.save(ASSETS / "app-icon.ico", sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
    icon.save(ASSETS / "app-icon.icns")

    background = Image.open(ASSETS / "splash-art.png").convert("RGB").resize((560, 320), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(background)
    font_dir = Path("C:/Windows/Fonts")
    title = ImageFont.truetype(str(font_dir / "segoeuib.ttf"), 32)
    subtitle = ImageFont.truetype(str(font_dir / "segoeui.ttf"), 14)
    draw.text((30, 32), APP_NAME, font=title, fill="#effaff")
    draw.text((32, 78), f"Version {APP_VERSION}  /  Audio measurement", font=subtitle, fill="#9cb8c7")
    background.save(ASSETS / "splash.png")


if __name__ == "__main__":
    build()
