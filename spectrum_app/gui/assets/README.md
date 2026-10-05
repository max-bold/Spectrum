# Branding assets

- `plot-logo.png` and `plot-logo-black.png`: unchanged white and black wordmarks
  copied from `SpectrumAndroid/logo/White@4x.png` and `Artboard 1@4x.png`.
- `app-icon.svg`: editable initial icon proposal, a spectrum envelope in cyan
  and blue on graphite. PNG, ICO and ICNS files are generated from the same
  geometry by `utils/build_brand_assets.py`.
- `splash-art.png`: background generated with the built-in ImageGen tool.
- `splash.png`: 560x320 background with title/version typography, prepared by
  `utils/build_brand_assets.py` for both Tk and the Windows bootloader splash.

Final ImageGen prompt:

> Use case: stylized-concept. Asset type: background illustration for a small
> desktop audio measurement application splash screen named BM Spectrum.
> Primary request: an elegant abstract visualization of sound and frequency
> analysis, a luminous flowing waveform merging into fine spectrum peaks, on
> a very dark graphite background. Wide landscape composition approximately
> 16:9. Keep the left half and bottom quarter dark and uncluttered for
> application title and loading progress added by code. Detailed soft cyan
> and cool blue light concentrated toward the right, subtle fine grid,
> restrained professional engineering instrument aesthetic, depth and gentle
> glow, not a music poster. No text, no letters, no logo, no watermark, no
> borders. Final image will be resized for a 560x320 splash.

To regenerate derived assets on Windows, install Pillow in the development
environment and run `python utils/build_brand_assets.py`. Update
`spectrum_app/version.py` before regenerating for a new release. Runtime image
display uses Tk and Dear PyGui and does not require Pillow.
