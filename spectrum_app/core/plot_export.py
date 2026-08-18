from collections.abc import Callable
from pathlib import Path
from typing import Any

import dearpygui.dearpygui as dpg
import numpy as np

ExportComplete = Callable[[Path | None, str | None], None]


class PlotExportError(RuntimeError):
    """Raised when a rendered Dear PyGui plot cannot be exported."""


class PlotExporter:
    """Save the pixels rendered by Dear PyGui for one plot as a PNG."""

    def export(
        self,
        path: str | Path,
        plot: int | str,
        *,
        on_complete: ExportComplete | None = None,
    ) -> Path:
        output_path = self._png_path(path)

        def save_framebuffer(sender: Any, framebuffer: Any, user_data=None) -> None:
            try:
                self._save_crop(output_path, plot, framebuffer)
            except Exception as error:
                if on_complete is not None:
                    on_complete(None, str(error) or error.__class__.__name__)
                return
            if on_complete is not None:
                on_complete(output_path, None)

        try:
            dpg.output_frame_buffer(callback=save_framebuffer)
        except Exception as error:
            raise PlotExportError(
                str(error) or "Could not request Dear PyGui framebuffer"
            ) from error
        return output_path

    @staticmethod
    def _save_crop(path: Path, plot: int | str, framebuffer: Any) -> None:
        frame_width = int(framebuffer.get_width())
        frame_height = int(framebuffer.get_height())
        if frame_width <= 0 or frame_height <= 0:
            raise PlotExportError("Dear PyGui returned an empty framebuffer")

        client_width = int(dpg.get_viewport_client_width())
        client_height = int(dpg.get_viewport_client_height())
        if client_width <= 0 or client_height <= 0:
            raise PlotExportError("Viewport has no drawable area")

        rect_min = dpg.get_item_rect_min(plot)
        rect_max = dpg.get_item_rect_max(plot)
        scale_x = frame_width / client_width
        scale_y = frame_height / client_height
        x_min = max(0, min(frame_width, round(float(rect_min[0]) * scale_x)))
        y_min = max(0, min(frame_height, round(float(rect_min[1]) * scale_y)))
        x_max = max(0, min(frame_width, round(float(rect_max[0]) * scale_x)))
        y_max = max(0, min(frame_height, round(float(rect_max[1]) * scale_y)))
        if x_max <= x_min or y_max <= y_min:
            raise PlotExportError("Plot has no visible rendered area")

        expected_size = frame_width * frame_height * 4
        pixel_count = len(framebuffer)
        if pixel_count != expected_size:
            raise PlotExportError(
                "Dear PyGui returned an invalid RGBA framebuffer "
                f"({pixel_count} values for {frame_width}x{frame_height})"
            )
        try:
            # DPG 2.3.1 reports a corrupt NumPy shape for mvBuffer on Windows:
            # C long is 32-bit there, while Py_ssize_t is 64-bit. Explicit
            # count makes NumPy use the correct byte length instead.
            pixels = np.frombuffer(
                framebuffer,
                dtype=np.float32,
                count=pixel_count,
            )
        except (TypeError, ValueError):
            pixels = np.asarray(framebuffer, dtype=np.float32).reshape(-1)
        pixels = pixels.reshape(frame_height, frame_width, 4)
        crop = np.ascontiguousarray(pixels[y_min:y_max, x_min:x_max])
        crop_bytes = np.rint(np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8)
        height, width, _ = crop.shape
        try:
            dpg.save_image(
                str(path),
                width,
                height,
                crop_bytes.reshape(-1),
                components=4,
            )
        except Exception as error:
            raise PlotExportError(str(error) or "Could not save plot PNG") from error

    @staticmethod
    def _png_path(path: str | Path) -> Path:
        result = Path(path)
        if result.suffix.lower() != ".png":
            result = result.with_suffix(".png")
        return result
