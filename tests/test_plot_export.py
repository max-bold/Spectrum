from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from spectrum_app.core.plot_export import PlotExporter


class FakeFrameBuffer:
    def __init__(self, pixels: np.ndarray) -> None:
        self.pixels = pixels.astype(np.float32)

    def get_width(self) -> int:
        return int(self.pixels.shape[1])

    def get_height(self) -> int:
        return int(self.pixels.shape[0])

    def __len__(self) -> int:
        return int(self.pixels.size)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self.pixels.reshape(-1), dtype=dtype)


class FakeExportBackend:
    def __init__(self) -> None:
        self.framebuffer = FakeFrameBuffer(
            np.linspace(0.0, 1.0, 6 * 8 * 4, dtype=np.float32).reshape(6, 8, 4)
        )
        self.saved: tuple[str, int, int, np.ndarray, int] | None = None

    def output_frame_buffer(self, *, callback) -> None:
        callback(None, self.framebuffer)

    def get_viewport_client_width(self) -> int:
        return 4

    def get_viewport_client_height(self) -> int:
        return 3

    def get_item_rect_min(self, item) -> list[float]:
        return [1.0, 1.0]

    def get_item_rect_max(self, item) -> list[float]:
        return [3.0, 2.5]

    def save_image(
        self,
        path: str,
        width: int,
        height: int,
        data,
        *,
        components: int,
    ) -> None:
        self.saved = (
            path,
            width,
            height,
            np.asarray(data).copy(),
            components,
        )


class PlotExporterTests(unittest.TestCase):
    def test_dpg_framebuffer_is_hidpi_scaled_cropped_and_saved_as_png(self) -> None:
        backend = FakeExportBackend()
        completed: list[tuple[Path | None, str | None]] = []

        with (
            TemporaryDirectory() as directory,
            patch("spectrum_app.core.plot_export.dpg", backend),
        ):
            path = PlotExporter().export(
                Path(directory) / "response",
                "plot",
                on_complete=lambda result, error: completed.append((result, error)),
            )

        self.assertEqual(path.suffix, ".png")
        self.assertEqual(completed, [(path, None)])
        saved = backend.saved
        self.assertIsNotNone(saved)
        assert saved is not None
        saved_path, width, height, data, components = saved
        self.assertEqual(saved_path, str(path))
        self.assertEqual((width, height, components), (4, 3, 4))
        expected = (
            np.rint(backend.framebuffer.pixels[2:5, 2:6] * 255.0)
            .astype(np.uint8)
            .reshape(-1)
        )
        np.testing.assert_array_equal(data, expected)


if __name__ == "__main__":
    unittest.main()
