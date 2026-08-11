"""Plot overlap of the logarithmic windows used by grid_smooth()."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BALANCED_GAUSSIAN_SIGMA_RATIO = 0.399284795102118


def load_smoothing_module():
    """Load smoothing.py without importing optional audio I/O dependencies."""
    module_path = ROOT / "audioanalysis" / "smoothing.py"
    spec = importlib.util.spec_from_file_location("smoothing_for_plot", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def placed_windows(
    smoothing,
    frequency,
    grid,
    window,
    width,
    gaussian_sigma_ratio=None,
    use_old=False,
):
    """Place log_window() weights on exactly the bins used by grid_smooth()."""
    frequency_step = float(frequency[1] - frequency[0])
    total = np.zeros_like(frequency)
    components: list[tuple[np.ndarray, np.ndarray]] = []

    for center in grid:
        if use_old:
            weights, start, end = smoothing.log_window_old(
                window,
                float(center),
                frequency_step,
                width,
            )
        elif (
            window == smoothing.SmoothingWindow.GAUSSIAN
            and gaussian_sigma_ratio is not None
        ):
            sigma = gaussian_sigma_ratio * width
            radius = sigma * np.sqrt(-2.0 * np.log(smoothing.WINDOW_EDGE_WEIGHT))
            low = float(center) / (2.0**radius)
            high = float(center) * (2.0**radius)
            start = int(np.ceil(low / frequency_step))
            end = int(np.ceil(high / frequency_step))
            window_frequency = frequency[start:end]
            offsets = np.log2(window_frequency / float(center))
            weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        else:
            weights, start, end = smoothing.log_window(
                window,
                float(center),
                frequency_step,
                width,
            )
        data_start = max(0, start)
        data_end = min(end, len(frequency))
        weight_start = data_start - start
        weight_end = weight_start + data_end - data_start
        if data_start >= data_end:
            continue

        placed_weights = weights[weight_start:weight_end]
        total[data_start:data_end] += placed_weights
        components.append((frequency[data_start:data_end], placed_weights))

    return components, total


def draw_overlap_axis(
    axis,
    frequency,
    components,
    total,
    x_limits,
    panel_title,
):
    for index, (window_frequency, weights) in enumerate(components):
        axis.semilogx(
            window_frequency,
            weights,
            color="tab:orange",
            alpha=0.38,
            linewidth=0.9,
            label="individual windows" if index == 0 else None,
        )

    visible = (frequency >= x_limits[0]) & (frequency <= x_limits[1])
    axis.semilogx(
        frequency[visible],
        total[visible],
        color="tab:blue",
        linewidth=2.2,
        label="sum of placed windows",
        zorder=4,
    )
    axis.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.75,
        label="target sum = 1",
    )
    axis.set_xlim(*x_limits)
    upper_limit = max(1.10, float(total[visible].max()) * 1.06)
    axis.set_ylim(-0.04, upper_limit)
    axis.set_title(panel_title)
    axis.set_ylabel("weight / sum")
    axis.grid(True, which="both", alpha=0.28)


def output_path_for(window, width):
    width_slug = f"{width:.9f}".replace(".", "")
    output_dir = ROOT / "artifacts" / f"smoothing_width_{width_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{window.value}_overlap_width_{width_slug}.png"


def draw_window_plot(
    smoothing,
    frequency: np.ndarray,
    grid: np.ndarray,
    window,
    width: float,
    gaussian_sigma_ratio: float | None = None,
) -> Path:
    components, total = placed_windows(
        smoothing,
        frequency,
        grid,
        window,
        width,
        gaussian_sigma_ratio,
    )
    spacing = float(np.diff(np.log2(grid)).mean())
    zoom = (650.0, 1650.0)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160)
    sigma_text = (
        f", sigma={gaussian_sigma_ratio:.9f} * width"
        if gaussian_sigma_ratio is not None
        else ""
    )
    figure.suptitle(
        f"{window.value}: width={width:.9f} octave, "
        f"grid spacing={spacing:.9f} octave{sigma_text}",
        fontsize=14,
    )

    for axis, x_limits, panel_title in (
        (axes[0], (20.0, 20_000.0), "Full smoothing grid"),
        (axes[1], zoom, "Between neighboring grid points (zoom)"),
    ):
        draw_overlap_axis(
            axis,
            frequency,
            components,
            total,
            x_limits,
            panel_title,
        )

    axes[1].set_xlabel("frequency, Hz (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path = output_path_for(window, width)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def draw_gaussian_comparison(
    smoothing,
    frequency: np.ndarray,
    grid: np.ndarray,
    width: float,
) -> Path:
    window = smoothing.SmoothingWindow.GAUSSIAN
    old_components, old_total = placed_windows(
        smoothing,
        frequency,
        grid,
        window,
        width,
        use_old=True,
    )
    balanced_components, balanced_total = placed_windows(
        smoothing,
        frequency,
        grid,
        window,
        width,
        BALANCED_GAUSSIAN_SIGMA_RATIO,
    )
    spacing = float(np.diff(np.log2(grid)).mean())
    zoom = (650.0, 1650.0)
    figure, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=160)
    figure.suptitle(
        f"Gaussian comparison: width={width:.9f} octave, "
        f"grid spacing={spacing:.9f} octave",
        fontsize=14,
    )

    columns = (
        (old_components, old_total, "log_window_old()"),
        (
            balanced_components,
            balanced_total,
            f"balanced: sigma={BALANCED_GAUSSIAN_SIGMA_RATIO:.9f} * width",
        ),
    )
    for column, (components, total, title) in enumerate(columns):
        draw_overlap_axis(
            axes[0, column],
            frequency,
            components,
            total,
            (20.0, 20_000.0),
            title,
        )
        draw_overlap_axis(
            axes[1, column],
            frequency,
            components,
            total,
            zoom,
            "Between neighboring centers (zoom)",
        )
        axes[1, column].set_xlabel("frequency, Hz (log scale)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path = output_path_for(window, width)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Window width in octaves; defaults to the exact logarithmic grid step",
    )
    parser.add_argument(
        "--window",
        choices=("all", "flat", "cosine", "gaussian", "triangular"),
        default="all",
    )
    parser.add_argument("--balanced-gaussian", action="store_true")
    args = parser.parse_args()

    smoothing = load_smoothing_module()
    frequency = np.linspace(0.0, 40e3, 100_000)
    grid = np.geomspace(20.0, 20e3, 30)

    spacing = float(np.log2(grid[-1] / grid[0]) / (len(grid) - 1))
    width = spacing if args.width is None else args.width
    print(f"FFT step: {frequency[1] - frequency[0]:.9f} Hz")
    print(f"Grid spacing: {spacing:.9f} octave")
    print(f"Window minus spacing: {width - spacing:.9f} octave")
    windows = (
        list(smoothing.SmoothingWindow)
        if args.window == "all"
        else [smoothing.SmoothingWindow(args.window)]
    )
    for window in windows:
        if (
            args.balanced_gaussian
            and window == smoothing.SmoothingWindow.GAUSSIAN
        ):
            print(draw_gaussian_comparison(smoothing, frequency, grid, width))
            continue
        gaussian_sigma_ratio = (
            BALANCED_GAUSSIAN_SIGMA_RATIO
            if args.balanced_gaussian
            and window == smoothing.SmoothingWindow.GAUSSIAN
            else None
        )
        print(
            draw_window_plot(
                smoothing,
                frequency,
                grid,
                window,
                width,
                gaussian_sigma_ratio,
            )
        )


if __name__ == "__main__":
    main()
