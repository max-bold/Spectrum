import numpy as np
from typing import Literal


def log_window(
    window: Literal["flat", "cosine", "gaussian", "triangular"],
    fc: float,
    df: float,
    w: float,
) -> tuple[np.ndarray, int, int]:
    """_summary_

    Args:
        window (str): Shape of window:
            - "flat" - for flat window (np.ones())
            - "cosine" - for cosine window
            - "gaussian" for gaussian window
            - "triangular" - a window with constant fall in db/oct
        fc (float): Central frequency in Hz
        df (float): Frequency step (fs/n)
        w (float): window width in octaves

    Returns:
        tuple[np.ndarray, int, int]: _description_
    """
    hw = w / 2
    f_min = fc / (2**hw)
    f_max = fc * (2**hw)
    fs = np.arange(f_min, f_max, df)
    fs_log = np.log2(fs) - np.log2(fc)

    if window == "flat":
        ws = np.ones_like(fs_log)
    elif window == "gaussian":
        ws = np.exp(-((fs_log / hw * 4) ** 2) / 2)
    elif window == "cosine":
        ws = np.cos(np.pi * fs_log / hw) / 2 + 0.5
    elif window == "triangular":
        n = 10 ** (-30 / hw / 10)
        ws = np.power(n, np.abs(fs_log))
    else:
        raise ValueError(f"Unknown window shape: {window}")

    start = int(np.rint(f_min / df))
    end = start + len(ws)
    return ws, start, end


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    fc = 1000
    df = 2
    w = 1
    f_min = fc / (2 ** (w / 2))
    f_max = fc * (2 ** (w / 2))
    fs = np.arange(f_min, f_max, df)
    for window_shape in Literal["flat", "cosine", "gaussian", "triangular"].__args__:
        ws, start, end = log_window(window_shape, fc, df, 1)
        fig, axs = plt.subplots(2)
        axs: list[Axes]
        axs[0].plot(np.log2(fs) - np.log2(fc), 10 * np.log10(ws.clip(1e-20)))
        axs[1].plot(fs, ws)
        axs[0].grid(True, "both")
        axs[1].grid(True, "both")
        axs[0].xaxis.set_label_text("w (octaves)")
        axs[0].yaxis.set_label_text("(db)")
        axs[1].xaxis.set_label_text("f (Hz)")
        axs[0].set_ylim(-40, 5)
        fig.suptitle(f"{window_shape} window")
        fig.tight_layout()
        with open(f"info/{window_shape} window.png", "wb") as f:
            fig.savefig(f)
