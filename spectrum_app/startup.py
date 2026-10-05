"""Lightweight startup UI: no DSP, audio, or Dear PyGui imports here."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from spectrum_app.version import APP_NAME, APP_VERSION


class StartupSplash:
    def __init__(self) -> None:
        self._process = None
        self._bundled = None
        self._directory = None
        self._status = None

    def start(self) -> None:
        if getattr(sys, "frozen", False):
            try:
                import pyi_splash
                if pyi_splash.is_alive():
                    self._bundled = pyi_splash
                    self.update("Loading application", 0)
                    return
            except (ImportError, RuntimeError):
                pass
        self._directory = tempfile.TemporaryDirectory(prefix="bm-spectrum-startup-")
        self._status = Path(self._directory.name) / "status.json"
        self._send({"message": "Loading application", "progress": 0})
        command = ([sys.executable, "--splash", str(self._status)] if getattr(sys, "frozen", False)
                   else [sys.executable, "-m", "spectrum_app.startup", "--splash", str(self._status)])
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
                cwd=str(Path(__file__).resolve().parents[1]),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            # The analyzer remains usable if the optional splash cannot start.
            self._process = None
            self.close()

    def update(self, message: str, progress: float) -> None:
        if self._bundled is not None:
            try:
                self._bundled.update_text(f"{message}  ({progress:.0%})")
            except (OSError, RuntimeError):
                self._bundled = None
        self._send({"message": message, "progress": progress})

    def _send(self, value: dict) -> None:
        if self._status is None:
            return
        try:
            temporary = self._status.with_suffix(".tmp")
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(self._status)
        except OSError:
            pass

    def close(self) -> None:
        if self._bundled is not None:
            try:
                self._bundled.close()
            except (OSError, RuntimeError):
                pass
            self._bundled = None
        process = self._process
        self._process = None
        if process is not None:
            self._send({"close": True})
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=1)
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None
            self._status = None


def show_splash(status_path: str) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.geometry(f"560x320+{(root.winfo_screenwidth()-560)//2}+{(root.winfo_screenheight()-320)//2}")
    canvas = tk.Canvas(root, width=560, height=320, highlightthickness=0, bg="#0b121b")
    canvas.pack()
    background = Path(__file__).parent / "gui/assets/splash.png"
    if background.exists():
        image = tk.PhotoImage(file=str(background))
        canvas.create_image(0, 0, image=image, anchor="nw")
    else:
        canvas.create_text(30, 35, text=f"{APP_NAME} {APP_VERSION}", anchor="nw",
                           fill="white", font=("Segoe UI", 24, "bold"))
    status = canvas.create_text(30, 270, text="Loading application", anchor="nw",
                                fill="#d5e6ef", font=("Segoe UI", 11))
    canvas.create_rectangle(30, 300, 530, 303, fill="#203543", outline="")
    bar = canvas.create_rectangle(30, 300, 30, 303, fill="#57daef", outline="")
    status_file = Path(status_path)

    def poll() -> None:
        try:
            value = json.loads(status_file.read_text(encoding="utf-8"))
            if value.get("close"):
                root.destroy()
                return
            canvas.itemconfigure(status, text=value.get("message", "Loading application"))
            progress = min(1.0, max(0.0, float(value.get("progress", 0))))
            canvas.coords(bar, 30, 300, 30 + progress * 500, 303)
        except (OSError, json.JSONDecodeError):
            if not status_file.parent.exists():
                root.destroy()
                return
        root.after(20, poll)

    root.deiconify()
    root.after(20, poll)
    root.mainloop()


if __name__ == "__main__":
    show_splash(sys.argv[2])
