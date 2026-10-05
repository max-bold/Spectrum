from pathlib import Path


class NativeFileDialogError(RuntimeError):
    pass


def choose_file(
    *, title: str, extension: str, description: str,
    save: bool = False, initial: Path | None = None,
) -> Path | None:
    """Run a native Tk dialog on the UI thread and release its hidden owner."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise NativeFileDialogError("System file dialogs require tkinter (Tcl/Tk)") from error

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        options = dict(
            parent=root, title=title, defaultextension=extension,
            filetypes=[(description, f"*{extension}"), ("All files", "*.*")],
        )
        if initial is not None:
            options["initialdir"] = str(initial.parent)
            if save:
                options["initialfile"] = initial.name
        dialog = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        value = dialog(**options)
        return Path(value) if value else None
    except tk.TclError as error:
        raise NativeFileDialogError(f"Cannot open system file dialog: {error}") from error
    finally:
        if root is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
