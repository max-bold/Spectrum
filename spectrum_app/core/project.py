from pathlib import Path
import pickle

from spectrum_app.core.model import AppState, Measurement


PROJECT_EXTENSION = ".bms"


class ProjectError(RuntimeError):
    pass


def ensure_project_extension(path: str | Path) -> Path:
    project_path = Path(path)
    if project_path.suffix.lower() == PROJECT_EXTENSION:
        return project_path
    return project_path.with_suffix(PROJECT_EXTENSION)


def save_project(state: AppState, path: str | Path) -> Path:
    project_path = ensure_project_extension(path)
    temporary_path = project_path.with_suffix(project_path.suffix + ".tmp")
    previous_path = state.project_path
    state.project_path = project_path

    try:
        payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        project_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_bytes(payload)
        temporary_path.replace(project_path)
    except Exception as error:
        state.project_path = previous_path
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ProjectError(f"Cannot save project: {error}") from error
    return project_path


def load_project(path: str | Path) -> AppState:
    project_path = Path(path)
    try:
        state = pickle.loads(project_path.read_bytes())
    except Exception as error:
        raise ProjectError(f"Cannot load project: {error}") from error

    if not isinstance(state, AppState):
        raise ProjectError("Project does not contain AppState")
    if not all(isinstance(item, Measurement) for item in state.measurements):
        raise ProjectError("Project contains an invalid measurement")

    state.project_path = project_path
    return state
