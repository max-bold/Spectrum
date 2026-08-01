from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .imp_measure import (
    CalibrationStage,
    FitResult,
    ImpedanceProjectData,
    MeasurementConfig,
    MeasurementState,
    PhaseDisplayMode,
    SpiceTableValues,
    WindowFunction,
    validate_impedance_project_data,
)

PROJECT_FORMAT = "bm-impedance-project"
PROJECT_VERSION = 1
PROJECT_EXTENSION = ".bmi"
MANIFEST_ARRAY = "manifest"


def ensure_project_extension(path: str | Path) -> Path:
    result = Path(path)
    if result.suffix.lower() == PROJECT_EXTENSION:
        return result
    return result.with_suffix(PROJECT_EXTENSION)


def save_impedance_project(
    path: str | Path,
    project: ImpedanceProjectData,
) -> Path:
    validate_impedance_project_data(project)
    target = ensure_project_extension(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    array_names: list[str] = []

    def add_array(name: str, value: np.ndarray | None) -> None:
        if value is None:
            return
        arrays[name] = np.ascontiguousarray(value)
        array_names.append(name)

    add_array(
        "channel_calibration_recording",
        project.channel_calibration_recording,
    )
    add_array("channel_calibration_signal", project.channel_calibration_signal)
    add_array("calibration_recording", project.calibration_recording)
    add_array("calibration_signal", project.calibration_signal)
    add_array("measurement_recording", project.measurement_recording)
    add_array("measurement_signal", project.measurement_signal)
    add_array("calibration_frequency", project.calibration_frequency)
    add_array("calibration_impedance", project.calibration_impedance)
    add_array("calibration_phase", project.calibration_phase)
    add_array(
        "calibration_phase_derivative",
        project.calibration_phase_derivative,
    )
    add_array("channel_correction", project.channel_correction)
    add_array("frequency", project.frequency)
    add_array("impedance", project.impedance)
    add_array("phase", project.phase)
    add_array("phase_derivative", project.phase_derivative)
    if project.fit_result is not None:
        add_array("spice_physical_params", project.fit_result.physical_params)

    manifest = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "state": project.state.value,
        "calibration_stage": project.calibration_stage.value,
        "phase_mode": project.phase_mode.value,
        "calibration_config": _config_to_dict(project.calibration_config),
        "result_config": (
            None
            if project.result_config is None
            else _config_to_dict(project.result_config)
        ),
        "sample_rates": {
            "channel_calibration_recording": (
                project.channel_calibration_recording_sample_rate
            ),
            "channel_calibration_signal": (
                project.channel_calibration_signal_sample_rate
            ),
            "calibration_recording": project.calibration_recording_sample_rate,
            "calibration_signal": project.calibration_signal_sample_rate,
            "measurement_recording": project.measurement_recording_sample_rate,
            "measurement_signal": project.measurement_signal_sample_rate,
        },
        "reference_resistor_estimated": project.reference_resistor_estimated,
        "reference_diagnostics": project.reference_diagnostics,
        "spice_fit": _fit_to_dict(project.fit_result),
        "spice_values": _spice_values_to_dict(project.spice_values),
        "arrays": array_names,
    }
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    arrays[MANIFEST_ARRAY] = np.frombuffer(manifest_bytes, dtype=np.uint8)

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target


def load_impedance_project(path: str | Path) -> ImpedanceProjectData:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if MANIFEST_ARRAY not in archive.files:
                raise ValueError("BMI project has no manifest")
            manifest = _decode_manifest(archive[MANIFEST_ARRAY])
            _validate_manifest(manifest)
            declared_arrays = manifest["arrays"]
            if not isinstance(declared_arrays, list) or not all(
                isinstance(name, str) for name in declared_arrays
            ):
                raise ValueError("BMI project has an invalid array index")
            missing = set(declared_arrays) - set(archive.files)
            if missing:
                raise ValueError(
                    "BMI project is missing arrays: " + ", ".join(sorted(missing))
                )
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in declared_arrays
            }
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"Cannot read BMI project: {exc}") from exc

    sample_rates = manifest.get("sample_rates")
    if not isinstance(sample_rates, dict):
        raise ValueError("BMI project has invalid sample rates")
    fit_result = _fit_from_dict(manifest.get("spice_fit"), arrays)
    project = ImpedanceProjectData(
        state=MeasurementState(manifest["state"]),
        calibration_stage=CalibrationStage(manifest["calibration_stage"]),
        phase_mode=PhaseDisplayMode(manifest["phase_mode"]),
        calibration_config=_config_from_dict(manifest["calibration_config"]),
        result_config=(
            None
            if manifest.get("result_config") is None
            else _config_from_dict(manifest["result_config"])
        ),
        channel_calibration_recording=_required_array(
            arrays,
            "channel_calibration_recording",
        ),
        channel_calibration_recording_sample_rate=_required_rate(
            sample_rates,
            "channel_calibration_recording",
        ),
        channel_calibration_signal=_required_array(
            arrays,
            "channel_calibration_signal",
        ),
        channel_calibration_signal_sample_rate=_required_rate(
            sample_rates,
            "channel_calibration_signal",
        ),
        calibration_recording=arrays.get("calibration_recording"),
        calibration_recording_sample_rate=_optional_rate(
            sample_rates,
            "calibration_recording",
        ),
        calibration_signal=arrays.get("calibration_signal"),
        calibration_signal_sample_rate=_optional_rate(
            sample_rates,
            "calibration_signal",
        ),
        measurement_recording=arrays.get("measurement_recording"),
        measurement_recording_sample_rate=_optional_rate(
            sample_rates,
            "measurement_recording",
        ),
        measurement_signal=arrays.get("measurement_signal"),
        measurement_signal_sample_rate=_optional_rate(
            sample_rates,
            "measurement_signal",
        ),
        calibration_frequency=arrays.get("calibration_frequency"),
        calibration_impedance=arrays.get("calibration_impedance"),
        calibration_phase=arrays.get("calibration_phase"),
        calibration_phase_derivative=arrays.get(
            "calibration_phase_derivative"
        ),
        channel_correction=_required_array(arrays, "channel_correction"),
        reference_resistor_estimated=_optional_float(
            manifest.get("reference_resistor_estimated")
        ),
        reference_diagnostics=manifest.get("reference_diagnostics"),
        frequency=arrays.get("frequency"),
        impedance=arrays.get("impedance"),
        phase=arrays.get("phase"),
        phase_derivative=arrays.get("phase_derivative"),
        fit_result=fit_result,
        spice_values=_spice_values_from_dict(manifest.get("spice_values")),
    )
    validate_impedance_project_data(project)
    return project


def _config_to_dict(config: MeasurementConfig) -> dict[str, object]:
    result = asdict(config)
    result["window_function"] = config.window_function.value
    return result


def _config_from_dict(value: object) -> MeasurementConfig:
    if not isinstance(value, dict):
        raise ValueError("BMI project has invalid measurement settings")
    fields = dict(value)
    try:
        fields["window_function"] = WindowFunction(fields["window_function"])
        config = MeasurementConfig(**fields)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BMI project has invalid measurement settings") from exc
    config.validate()
    return config


def _fit_to_dict(result: FitResult | None) -> dict[str, object] | None:
    if result is None:
        return None
    return {
        "sections": result.sections,
        "rms_log_error": result.rms_log_error,
        "max_abs_log_error": result.max_abs_log_error,
        "selection_score": (
            result.selection_score
            if math.isfinite(result.selection_score)
            else None
        ),
    }


def _fit_from_dict(
    value: object,
    arrays: dict[str, np.ndarray],
) -> FitResult | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("BMI project has invalid SPICE fit results")
    try:
        selection_score = value.get("selection_score")
        return FitResult(
            sections=int(value["sections"]),
            physical_params=_required_array(
                arrays,
                "spice_physical_params",
            ),
            rms_log_error=float(value["rms_log_error"]),
            max_abs_log_error=float(value["max_abs_log_error"]),
            selection_score=(
                math.nan
                if selection_score is None
                else float(selection_score)
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BMI project has invalid SPICE fit results") from exc


def _spice_values_to_dict(
    values: SpiceTableValues | None,
) -> dict[str, object] | None:
    if values is None:
        return None
    return {
        "l1": values.l1,
        "sections": [list(section) for section in values.sections],
        "r1": values.r1,
    }


def _spice_values_from_dict(value: object) -> SpiceTableValues | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("sections"), list):
        raise ValueError("BMI project has invalid SPICE table values")
    try:
        sections = tuple(
            tuple(str(item) for item in section)
            for section in value["sections"]
        )
        if any(len(section) != 3 for section in sections):
            raise ValueError
        return SpiceTableValues(
            l1=str(value["l1"]),
            sections=sections,
            r1=str(value["r1"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BMI project has invalid SPICE table values") from exc


def _decode_manifest(value: np.ndarray) -> dict[str, object]:
    data = np.asarray(value)
    if data.dtype != np.uint8 or data.ndim != 1:
        raise ValueError("BMI project has an invalid manifest encoding")
    try:
        manifest = json.loads(data.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("BMI project has an invalid manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("BMI project has an invalid manifest")
    return manifest


def _validate_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("format") != PROJECT_FORMAT:
        raise ValueError("unsupported BMI project format")
    if manifest.get("version") != PROJECT_VERSION:
        raise ValueError("unsupported BMI project version")
    for key in (
        "state",
        "calibration_stage",
        "phase_mode",
        "calibration_config",
        "arrays",
    ):
        if key not in manifest:
            raise ValueError(f"BMI project manifest has no {key}")


def _required_array(
    arrays: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"BMI project has no {name}")
    return arrays[name]


def _required_rate(value: dict[str, object], name: str) -> int:
    rate = _optional_rate(value, name)
    if rate is None:
        raise ValueError(f"BMI project has no {name} sample rate")
    return rate


def _optional_rate(value: dict[str, object], name: str) -> int | None:
    rate = value.get(name)
    if rate is None:
        return None
    try:
        result = int(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"BMI project has invalid {name} sample rate") from exc
    if result <= 0:
        raise ValueError(f"BMI project has invalid {name} sample rate")
    return result


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("BMI project has an invalid numeric value") from exc
    if not math.isfinite(result):
        raise ValueError("BMI project has an invalid numeric value")
    return result
