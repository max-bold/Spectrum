from __future__ import annotations

import base64
import binascii
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path
import sys
from typing import Any
import zlib

import numpy as np

from audioanalysis import ASignal
from spectrum_app.core.model import AxisSpec, GraphData, Measurement, PlotType


MEASUREMENT_EXTENSION = ".bmm"
MEASUREMENT_FORMAT = "bm-spectrum-measurement"
MEASUREMENT_FORMAT_VERSION = 1
AUDIO_ARRAY_ENCODING = "float32-le-zlib-base64"
ARRAY_ENCODING = "raw-le-zlib-base64"
TYPE_KEY = "$bmm"


class MeasurementIOError(RuntimeError):
    pass


def ensure_measurement_extension(path: str | Path) -> Path:
    measurement_path = Path(path)
    if measurement_path.suffix.lower() == MEASUREMENT_EXTENSION:
        return measurement_path
    return measurement_path.with_suffix(MEASUREMENT_EXTENSION)


def save_measurement(measurement: Measurement, path: str | Path) -> Path:
    measurement_path = ensure_measurement_extension(path)
    temporary_path = measurement_path.with_suffix(measurement_path.suffix + ".tmp")
    document = {
        "format": MEASUREMENT_FORMAT,
        "version": MEASUREMENT_FORMAT_VERSION,
        "measurement": _encode_measurement(measurement),
    }

    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        measurement_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(measurement_path)
    except Exception as error:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise MeasurementIOError(f"Cannot export measurement: {error}") from error
    return measurement_path


def load_measurement(path: str | Path) -> Measurement:
    measurement_path = Path(path)
    try:
        document = json.loads(measurement_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("measurement document must be an object")
        if document.get("format") != MEASUREMENT_FORMAT:
            raise ValueError("unsupported measurement format")
        if document.get("version") != MEASUREMENT_FORMAT_VERSION:
            raise ValueError("unsupported measurement format version")
        return _decode_measurement(document.get("measurement"))
    except MeasurementIOError:
        raise
    except Exception as error:
        raise MeasurementIOError(f"Cannot import measurement: {error}") from error


def _encode_measurement(measurement: Measurement) -> dict[str, Any]:
    return {
        "module_id": measurement.module_id,
        "name": measurement.name,
        "module_state": _encode_value(measurement.module_state),
        "graphs": [_encode_graph(graph) for graph in measurement.graphs],
    }


def _decode_measurement(value: object) -> Measurement:
    if not isinstance(value, dict):
        raise ValueError("measurement must be an object")
    module_id = value.get("module_id")
    name = value.get("name")
    graphs_value = value.get("graphs")
    if not isinstance(module_id, str) or not module_id:
        raise ValueError("measurement module_id is invalid")
    if not isinstance(name, str):
        raise ValueError("measurement name is invalid")
    if not isinstance(graphs_value, list):
        raise ValueError("measurement graphs are invalid")
    module_state = _decode_value(value.get("module_state"))
    if not isinstance(module_state, dict) or not all(
        isinstance(key, str) for key in module_state
    ):
        raise ValueError("measurement module_state is invalid")
    return Measurement(
        module_id=module_id,
        name=name,
        module_state=module_state,
        graphs=[_decode_graph(graph) for graph in graphs_value],
    )


def _encode_graph(graph: GraphData) -> dict[str, Any]:
    return {
        "name": graph.name,
        "x": _encode_array(graph.x),
        "y": _encode_array(graph.y),
        "x_axis": graph.x_axis.value,
        "y_axis": graph.y_axis.value,
        "plot_type": getattr(graph, "plot_type", PlotType.LINE).value,
    }


def _decode_graph(value: object) -> GraphData:
    if not isinstance(value, dict):
        raise ValueError("graph must be an object")
    name = value.get("name")
    if not isinstance(name, str):
        raise ValueError("graph name is invalid")
    x = _decode_array(value.get("x"))
    y = _decode_array(value.get("y"))
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("graph series must be equally-sized one-dimensional arrays")
    try:
        x_axis = AxisSpec(value.get("x_axis"))
        y_axis = AxisSpec(value.get("y_axis"))
        plot_type = PlotType(value.get("plot_type", PlotType.LINE.value))
    except (TypeError, ValueError) as error:
        raise ValueError("graph axis is invalid") from error
    return GraphData(
        name=name,
        x=x,
        y=y,
        x_axis=x_axis,
        y_axis=y_axis,
        plot_type=plot_type,
    )


def _encode_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return {
            TYPE_KEY: "enum",
            "class": _type_name(type(value)),
            "value": _encode_value(value.value),
        }
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {TYPE_KEY: "float", "value": repr(value)}
    if isinstance(value, complex):
        return {TYPE_KEY: "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return _encode_value(value.item())
    if isinstance(value, ASignal):
        return {
            TYPE_KEY: "asignal",
            "sample_rate": value.sample_rate,
            "array": _encode_array(
                value.as_array(np.float32),
                encoding=AUDIO_ARRAY_ENCODING,
            ),
        }
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, tuple):
        return {TYPE_KEY: "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("measurement dictionaries must use string keys")
        if TYPE_KEY in value:
            return {
                TYPE_KEY: "dict",
                "items": {key: _encode_value(item) for key, item in value.items()},
            }
        return {key: _encode_value(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            TYPE_KEY: "dataclass",
            "class": _type_name(type(value)),
            "fields": {
                field.name: _encode_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    raise TypeError(f"unsupported measurement value: {type(value).__name__}")


def _decode_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError("invalid measurement value")

    value_type = value.get(TYPE_KEY)
    if value_type is None:
        return {key: _decode_value(item) for key, item in value.items()}
    if value_type == "float":
        text = value.get("value")
        if text not in {"nan", "inf", "-inf"}:
            raise ValueError("invalid non-finite float")
        return float(text)
    if value_type == "complex":
        real = value.get("real")
        imag = value.get("imag")
        if not isinstance(real, (int, float)) or not isinstance(imag, (int, float)):
            raise ValueError("invalid complex value")
        return complex(real, imag)
    if value_type == "asignal":
        sample_rate = value.get("sample_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("invalid ASignal sample rate")
        array = _decode_array(
            value.get("array"),
            expected_encoding=AUDIO_ARRAY_ENCODING,
        )
        if array.ndim != 2 or array.shape[1] < 1:
            raise ValueError("invalid ASignal shape")
        return ASignal(array, sample_rate)
    if value_type == "ndarray":
        return _decode_array(value)
    if value_type == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("invalid tuple value")
        return tuple(_decode_value(item) for item in items)
    if value_type == "dict":
        items = value.get("items")
        if not isinstance(items, dict):
            raise ValueError("invalid dictionary value")
        return {key: _decode_value(item) for key, item in items.items()}
    if value_type == "enum":
        enum_type = _resolve_type(value.get("class"), Enum)
        return enum_type(_decode_value(value.get("value")))
    if value_type == "dataclass":
        dataclass_type = _resolve_dataclass(value.get("class"))
        field_values = value.get("fields")
        if not isinstance(field_values, dict):
            raise ValueError("invalid dataclass fields")
        return dataclass_type(
            **{key: _decode_value(item) for key, item in field_values.items()}
        )
    raise ValueError(f"unsupported measurement value type: {value_type}")


def _encode_array(
    value: np.ndarray,
    *,
    encoding: str = ARRAY_ENCODING,
) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be exported")
    dtype = array.dtype.newbyteorder("<")
    data = np.ascontiguousarray(array, dtype=dtype)
    compressed = zlib.compress(data.tobytes())
    return {
        TYPE_KEY: "ndarray",
        "encoding": encoding,
        "dtype": dtype.str,
        "shape": list(data.shape),
        "data": base64.b64encode(compressed).decode("ascii"),
    }


def _decode_array(
    value: object,
    *,
    expected_encoding: str | None = None,
) -> np.ndarray:
    if not isinstance(value, dict) or value.get(TYPE_KEY) != "ndarray":
        raise ValueError("invalid array")
    encoding = value.get("encoding")
    if encoding not in {ARRAY_ENCODING, AUDIO_ARRAY_ENCODING}:
        raise ValueError("unsupported array encoding")
    if expected_encoding is not None and encoding != expected_encoding:
        raise ValueError("unexpected array encoding")
    shape = value.get("shape")
    dtype_value = value.get("dtype")
    encoded_data = value.get("data")
    if (
        not isinstance(shape, list)
        or not all(isinstance(size, int) and size >= 0 for size in shape)
        or not isinstance(dtype_value, str)
        or not isinstance(encoded_data, str)
    ):
        raise ValueError("invalid array metadata")
    try:
        dtype = np.dtype(dtype_value)
    except TypeError as error:
        raise ValueError("invalid array dtype") from error
    if dtype.hasobject:
        raise ValueError("object array dtype is not supported")
    if expected_encoding == AUDIO_ARRAY_ENCODING and dtype != np.dtype("<f4"):
        raise ValueError("audio array must use float32 little-endian data")
    try:
        raw = zlib.decompress(base64.b64decode(encoded_data.encode("ascii")))
    except (binascii.Error, ValueError, zlib.error) as error:
        raise ValueError("invalid compressed array data") from error
    expected_size = math.prod(shape) * dtype.itemsize
    if len(raw) != expected_size:
        raise ValueError("array data size does not match its shape")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _type_name(value_type: type[Any]) -> str:
    return f"{value_type.__module__}:{value_type.__qualname__}"


def _resolve_type(value: object, base_type: type[Any]) -> type[Any]:
    if not isinstance(value, str) or ":" not in value:
        raise ValueError("invalid stored type name")
    module_name, qualname = value.split(":", 1)
    module = sys.modules.get(module_name)
    if module is None:
        raise ValueError(f"stored type module is not loaded: {module_name}")
    candidate: Any = module
    for part in qualname.split("."):
        if part == "<locals>" or not hasattr(candidate, part):
            raise ValueError(f"stored type is unavailable: {value}")
        candidate = getattr(candidate, part)
    if not isinstance(candidate, type) or not issubclass(candidate, base_type):
        raise ValueError(f"stored type is invalid: {value}")
    return candidate


def _resolve_dataclass(value: object) -> type[Any]:
    candidate = _resolve_type(value, object)
    if not is_dataclass(candidate):
        raise ValueError(f"stored type is not a dataclass: {value}")
    return candidate
