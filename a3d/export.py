# FILE: a3d/export.py
from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import numpy as np


def _export_columns(rows: List[Dict[str, Any]]) -> list[str]:
    """Preserve the union of every evidence record's fields."""
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError("all exported records must be dictionaries")
    if not all(isinstance(key, str) for row in rows for key in row):
        raise TypeError("scientific record column names must be strings")
    return sorted({key for row in rows for key in row})


def export_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Atomically export *all* columns, including those absent from row one."""
    import os
    import tempfile
    from pathlib import Path

    columns = _export_columns(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="",
            prefix=".aegis-csv-", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temp = Path(handle.name)
            if rows:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def export_parquet(path: str, rows: List[Dict[str, Any]]) -> None:
    import os
    import tempfile
    from pathlib import Path

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow not available") from exc

    columns = _export_columns(rows)
    table = pa.Table.from_pydict({
        key: [row.get(key) for row in rows] for key in columns
    })
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".aegis-parquet-", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as handle:
            temp = Path(handle.name)
        pq.write_table(table, str(temp))
        # Windows fsync requires a writable file descriptor.
        with temp.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def export_fixed_point_header(
    path: str,
    arrays: Dict[str, np.ndarray],
    scales: Dict[str, float],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically export verified int16 weights as deterministic, legal C code.

    All scales are strictly positive and finite. Invalid names, non-finite
    tensors and metadata injection must fail *before* replacing an accepted
    header. Quantization uses rounded and saturated signed int16 values.
    """
    import math
    import os
    import re
    import tempfile
    from pathlib import Path

    identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
    prepared: list[tuple[str, float, np.ndarray]] = []
    if not arrays:
        raise ValueError("cannot export an empty fixed-point artifact")
    for name, values in arrays.items():
        if not isinstance(name, str) or not identifier.fullmatch(name):
            raise ValueError(f"invalid C array identifier: {name!r}")
        if name not in scales:
            raise ValueError(f"missing quantization scale for {name}")
        scale = float(scales[name])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale for {name} must be finite and positive")
        data = np.asarray(values, dtype=np.float64)
        if data.ndim == 0 or data.size == 0:
            raise ValueError(f"array {name} must have a non-empty dimension")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"array {name} contains nonfinite weights")
        with np.errstate(over="ignore"):
            raw = np.rint(data / scale)
        quantized = np.clip(raw, -32768, 32767).astype(np.int16)
        prepared.append((name, scale, quantized))

    comments: list[tuple[str, str]] = []
    for key, value in (meta or {}).items():
        tag, text = str(key), str(value)
        if any(token in part for part in (tag, text)
               for token in ("*/", "\n", "\r")):
            raise ValueError("metadata contains unsafe C comment content")
        comments.append((tag, text))

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=".aegis-export-", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write("/* Aegis validated fixed-point export */\n")
            output.write("#pragma once\n#include <stdint.h>\n\n")
            for key, value in comments:
                output.write(f"/* {key}: {value} */\n")
            for name, scale, values in prepared:
                macro = name.upper()
                output.write(f"#define {macro}_SCALE {scale:.12g}\n")
                for axis, length in enumerate(values.shape):
                    output.write(f"#define {macro}_DIM{axis} {int(length)}\n")
                output.write(f"static const int16_t {name}[] = {{\n")
                for value in values.flat:
                    output.write(f"  {int(value)},\n")
                output.write("};\n\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
