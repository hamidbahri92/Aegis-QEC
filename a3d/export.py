# FILE: a3d/export.py
from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional

import numpy as np


def export_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            f.write("")
        return
    headers = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_parquet(path: str, rows: List[Dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow not available") from e
    if not rows:
        table = pa.Table.from_pydict({})
    else:
        keys = sorted(rows[0].keys())
        cols = {k: [r.get(k, None) for r in rows] for k in keys}
        table = pa.Table.from_pydict(cols)
    pq.write_table(table, path)


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
