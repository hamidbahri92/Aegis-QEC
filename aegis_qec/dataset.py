from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_DATASET_FORMAT = "aegis-qec-dataset-v1"


def _require_dependencies():
    try:
        import h5py
        import stim
    except ImportError as exc:
        raise RuntimeError(
            "Aegis dataset support requires h5py and Stim. Install with: "
            "python -m pip install -U 'aegis-qec[full]'"
        ) from exc
    return h5py, stim


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _build_circuit(
    *,
    circuit_path: str | None,
    distance: int,
    physical_error_rate: float,
    basis: str,
    rounds: int | None,
):
    _, stim = _require_dependencies()
    if circuit_path:
        return stim.Circuit.from_file(circuit_path), {
            "source": "stim-file",
            "circuit_path": str(Path(circuit_path).resolve()),
        }

    basis = str(basis).lower()
    if basis not in {"x", "z"}:
        raise ValueError("basis must be 'x' or 'z'")
    distance = int(distance)
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    p = float(physical_error_rate)
    if not 0.0 <= p < 0.5:
        raise ValueError("physical_error_rate must satisfy 0 <= p < 0.5")
    point_rounds = distance if rounds is None else int(rounds)
    if point_rounds < 1:
        raise ValueError("rounds must be positive")

    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=point_rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    return circuit, {
        "source": "generated",
        "code": "rotated_surface_code_memory",
        "distance": distance,
        "rounds": point_rounds,
        "basis": basis,
        "physical_error_rate": p,
    }


def extract_dem_mechanisms(dem: Any) -> dict[str, np.ndarray]:
    """Extract sparse detector/observable incidence for DEM error mechanisms.

    Repeat blocks and detector shifts are flattened first. Separator targets are
    decomposition hints, not independent physical mechanisms, so the exported
    incidence records the XOR effect of the complete error instruction.
    """

    probabilities: list[float] = []
    detector_indices: list[int] = []
    detector_indptr = [0]
    observable_indices: list[int] = []
    observable_indptr = [0]

    for instruction in dem.flattened():
        if getattr(instruction, "type", None) != "error":
            continue

        args = instruction.args_copy()
        if len(args) != 1:
            raise ValueError("DEM error instruction did not have one probability")
        probability = float(args[0])
        detectors: set[int] = set()
        observables: set[int] = set()

        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detectors ^= {int(target.val)}
            elif target.is_logical_observable_id():
                observables ^= {int(target.val)}
            elif target.is_separator():
                continue
            else:
                raise ValueError(f"Unsupported DEM error target: {target!r}")

        probabilities.append(probability)
        detector_indices.extend(sorted(detectors))
        detector_indptr.append(len(detector_indices))
        observable_indices.extend(sorted(observables))
        observable_indptr.append(len(observable_indices))

    return {
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "detector_indices": np.asarray(detector_indices, dtype=np.int64),
        "detector_indptr": np.asarray(detector_indptr, dtype=np.int64),
        "observable_indices": np.asarray(observable_indices, dtype=np.int64),
        "observable_indptr": np.asarray(observable_indptr, dtype=np.int64),
    }


def _dense_mechanism_matrices(
    mechanisms: dict[str, np.ndarray],
    *,
    num_detectors: int,
    num_observables: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = mechanisms["probabilities"]
    check_matrix = np.zeros(
        (int(num_detectors), len(probabilities)),
        dtype=np.uint8,
    )
    obs_matrix = np.zeros(
        (int(num_observables), len(probabilities)),
        dtype=np.uint8,
    )

    detector_indices = mechanisms["detector_indices"]
    detector_indptr = mechanisms["detector_indptr"]
    observable_indices = mechanisms["observable_indices"]
    observable_indptr = mechanisms["observable_indptr"]

    for mechanism_index in range(len(probabilities)):
        dets = detector_indices[
            detector_indptr[mechanism_index] : detector_indptr[mechanism_index + 1]
        ]
        obs = observable_indices[
            observable_indptr[mechanism_index] : observable_indptr[mechanism_index + 1]
        ]
        check_matrix[dets, mechanism_index] = 1
        obs_matrix[obs, mechanism_index] = 1

    return check_matrix, obs_matrix


def _hash_bool_dataset(dataset, *, rows_per_chunk: int = 8192) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{dataset.shape[0]}x{dataset.shape[1]}|little-bitpack|".encode("ascii")
    )
    for start in range(0, dataset.shape[0], rows_per_chunk):
        stop = min(dataset.shape[0], start + rows_per_chunk)
        values = np.asarray(dataset[start:stop], dtype=np.bool_)
        packed = np.packbits(values, axis=1, bitorder="little")
        digest.update(packed.tobytes(order="C"))
    return digest.hexdigest()


def _hash_uint8_dataset(dataset, *, rows_per_chunk: int = 65536) -> str:
    digest = hashlib.sha256()
    digest.update(f"{dataset.shape[0]}|uint8|".encode("ascii"))
    for start in range(0, dataset.shape[0], rows_per_chunk):
        stop = min(dataset.shape[0], start + rows_per_chunk)
        values = np.asarray(dataset[start:stop], dtype=np.uint8)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()



def _chunk_sha256(
    syndromes: np.ndarray,
    observables: np.ndarray,
    splits: np.ndarray,
) -> str:
    """Hash a durable shot chunk including array boundaries and split membership."""
    digest = hashlib.sha256()
    for label, values in (
        ("syndromes", syndromes),
        ("observables", observables),
        ("split", splits),
    ):
        data = np.asarray(values, dtype=np.uint8)
        digest.update(
            f"{label}|{','.join(str(v) for v in data.shape)}|uint8|".encode("ascii")
        )
        digest.update(data.tobytes(order="C"))
    return digest.hexdigest()


def _verify_chunk_journal(handle, *, committed: int, chunk_size: int) -> bool:
    """Reject changed committed rows before resuming any incomplete dataset."""
    if "chunk_sha256" not in handle:
        return False
    expected_count = (committed + chunk_size - 1) // chunk_size
    journal = handle["chunk_sha256"]
    if int(journal.shape[0]) < expected_count:
        raise ValueError("incomplete dataset commit journal is truncated")
    for index in range(expected_count):
        start = index * chunk_size
        stop = min(committed, start + chunk_size)
        expected = journal[index]
        if isinstance(expected, bytes):
            expected = expected.decode("ascii")
        actual = _chunk_sha256(
            handle["syndromes"][start:stop],
            handle["observables"][start:stop],
            handle["split"][start:stop],
        )
        if actual != expected:
            raise ValueError(
                f"incomplete dataset committed chunk integrity mismatch: {index}"
            )
    return True


def _replay_legacy_chunks(
    handle, circuit, *,
    committed: int,
    chunk_size: int,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    """Qualify old resumable datasets before adding missing durable hashes."""
    hashes = []
    for index, start in enumerate(range(0, committed, chunk_size)):
        count = min(chunk_size, committed - start)
        sampler = circuit.compile_detector_sampler(
            seed=(int(seed) + index) & ((1 << 64) - 1)
        )
        det, obs = sampler.sample(
            shots=count, separate_observables=True, bit_packed=False
        )
        split = _split_labels(
            count, seed=seed, chunk_index=index,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        if (
            not np.array_equal(det, handle["syndromes"][start:start + count])
            or not np.array_equal(obs, handle["observables"][start:start + count])
            or not np.array_equal(split, handle["split"][start:start + count])
        ):
            raise ValueError(
                f"legacy dataset committed chunk replay mismatch: {index}"
            )
        hashes.append(_chunk_sha256(det, obs, split))
    journal = handle.create_dataset(
        "chunk_sha256",
        shape=(len(hashes),),
        maxshape=(None,),
        chunks=True,
        dtype="S64",
    )
    if hashes:
        journal[:] = np.asarray(hashes, dtype="S64")
    handle.flush()


def _count_split_labels(
    dataset,
    *,
    rows_per_chunk: int = 1_000_000,
) -> dict[str, int]:
    counts = np.zeros(3, dtype=np.int64)
    for start in range(0, dataset.shape[0], rows_per_chunk):
        stop = min(dataset.shape[0], start + rows_per_chunk)
        values = np.asarray(dataset[start:stop], dtype=np.uint8)
        chunk_counts = np.bincount(values, minlength=3)
        if len(chunk_counts) > 3:
            raise ValueError("dataset split contains an unknown label")
        counts += chunk_counts[:3]
    return {
        "train": int(counts[0]),
        "validation": int(counts[1]),
        "test": int(counts[2]),
    }


def _split_labels(
    count: int,
    *,
    seed: int,
    chunk_index: int,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    mixed_seed = (
        int(seed)
        ^ 0xD1B54A32D192ED03
        ^ (int(chunk_index) * 0x9E3779B97F4A7C15)
    ) & ((1 << 64) - 1)
    rng = np.random.default_rng(mixed_seed)
    values = rng.random(int(count))
    result = np.full(int(count), 2, dtype=np.uint8)
    result[values < train_fraction + validation_fraction] = 1
    result[values < train_fraction] = 0
    return result


def _validate_split_fractions(
    train_fraction: float,
    validation_fraction: float,
) -> None:
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction > 1.0:
        raise ValueError(
            "train_fraction + validation_fraction must not exceed 1"
        )


def _dataset_identity(
    *,
    circuit_sha256: str,
    raw_dem_sha256: str,
    shots: int,
    seed: int,
    chunk_size: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    value = {
        "format": _DATASET_FORMAT,
        "circuit_sha256": circuit_sha256,
        "raw_dem_sha256": raw_dem_sha256,
        "shots": int(shots),
        "seed": int(seed),
        "chunk_size": int(chunk_size),
        "train_fraction": float(train_fraction),
        "validation_fraction": float(validation_fraction),
    }
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def generate_qec_dataset(
    output_path: str,
    *,
    shots: int = 100_000,
    seed: int = 1234,
    chunk_size: int = 10_000,
    max_chunks_per_run: int | None = None,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    circuit_path: str | None = None,
    distance: int = 5,
    physical_error_rate: float = 0.006,
    basis: str = "x",
    rounds: int | None = None,
    dense_matrix_max_cells: int = 20_000_000,
) -> dict[str, Any]:
    """Generate or resume a provenance-rich HDF5 decoder dataset."""

    h5py, _ = _require_dependencies()
    shots = int(shots)
    chunk_size = int(chunk_size)
    if shots < 1:
        raise ValueError("shots must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_chunks_per_run is not None and int(max_chunks_per_run) < 1:
        raise ValueError("max_chunks_per_run must be positive")
    _validate_split_fractions(train_fraction, validation_fraction)

    circuit, source_metadata = _build_circuit(
        circuit_path=circuit_path,
        distance=distance,
        physical_error_rate=physical_error_rate,
        basis=basis,
        rounds=rounds,
    )
    if circuit.num_detectors < 1:
        raise ValueError("dataset circuits must contain at least one detector")
    if circuit.num_observables < 1:
        raise ValueError("dataset circuits must contain at least one observable")

    raw_dem = circuit.detector_error_model(
        decompose_errors=False,
        flatten_loops=True,
    )
    try:
        decoding_dem = circuit.detector_error_model(
            decompose_errors=True,
            flatten_loops=True,
        )
        decoding_dem_text: str | None = str(decoding_dem)
    except ValueError:
        decoding_dem_text = None

    circuit_text = str(circuit)
    raw_dem_text = str(raw_dem)
    circuit_sha256 = _sha256_bytes(circuit_text.encode("utf-8"))
    raw_dem_sha256 = _sha256_bytes(raw_dem_text.encode("utf-8"))
    decoding_dem_sha256 = (
        _sha256_bytes(decoding_dem_text.encode("utf-8"))
        if decoding_dem_text is not None
        else None
    )

    mechanisms = extract_dem_mechanisms(raw_dem)
    identity = _dataset_identity(
        circuit_sha256=circuit_sha256,
        raw_dem_sha256=raw_dem_sha256,
        shots=shots,
        seed=seed,
        chunk_size=chunk_size,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with h5py.File(path, "a") as handle:
        if is_new:
            handle.attrs["format"] = _DATASET_FORMAT
            handle.attrs["schema_version"] = 1
            handle.attrs["identity_sha256"] = identity
            handle.attrs["target_shots"] = shots
            handle.attrs["seed"] = int(seed)
            handle.attrs["chunk_size"] = chunk_size
            handle.attrs["train_fraction"] = float(train_fraction)
            handle.attrs["validation_fraction"] = float(validation_fraction)
            handle.attrs["test_fraction"] = float(
                1.0 - train_fraction - validation_fraction
            )
            handle.attrs["complete"] = False
            handle.create_dataset(
                "chunk_sha256",
                shape=(0,),
                maxshape=(None,),
                chunks=True,
                dtype="S64",
            )
            handle.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["circuit_sha256"] = circuit_sha256
            handle.attrs["raw_dem_sha256"] = raw_dem_sha256
            handle.attrs["decoding_dem_sha256"] = decoding_dem_sha256 or ""

            row_chunk = min(chunk_size, shots)
            handle.create_dataset(
                "syndromes",
                shape=(0, int(circuit.num_detectors)),
                maxshape=(shots, int(circuit.num_detectors)),
                chunks=(row_chunk, int(circuit.num_detectors)),
                dtype=np.bool_,
                compression="gzip",
                shuffle=True,
            )
            handle.create_dataset(
                "observables",
                shape=(0, int(circuit.num_observables)),
                maxshape=(shots, int(circuit.num_observables)),
                chunks=(row_chunk, int(circuit.num_observables)),
                dtype=np.bool_,
                compression="gzip",
                shuffle=True,
            )
            handle.create_dataset(
                "split",
                shape=(0,),
                maxshape=(shots,),
                chunks=(row_chunk,),
                dtype=np.uint8,
                compression="gzip",
                shuffle=True,
            )

            string_dtype = h5py.string_dtype(encoding="utf-8")
            handle.create_dataset("circuit", data=circuit_text, dtype=string_dtype)
            handle.create_dataset(
                "detector_error_model",
                data=raw_dem_text,
                dtype=string_dtype,
            )
            handle.create_dataset(
                "decoding_detector_error_model",
                data=decoding_dem_text or "",
                dtype=string_dtype,
            )
            metadata = {
                "source": source_metadata,
                "format": _DATASET_FORMAT,
                "num_detectors": int(circuit.num_detectors),
                "num_observables": int(circuit.num_observables),
                "num_error_mechanisms": int(len(mechanisms["probabilities"])),
                "split_labels": {
                    "0": "train",
                    "1": "validation",
                    "2": "test",
                },
            }
            handle.create_dataset(
                "metadata_json",
                data=json.dumps(metadata, sort_keys=True),
                dtype=string_dtype,
            )

            mechanism_group = handle.create_group("error_mechanisms")
            for name, values in mechanisms.items():
                mechanism_group.create_dataset(
                    name,
                    data=values,
                    compression="gzip",
                    shuffle=True,
                )

            handle.create_dataset(
                "priors",
                data=mechanisms["probabilities"],
                compression="gzip",
                shuffle=True,
            )
            cells = int(circuit.num_detectors) * len(
                mechanisms["probabilities"]
            )
            obs_cells = int(circuit.num_observables) * len(
                mechanisms["probabilities"]
            )
            if (
                int(dense_matrix_max_cells) > 0
                and cells <= int(dense_matrix_max_cells)
                and obs_cells <= int(dense_matrix_max_cells)
            ):
                check_matrix, obs_matrix = _dense_mechanism_matrices(
                    mechanisms,
                    num_detectors=int(circuit.num_detectors),
                    num_observables=int(circuit.num_observables),
                )
                handle.create_dataset(
                    "check_matrix",
                    data=check_matrix,
                    compression="gzip",
                    shuffle=True,
                )
                handle.create_dataset(
                    "obs_matrix",
                    data=obs_matrix,
                    compression="gzip",
                    shuffle=True,
                )
                handle.attrs["dense_mechanism_matrices"] = True
            else:
                handle.attrs["dense_mechanism_matrices"] = False
            handle.flush()
        else:
            if handle.attrs.get("format") != _DATASET_FORMAT:
                raise ValueError(f"{path} is not an Aegis QEC dataset")
            if handle.attrs.get("identity_sha256") != identity:
                raise ValueError(
                    "dataset configuration does not match the existing file; "
                    "use the original parameters or a new output path"
                )

        syndromes = handle["syndromes"]
        observables = handle["observables"]
        split = handle["split"]
        written = int(syndromes.shape[0])

        if not is_new and bool(handle.attrs.get("complete", False)):
            # Never re-sign already-complete data by overwriting the stored
            # digests. A corrupted HDF5 file must stay corrupt on re-entry.
            if (
                written != shots
                or int(observables.shape[0]) != written
                or int(split.shape[0]) != written
                or int(handle.attrs.get("written_shots", -1)) != written
            ):
                raise ValueError("completed dataset row counts changed")
            for name, dataset, hasher in (
                ("syndromes_sha256", syndromes, _hash_bool_dataset),
                ("observables_sha256", observables, _hash_bool_dataset),
                ("split_sha256", split, _hash_uint8_dataset),
            ):
                if str(handle.attrs.get(name, "")) != hasher(dataset):
                    raise ValueError(
                        f"completed dataset integrity mismatch: {name}"
                    )

        if not is_new and not bool(handle.attrs.get("complete", False)):
            # The written_shots attribute is the durable commit boundary:
            # discard any partially appended rows after an interrupted chunk.
            committed = int(handle.attrs.get("written_shots", 0))
            lengths = (
                written, int(observables.shape[0]), int(split.shape[0])
            )
            if committed < 0 or committed > shots:
                raise RuntimeError("invalid incomplete dataset commit boundary")
            if any(length < committed for length in lengths):
                raise RuntimeError(
                    "committed dataset rows are missing; refusing to resume"
                )
            if committed % chunk_size and committed != shots:
                raise RuntimeError(
                    "incomplete dataset commit boundary is not a full chunk"
                )
            journal_present = _verify_chunk_journal(
                handle, committed=committed, chunk_size=chunk_size
            )
            if not journal_present:
                _replay_legacy_chunks(
                    handle, circuit, committed=committed,
                    chunk_size=chunk_size, seed=seed,
                    train_fraction=train_fraction,
                    validation_fraction=validation_fraction,
                )
            if any(length != committed for length in lengths):
                syndromes.resize((committed, syndromes.shape[1]))
                observables.resize((committed, observables.shape[1]))
                split.resize((committed,))
            journal = handle["chunk_sha256"]
            expected_chunks = (committed + chunk_size - 1) // chunk_size
            if int(journal.shape[0]) > expected_chunks:
                journal.resize((expected_chunks,))
            handle.attrs["written_shots"] = committed
            handle.flush()
            written = committed

        if int(observables.shape[0]) != written or int(split.shape[0]) != written:
            raise RuntimeError("dataset row counts disagree; refusing to resume")
        if written > shots:
            raise RuntimeError("dataset contains more rows than its target")

        chunks_written = 0
        while written < shots:
            if (
                max_chunks_per_run is not None
                and chunks_written >= int(max_chunks_per_run)
            ):
                break

            count = min(chunk_size, shots - written)
            chunk_index = written // chunk_size
            chunk_seed = (int(seed) + int(chunk_index)) & ((1 << 64) - 1)
            sampler = circuit.compile_detector_sampler(seed=chunk_seed)
            detector_samples, observable_samples = sampler.sample(
                shots=count,
                separate_observables=True,
                bit_packed=False,
            )
            detector_samples = np.asarray(
                detector_samples,
                dtype=np.bool_,
            )
            observable_samples = np.asarray(
                observable_samples,
                dtype=np.bool_,
            )
            split_values = _split_labels(
                count,
                seed=seed,
                chunk_index=chunk_index,
                train_fraction=float(train_fraction),
                validation_fraction=float(validation_fraction),
            )

            stop = written + count
            syndromes.resize((stop, syndromes.shape[1]))
            observables.resize((stop, observables.shape[1]))
            split.resize((stop,))
            syndromes[written:stop] = detector_samples
            observables[written:stop] = observable_samples
            split[written:stop] = split_values
            chunk_digest = _chunk_sha256(
                detector_samples, observable_samples, split_values
            )
            journal = handle["chunk_sha256"]
            journal.resize((chunk_index + 1,))
            journal[chunk_index] = chunk_digest.encode("ascii")
            # Flush the data and hash before advancing the committed boundary.
            handle.flush()
            handle.attrs["written_shots"] = stop
            handle.attrs["last_completed_chunk"] = int(chunk_index)
            handle.flush()

            written = stop
            chunks_written += 1

        complete = written == shots
        handle.attrs["complete"] = complete
        handle.attrs["written_shots"] = written
        if complete:
            syndrome_hash = _hash_bool_dataset(syndromes)
            observable_hash = _hash_bool_dataset(observables)
            split_hash = _hash_uint8_dataset(split)
            handle.attrs["syndromes_sha256"] = syndrome_hash
            handle.attrs["observables_sha256"] = observable_hash
            handle.attrs["split_sha256"] = split_hash
            handle.attrs["completed_utc"] = datetime.now(timezone.utc).isoformat()
        else:
            syndrome_hash = None
            observable_hash = None
            split_hash = None
        handle.flush()

        report = {
            "schema_version": 1,
            "format": _DATASET_FORMAT,
            "path": str(path),
            "complete": bool(complete),
            "target_shots": shots,
            "written_shots": written,
            "new_chunks": chunks_written,
            "num_detectors": int(circuit.num_detectors),
            "num_observables": int(circuit.num_observables),
            "num_error_mechanisms": int(len(mechanisms["probabilities"])),
            "identity_sha256": identity,
            "circuit_sha256": circuit_sha256,
            "raw_dem_sha256": raw_dem_sha256,
            "decoding_dem_sha256": decoding_dem_sha256,
            "syndromes_sha256": syndrome_hash,
            "observables_sha256": observable_hash,
            "split_sha256": split_hash,
            "dense_mechanism_matrices": bool(
                handle.attrs["dense_mechanism_matrices"]
            ),
        }

    report["file_sha256"] = _sha256_file(path)
    report["bytes"] = int(path.stat().st_size)
    return report


def inspect_qec_dataset(
    path: str,
    *,
    verify: bool = True,
) -> dict[str, Any]:
    """Inspect and optionally verify a completed or partial Aegis dataset."""

    h5py, _ = _require_dependencies()
    file_path = Path(path).resolve()
    with h5py.File(file_path, "r") as handle:
        if handle.attrs.get("format") != _DATASET_FORMAT:
            raise ValueError(f"{file_path} is not an Aegis QEC dataset")

        syndromes = handle["syndromes"]
        observables = handle["observables"]
        split = handle["split"]
        complete = bool(handle.attrs.get("complete", False))
        result: dict[str, Any] = {
            "schema_version": int(handle.attrs.get("schema_version", 0)),
            "format": str(handle.attrs["format"]),
            "path": str(file_path),
            "complete": complete,
            "target_shots": int(handle.attrs["target_shots"]),
            "written_shots": int(syndromes.shape[0]),
            "num_detectors": int(syndromes.shape[1]),
            "num_observables": int(observables.shape[1]),
            "num_error_mechanisms": int(
                handle["error_mechanisms/probabilities"].shape[0]
            ),
            "identity_sha256": str(handle.attrs["identity_sha256"]),
            "circuit_sha256": str(handle.attrs["circuit_sha256"]),
            "raw_dem_sha256": str(handle.attrs["raw_dem_sha256"]),
            "decoding_dem_sha256": (
                str(handle.attrs.get("decoding_dem_sha256", "")) or None
            ),
            "dense_mechanism_matrices": bool(
                handle.attrs.get("dense_mechanism_matrices", False)
            ),
            "split_counts": _count_split_labels(split),
        }

        failures: list[str] = []
        if int(observables.shape[0]) != int(syndromes.shape[0]):
            failures.append("observable row count does not match syndromes")
        if int(split.shape[0]) != int(syndromes.shape[0]):
            failures.append("split row count does not match syndromes")

        if verify:
            committed = int(handle.attrs.get("written_shots", 0))
            if committed != int(syndromes.shape[0]):
                failures.append(
                    "dataset contains uncommitted rows; resume before verifying"
                )
            elif "chunk_sha256" in handle:
                try:
                    _verify_chunk_journal(
                        handle,
                        committed=committed,
                        chunk_size=int(handle.attrs["chunk_size"]),
                    )
                except ValueError as exc:
                    failures.append(str(exc))
            elif not complete and committed:
                failures.append(
                    "legacy incomplete dataset has no chunk journal; "
                    "resume with original parameters for deterministic replay"
                )

        if verify and complete:
            expected = {
                "syndromes_sha256": _hash_bool_dataset(syndromes),
                "observables_sha256": _hash_bool_dataset(observables),
                "split_sha256": _hash_uint8_dataset(split),
            }
            for key, actual in expected.items():
                stored = str(handle.attrs.get(key, ""))
                if stored != actual:
                    failures.append(
                        f"{key} mismatch: stored={stored!r}, actual={actual!r}"
                    )
                result[key] = actual

        result["valid"] = not failures
        result["failures"] = failures

    result["file_sha256"] = _sha256_file(file_path)
    result["bytes"] = int(file_path.stat().st_size)
    return result

def evaluate_decoders_on_dataset(
    path: str,
    *,
    decoders: list[str],
    split_name: str = "test",
    max_shots: int | None = None,
    batch_size: int = 10000,
    include_external_plugins: bool = True,
    verify_dataset: bool = True,
) -> dict[str, Any]:
    """Evaluate multiple decoders on identical stored detector-shot rows."""

    if not decoders:
        raise ValueError("at least one decoder is required")
    if len(set(decoders)) != len(decoders):
        raise ValueError("decoder names must be unique")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_shots is not None and int(max_shots) < 1:
        raise ValueError("max_shots must be positive")

    split_name = str(split_name).lower()
    split_labels = {
        "train": 0,
        "validation": 1,
        "test": 2,
    }
    if split_name != "all" and split_name not in split_labels:
        raise ValueError(
            "split_name must be one of: train, validation, test, all"
        )

    inspection = inspect_qec_dataset(path, verify=verify_dataset)
    if not inspection["valid"]:
        raise ValueError(
            "dataset integrity verification failed: "
            + "; ".join(inspection["failures"])
        )
    if not inspection["complete"]:
        raise ValueError("decoder evaluation requires a completed dataset")

    h5py, stim = _require_dependencies()
    from .decoder_plugins import custom_decoder_registry
    from .research import wilson_interval

    registry = custom_decoder_registry(
        include_external=include_external_plugins,
    )
    unknown = [name for name in decoders if name not in registry]
    if unknown:
        raise ValueError(
            "Unknown dataset decoder(s): "
            + ", ".join(unknown)
            + ". Available: "
            + ", ".join(sorted(registry))
        )

    file_path = Path(path).resolve()
    syndrome_digest = hashlib.sha256()
    observable_digest = hashlib.sha256()
    syndrome_digest.update(
        f"{_DATASET_FORMAT}|selected-syndromes|{split_name}|".encode("ascii")
    )
    observable_digest.update(
        f"{_DATASET_FORMAT}|selected-observables|{split_name}|".encode("ascii")
    )

    decoder_errors = {name: 0 for name in decoders}
    decoder_seconds = {name: 0.0 for name in decoders}
    pairwise_disagreements = {
        (left, right): 0
        for left_index, left in enumerate(decoders)
        for right in decoders[left_index + 1 :]
    }
    pairwise_failures = {
        (left, right): {
            "both_success": 0,
            "left_only_failure": 0,
            "right_only_failure": 0,
            "both_failure": 0,
        }
        for left_index, left in enumerate(decoders)
        for right in decoders[left_index + 1 :]
    }
    selected_shots = 0

    with h5py.File(file_path, "r") as handle:
        syndromes = handle["syndromes"]
        observables = handle["observables"]
        split = handle["split"]

        dem_text = handle["decoding_detector_error_model"][()]
        if isinstance(dem_text, bytes):
            dem_text = dem_text.decode("utf-8")
        if not dem_text:
            dem_text = handle["detector_error_model"][()]
            if isinstance(dem_text, bytes):
                dem_text = dem_text.decode("utf-8")
        dem = stim.DetectorErrorModel(str(dem_text))

        compiled = {
            name: registry[name].compile_decoder_for_dem(dem=dem)
            for name in decoders
        }

        for start in range(0, int(syndromes.shape[0]), int(batch_size)):
            if max_shots is not None and selected_shots >= int(max_shots):
                break

            stop = min(int(syndromes.shape[0]), start + int(batch_size))
            detector_chunk = np.asarray(
                syndromes[start:stop],
                dtype=np.bool_,
            )
            observable_chunk = np.asarray(
                observables[start:stop],
                dtype=np.bool_,
            )

            if split_name != "all":
                split_chunk = np.asarray(split[start:stop], dtype=np.uint8)
                mask = split_chunk == split_labels[split_name]
                detector_chunk = detector_chunk[mask]
                observable_chunk = observable_chunk[mask]

            if detector_chunk.shape[0] == 0:
                continue

            if max_shots is not None:
                remaining = int(max_shots) - selected_shots
                detector_chunk = detector_chunk[:remaining]
                observable_chunk = observable_chunk[:remaining]

            detector_packed = np.packbits(
                detector_chunk,
                axis=1,
                bitorder="little",
            )
            observable_packed = np.packbits(
                observable_chunk,
                axis=1,
                bitorder="little",
            )
            detector_packed = np.asarray(detector_packed, dtype=np.uint8)
            observable_packed = np.asarray(observable_packed, dtype=np.uint8)

            syndrome_digest.update(detector_packed.tobytes(order="C"))
            observable_digest.update(observable_packed.tobytes(order="C"))

            chunk_predictions: dict[str, np.ndarray] = {}
            chunk_failures: dict[str, np.ndarray] = {}
            for name in decoders:
                started = time.perf_counter()
                predicted = compiled[name].decode_shots_bit_packed(
                    bit_packed_detection_event_data=detector_packed
                )
                decoder_seconds[name] += time.perf_counter() - started
                predicted = np.asarray(predicted, dtype=np.uint8)
                chunk_predictions[name] = predicted

                failures = np.any(
                    np.bitwise_xor(predicted, observable_packed) != 0,
                    axis=1,
                )
                decoder_errors[name] += int(np.count_nonzero(failures))
                chunk_failures[name] = failures

            for pair in pairwise_disagreements:
                left, right = pair
                disagreement = np.any(
                    np.bitwise_xor(
                        chunk_predictions[left],
                        chunk_predictions[right],
                    )
                    != 0,
                    axis=1,
                )
                pairwise_disagreements[pair] += int(
                    np.count_nonzero(disagreement)
                )

                left_fail = chunk_failures[left]
                right_fail = chunk_failures[right]
                counts = pairwise_failures[pair]
                counts["both_success"] += int(
                    np.count_nonzero(~left_fail & ~right_fail)
                )
                counts["left_only_failure"] += int(
                    np.count_nonzero(left_fail & ~right_fail)
                )
                counts["right_only_failure"] += int(
                    np.count_nonzero(~left_fail & right_fail)
                )
                counts["both_failure"] += int(
                    np.count_nonzero(left_fail & right_fail)
                )

            selected_shots += int(detector_chunk.shape[0])

    if selected_shots < 1:
        raise ValueError(
            f"dataset split {split_name!r} contains no selected shots"
        )

    rows: list[dict[str, Any]] = []
    for name in decoders:
        errors = int(decoder_errors[name])
        ci_low, ci_high = wilson_interval(errors, selected_shots)
        seconds = float(decoder_seconds[name])
        rows.append(
            {
                "decoder": name,
                "shots": selected_shots,
                "errors": errors,
                "logical_error_rate": float(errors / selected_shots),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "decode_seconds": seconds,
                "decode_shots_per_second": float(
                    selected_shots / max(seconds, 1.0e-12)
                ),
            }
        )

    disagreements = []
    paired_failure_statistics = []
    for (left, right), count in pairwise_disagreements.items():
        disagreements.append(
            {
                "left": left,
                "right": right,
                "disagreement_shots": int(count),
                "disagreement_rate": float(count / selected_shots),
            }
        )

        counts = pairwise_failures[(left, right)]
        left_only = int(counts["left_only_failure"])
        right_only = int(counts["right_only_failure"])
        discordant = left_only + right_only
        risk_difference = float((left_only - right_only) / selected_shots)
        second_moment = float(discordant / selected_shots)
        variance = max(0.0, second_moment - risk_difference * risk_difference)
        standard_error = math.sqrt(variance / selected_shots)
        risk_ci_low = max(-1.0, risk_difference - 1.96 * standard_error)
        risk_ci_high = min(1.0, risk_difference + 1.96 * standard_error)

        if discordant == 0:
            mcnemar_chi2 = 0.0
            mcnemar_p = 1.0
        else:
            corrected = max(0.0, abs(left_only - right_only) - 1.0)
            mcnemar_chi2 = float(corrected * corrected / discordant)
            mcnemar_p = float(
                math.erfc(math.sqrt(mcnemar_chi2 / 2.0))
            )

        paired_failure_statistics.append(
            {
                "left": left,
                "right": right,
                **counts,
                "discordant_failure_shots": discordant,
                "left_minus_right_error_rate": risk_difference,
                "left_minus_right_error_rate_ci95_normal": [
                    float(risk_ci_low),
                    float(risk_ci_high),
                ],
                "mcnemar_chi2_continuity_corrected": mcnemar_chi2,
                "mcnemar_p_value_asymptotic": mcnemar_p,
            }
        )

    return {
        "schema_version": 1,
        "evaluation_type": "fixed_qec_dataset",
        "dataset": {
            "path": str(file_path),
            "identity_sha256": inspection["identity_sha256"],
            "file_sha256": inspection["file_sha256"],
            "circuit_sha256": inspection["circuit_sha256"],
            "raw_dem_sha256": inspection["raw_dem_sha256"],
            "split": split_name,
            "selected_shots": selected_shots,
            "selected_syndromes_sha256": syndrome_digest.hexdigest(),
            "selected_observables_sha256": observable_digest.hexdigest(),
        },
        "configuration": {
            "decoders": list(decoders),
            "max_shots": max_shots,
            "batch_size": int(batch_size),
            "verify_dataset": bool(verify_dataset),
        },
        "rows": rows,
        "pairwise_disagreements": disagreements,
        "paired_failure_statistics": paired_failure_statistics,
        "interpretation": (
            "Every decoder received the identical stored detector-shot rows. "
            "Differences therefore reflect paired decoder behavior on a fixed "
            "dataset rather than independent Monte Carlo sampling."
        ),
    }


def write_dataset_evaluation_json(
    evaluation: dict[str, Any],
    path: str,
) -> None:
    """Write a fixed-dataset decoder evaluation artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
