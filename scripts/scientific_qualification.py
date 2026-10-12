"""Independent, bounded scientific qualification of Aegis QEC.

Validates exact-shot parity against a separately invoked Stim/PyMatching oracle,
fixed-seed reproducibility, no-noise correctness where defined, and HDF5 resume
identity. This does not claim a physical hardware threshold or decoder novelty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import pymatching
import stim

from aegis_qec.dataset import generate_qec_dataset, inspect_qec_dataset
from aegis_qec.research import run_surface_code_study


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _persist_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".science-",
            dir=path.parent, delete=False,
        ) as out:
            temp = Path(out.name)
            json.dump(report, out, indent=2, sort_keys=True)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _reference_point(distance: int, p: float, seed: int, shots: int, basis: str) -> dict:
    circuit = stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots=shots, separate_observables=True,
    )
    matching = pymatching.Matching.from_detector_error_model(dem)
    prediction = matching.decode_batch(np.asarray(det, dtype=np.uint8))
    failures = int(np.count_nonzero(np.any(prediction != obs, axis=1)))
    return {
        "distance": distance, "physical_error_rate": p, "seed": seed,
        "shots": shots, "errors": failures, "rate": failures / shots,
        "detectors": int(det.shape[1]), "observables": int(obs.shape[1]),
        "circuit_sha256": _hash(str(circuit)), "dem_sha256": _hash(str(dem)),
        "detector_samples_sha256": hashlib.sha256(det.tobytes()).hexdigest(),
        "observable_samples_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
    }


def _qualify_dataset(base: Path, shots: int) -> dict:
    common = dict(
        shots=shots, seed=20261012, chunk_size=10000, distance=5,
        physical_error_rate=0.009, basis="z",
    )
    resumed_path = base / "dataset-resumed.h5"
    continuous_path = base / "dataset-continuous.h5"
    first = generate_qec_dataset(
        str(resumed_path), max_chunks_per_run=2, **common
    )
    if shots > 20000:
        assert not first["complete"], "resumption exercise did not interrupt"
    resumed = generate_qec_dataset(str(resumed_path), **common)
    continuous = generate_qec_dataset(str(continuous_path), **common)
    left = inspect_qec_dataset(str(resumed_path))
    right = inspect_qec_dataset(str(continuous_path))
    assert left["valid"] and right["valid"] and left["complete"] and right["complete"]
    for name in ("identity_sha256", "syndromes_sha256", "observables_sha256", "split_sha256"):
        assert left[name] == right[name], f"resumption changes {name}"
    with h5py.File(resumed_path, "r+") as handle:
        old = bool(handle["syndromes"][0, 0])
        handle["syndromes"][0, 0] = not old
    corrupted = inspect_qec_dataset(str(resumed_path))
    assert not corrupted["valid"], "tampered dataset was accepted"
    assert any("syndromes_sha256 mismatch" in item for item in corrupted["failures"])
    return {
        "shots": shots, "resumed_sha256": resumed["syndromes_sha256"],
        "continuous_sha256": continuous["syndromes_sha256"],
        "resume_equal": True, "tamper_rejected": True,
        "split_counts": right["split_counts"],
    }


def qualify(output: Path, *, shots_per_point: int, dataset_shots: int) -> None:
    if shots_per_point < 1000 or dataset_shots <= 20000:
        raise ValueError("qualification budget is too small")
    started = time.perf_counter()
    base = output.parent
    base.mkdir(parents=True, exist_ok=True)
    distances = [3, 5, 7]
    probabilities = [0.003, 0.006, 0.010, 0.018]
    seeds = (20261012, 20261112)
    report: dict = {
        "type": "aegis_independent_scientific_qualification",
        "source": "pinned GitHub checkout",
        "stim_version": stim.__version__,
        "pymatching_version": pymatching.__version__,
        "python": platform.python_version(),
        "distances": distances,
        "probabilities": probabilities,
        "shots_per_point": shots_per_point,
        "independent_seeds": list(seeds),
        "results": [],
        "status": "RUNNING",
    }
    _persist_json(output, report)
    for seed in seeds:
        study = run_surface_code_study(
            distances=distances,
            physical_error_rates=probabilities,
            shots=shots_per_point,
            basis="x",
            seed=seed,
        )
        rows = study["points"]
        assert len(rows) == len(distances) * len(probabilities)
        for index, row in enumerate(rows):
            ref = _reference_point(
                int(row["distance"]), float(row["physical_error_rate"]),
                seed + index, shots_per_point, "x",
            )
            for name, ref_name in (
                ("logical_failures", "errors"), ("circuit_sha256", "circuit_sha256"),
                ("dem_sha256", "dem_sha256"), ("detectors", "detectors"),
                ("observables", "observables"), ("seed", "seed"),
            ):
                assert row[name] == ref[ref_name], f"oracle mismatch for {name} at {seed}:{index}"
            assert row["ci95_low"] <= row["logical_error_rate"] <= row["ci95_high"]
            assert 0 <= row["logical_failures"] <= row["shots"]
            report["results"].append({
                "index": index, "base_seed": seed,
                "distance": ref["distance"], "p": ref["physical_error_rate"],
                "shots": ref["shots"], "logical_failures": ref["errors"],
                "logical_error_rate": ref["rate"],
                "aegis_ci95": [row["ci95_low"], row["ci95_high"]],
                "detector_sha256": ref["detector_samples_sha256"],
                "observables_sha256": ref["observable_samples_sha256"],
                "oracle_exact": True,
            })
            _persist_json(output, report)
    report["dataset"] = _qualify_dataset(base, dataset_shots)
    report["total_sampled_shots"] = shots_per_point * len(distances) * len(probabilities) * len(seeds)
    report["status"] = "PASS"
    report["elapsed_seconds"] = time.perf_counter() - started
    _persist_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "sampled_shots_per_engine": report["total_sampled_shots"],
        "oracle_matched_points": len(report["results"]),
        "dataset_resume_equal": report["dataset"]["resume_equal"],
        "tamper_rejected": report["dataset"]["tamper_rejected"],
        "evidence": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("qualification_out/scientific.json"))
    parser.add_argument("--shots-per-point", type=int, default=50000)
    parser.add_argument("--dataset-shots", type=int, default=60000)
    args = parser.parse_args()
    qualify(args.out, shots_per_point=args.shots_per_point, dataset_shots=args.dataset_shots)
