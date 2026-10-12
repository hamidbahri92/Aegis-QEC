from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return cleaned or "experiment"


def load_experiment_manifest(path: str) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("experiment manifest must contain a JSON object")
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("unsupported experiment manifest schema_version")
    operation = str(value.get("operation", "")).strip().lower()
    if operation not in {
        "study",
        "campaign",
        "compare",
        "predict",
        "scaling",
        "explain",
    }:
        raise ValueError(
            "operation must be one of: study, campaign, compare, predict, "
            "scaling, explain"
        )
    parameters = value.get("parameters", {})
    outputs = value.get("outputs", {})
    if not isinstance(parameters, dict):
        raise ValueError("manifest parameters must be an object")
    if not isinstance(outputs, dict):
        raise ValueError("manifest outputs must be an object")

    required_parameters = {
        "predict": ("dem_path", "dets_path"),
        "scaling": ("campaign_json", "decoder"),
    }
    missing = [
        key
        for key in required_parameters.get(operation, ())
        if key not in parameters
    ]
    if missing:
        raise ValueError(
            f"{operation} manifest is missing required parameter(s): "
            + ", ".join(missing)
        )

    value["operation"] = operation
    return value


def _resolve_input(value: str, manifest_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_dir / path
    return str(path.resolve())


def _resolve_output(
    value: str | None,
    output_dir: Path,
    default_name: str,
) -> str:
    path = Path(value or default_name)
    if not path.is_absolute():
        path = output_dir / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path.resolve())


def _environment_record() -> dict[str, Any]:
    packages = [
        "aegis-qec",
        "stim",
        "sinter",
        "pymatching",
        "numpy",
        "scipy",
        "matplotlib",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _distribution_version(name)
            for name in packages
        },
        "github_sha": os.environ.get("GITHUB_SHA"),
    }


def _artifact_record(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"expected artifact was not created: {file_path}")
    return {
        "path": str(file_path),
        "sha256": _sha256_file(file_path),
        "bytes": int(file_path.stat().st_size),
    }


def _existing_artifacts(paths: dict[str, str | None]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for logical_name, raw_path in paths.items():
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file():
            result[logical_name] = _artifact_record(str(path))
    return result


def _run_study(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .research import run_surface_code_study, write_study_artifacts

    result = run_surface_code_study(
        distances=parameters.get("distances", [3, 5, 7]),
        physical_error_rates=parameters.get(
            "physical_error_rates",
            [0.001, 0.003, 0.006, 0.01],
        ),
        shots=int(parameters.get("shots", 1000)),
        basis=str(parameters.get("basis", "x")),
        rounds=parameters.get("rounds"),
        seed=int(parameters.get("seed", 1234)),
    )
    artifact_paths = {
        "study_json": _resolve_output(
            outputs.get("json"),
            output_dir,
            "study.json",
        ),
        "study_csv": _resolve_output(
            outputs.get("csv"),
            output_dir,
            "study.csv",
        ),
        "study_plot": _resolve_output(
            outputs.get("plot"),
            output_dir,
            "study.png",
        ),
    }
    write_study_artifacts(
        result,
        json_path=artifact_paths["study_json"],
        csv_path=artifact_paths["study_csv"],
        plot_path=artifact_paths["study_plot"],
    )
    return result, artifact_paths, {}


def _run_campaign(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
    manifest_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .campaign import run_campaign, write_campaign_summary

    circuit_paths = [
        _resolve_input(str(value), manifest_dir)
        for value in parameters.get("circuit_paths", [])
    ]
    resume_csv = _resolve_output(
        outputs.get("resume_csv"),
        output_dir,
        "campaign.csv",
    )
    result = run_campaign(
        distances=parameters.get("distances", [3, 5, 7]),
        physical_error_rates=parameters.get(
            "physical_error_rates",
            [0.003, 0.006, 0.01],
        ),
        basis=str(parameters.get("basis", "x")),
        rounds=parameters.get("rounds"),
        circuit_paths=circuit_paths,
        decoders=parameters.get(
            "decoders",
            ["pymatching", "aegis-pymatching"],
        ),
        workers=parameters.get("workers", "auto"),
        max_shots=int(parameters.get("max_shots", 100000)),
        max_errors=parameters.get("max_errors", 1000),
        resume_csv=resume_csv,
        max_batch_seconds=parameters.get("max_batch_seconds", 30),
        include_external_plugins=bool(
            parameters.get("include_external_plugins", True)
        ),
        print_progress=bool(parameters.get("print_progress", True)),
    )
    artifact_paths = {
        "campaign_resume_csv": resume_csv,
        "campaign_json": _resolve_output(
            outputs.get("json"),
            output_dir,
            "campaign.json",
        ),
        "campaign_plot": _resolve_output(
            outputs.get("plot"),
            output_dir,
            "campaign.png",
        ),
    }
    write_campaign_summary(
        result,
        json_path=artifact_paths["campaign_json"],
        plot_path=artifact_paths["campaign_plot"],
    )
    inputs = {
        f"circuit_{index}": path
        for index, path in enumerate(circuit_paths)
    }
    return result, artifact_paths, inputs


def _run_compare(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
    manifest_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .comparison import (
        compare_decoders_exact_shots,
        write_comparison_json,
    )

    circuit_path = parameters.get("circuit_path")
    resolved_circuit = (
        _resolve_input(str(circuit_path), manifest_dir)
        if circuit_path
        else None
    )
    result = compare_decoders_exact_shots(
        decoders=parameters.get(
            "decoders",
            ["aegis-pymatching", "aegis-pymatching-correlated"],
        ),
        shots=int(parameters.get("shots", 10000)),
        seed=int(parameters.get("seed", 1234)),
        circuit_path=resolved_circuit,
        distance=int(parameters.get("distance", 5)),
        physical_error_rate=float(
            parameters.get("physical_error_rate", 0.006)
        ),
        basis=str(parameters.get("basis", "x")),
        rounds=parameters.get("rounds"),
        include_external_plugins=bool(
            parameters.get("include_external_plugins", True)
        ),
    )
    json_path = _resolve_output(
        outputs.get("json"),
        output_dir,
        "comparison.json",
    )
    write_comparison_json(result, json_path)
    inputs = {"circuit": resolved_circuit} if resolved_circuit else {}
    return result, {"comparison_json": json_path}, inputs


def _run_predict(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
    manifest_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .io_decode import predict_observables_from_files

    dem_path = _resolve_input(str(parameters["dem_path"]), manifest_dir)
    dets_path = _resolve_input(str(parameters["dets_path"]), manifest_dir)
    prediction_path = _resolve_output(
        outputs.get("predictions"),
        output_dir,
        "predictions.b8",
    )
    provenance_path = _resolve_output(
        outputs.get("provenance_json"),
        output_dir,
        "prediction-provenance.json",
    )
    result = predict_observables_from_files(
        dem_path=dem_path,
        dets_path=dets_path,
        dets_format=str(parameters.get("dets_format", "b8")),
        output_path=prediction_path,
        output_format=str(parameters.get("output_format", "b8")),
        decoder=str(parameters.get("decoder", "aegis-pymatching")),
        provenance_json=provenance_path,
        include_external_plugins=bool(
            parameters.get("include_external_plugins", True)
        ),
    )
    return (
        result,
        {
            "predictions": prediction_path,
            "prediction_provenance": provenance_path,
        },
        {"dem": dem_path, "detector_shots": dets_path},
    )


def _run_scaling(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
    manifest_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .scaling import (
        fit_surface_code_scaling,
        load_campaign_json,
        write_scaling_artifacts,
    )

    campaign_path = _resolve_input(
        str(parameters["campaign_json"]),
        manifest_dir,
    )
    campaign = load_campaign_json(campaign_path)
    result = fit_surface_code_scaling(
        campaign,
        decoder=str(parameters["decoder"]),
        nu_min=float(parameters.get("nu_min", 0.5)),
        nu_max=float(parameters.get("nu_max", 3.0)),
        pc_grid_points=int(parameters.get("pc_grid_points", 61)),
        nu_grid_points=int(parameters.get("nu_grid_points", 31)),
    )
    artifact_paths = {
        "scaling_json": _resolve_output(
            outputs.get("json"),
            output_dir,
            "scaling.json",
        ),
        "scaling_plot": _resolve_output(
            outputs.get("plot"),
            output_dir,
            "scaling.png",
        ),
    }
    write_scaling_artifacts(
        campaign,
        result,
        json_path=artifact_paths["scaling_json"],
        plot_path=artifact_paths["scaling_plot"],
    )
    return result, artifact_paths, {"campaign_json": campaign_path}


def _run_explain(
    parameters: dict[str, Any],
    outputs: dict[str, Any],
    output_dir: Path,
) -> tuple[Any, dict[str, str | None], dict[str, str]]:
    from .explain import explain_surface_code_shot, write_shot_explanation

    result = explain_surface_code_shot(
        distance=int(parameters.get("distance", 5)),
        physical_error_rate=float(
            parameters.get("physical_error_rate", 0.006)
        ),
        basis=str(parameters.get("basis", "x")),
        rounds=parameters.get("rounds"),
        seed=int(parameters.get("seed", 1234)),
    )
    artifact_paths = {
        "explanation_json": _resolve_output(
            outputs.get("json"),
            output_dir,
            "shot-explanation.json",
        ),
        "explanation_plot": _resolve_output(
            outputs.get("plot"),
            output_dir,
            "shot-explanation.png",
        ),
    }
    write_shot_explanation(
        result,
        json_path=artifact_paths["explanation_json"],
        plot_path=artifact_paths["explanation_plot"],
    )
    return result, artifact_paths, {}


def run_experiment_manifest(
    manifest_path: str,
    *,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Execute a version-controlled Aegis experiment manifest."""

    source = Path(manifest_path).resolve()
    manifest_bytes = source.read_bytes()
    manifest = load_experiment_manifest(str(source))
    manifest_dir = source.parent
    name = _safe_name(str(manifest.get("name", source.stem)))
    destination = Path(output_dir or f"research_out/{name}").resolve()
    destination.mkdir(parents=True, exist_ok=True)

    parameters = dict(manifest.get("parameters", {}))
    outputs = dict(manifest.get("outputs", {}))
    operation = str(manifest["operation"])

    if operation == "study":
        result, paths, inputs = _run_study(parameters, outputs, destination)
    elif operation == "campaign":
        result, paths, inputs = _run_campaign(
            parameters,
            outputs,
            destination,
            manifest_dir,
        )
    elif operation == "compare":
        result, paths, inputs = _run_compare(
            parameters,
            outputs,
            destination,
            manifest_dir,
        )
    elif operation == "predict":
        result, paths, inputs = _run_predict(
            parameters,
            outputs,
            destination,
            manifest_dir,
        )
    elif operation == "scaling":
        result, paths, inputs = _run_scaling(
            parameters,
            outputs,
            destination,
            manifest_dir,
        )
    elif operation == "explain":
        result, paths, inputs = _run_explain(parameters, outputs, destination)
    else:
        raise AssertionError(operation)

    run_record_path = destination / "aegis-run.json"
    run_record = {
        "schema_version": 1,
        "record_type": "aegis_qec_experiment_run",
        "name": name,
        "operation": operation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": {
            "path": str(source),
            "sha256": _sha256_bytes(manifest_bytes),
        },
        "environment": _environment_record(),
        "inputs": _existing_artifacts(inputs),
        "artifacts": _existing_artifacts(paths),
        "result": result,
    }
    run_record_path.write_text(
        json.dumps(run_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_record["run_record"] = _artifact_record(str(run_record_path))
    return run_record


def _zip_write_bytes(
    archive: zipfile.ZipFile,
    name: str,
    data: bytes,
) -> None:
    info = zipfile.ZipInfo(name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data)


def create_research_bundle(
    *,
    manifest_path: str,
    run_record: dict[str, Any],
    bundle_path: str,
) -> dict[str, Any]:
    """Create a self-verifying ZIP bundle for one Aegis experiment run."""

    source = Path(manifest_path).resolve()
    bundle = Path(bundle_path).resolve()
    bundle.parent.mkdir(parents=True, exist_ok=True)

    run_record_info = run_record.get("run_record", {})
    run_record_path = Path(str(run_record_info.get("path", "")))
    if not run_record_path.is_file():
        raise FileNotFoundError(
            "run_record does not point to the exact on-disk aegis-run.json"
        )

    manifest_bytes = source.read_bytes()
    run_bytes = run_record_path.read_bytes()
    for label, data, record in (
        ("manifest", manifest_bytes, run_record.get("manifest", {})),
        ("run_record", run_bytes, run_record_info),
    ):
        if not isinstance(record, dict) or record.get("sha256") != _sha256_bytes(data):
            raise ValueError(f"{label} hash mismatch before bundling")
        if record.get("bytes") is not None and int(record["bytes"]) != len(data):
            raise ValueError(f"{label} size mismatch before bundling")

    payloads: dict[str, bytes] = {
        "experiment.json": manifest_bytes,
        "aegis-run.json": run_bytes,
    }

    for category in ("inputs", "artifacts"):
        values = run_record.get(category, {})
        if not isinstance(values, dict):
            raise ValueError(f"invalid {category} inventory in run record")
        for logical_name, record in sorted(values.items()):
            if not isinstance(record, dict) or not record.get("path"):
                raise ValueError(f"invalid {category} record: {logical_name}")
            path = Path(str(record["path"]))
            if not path.is_file():
                raise FileNotFoundError(
                    f"required {category} evidence missing: {path}"
                )
            archive_name = (
                f"{category}/{_safe_name(str(logical_name))}/{path.name}"
            )
            if archive_name in payloads:
                raise ValueError(f"bundle evidence name collision: {archive_name}")
            evidence_bytes = path.read_bytes()
            if record.get("sha256") != _sha256_bytes(evidence_bytes):
                raise ValueError(
                    f"recorded {category} evidence hash mismatch: {logical_name}"
                )
            if record.get("bytes") is not None and int(record["bytes"]) != len(evidence_bytes):
                raise ValueError(
                    f"recorded {category} evidence size mismatch: {logical_name}"
                )
            payloads[archive_name] = evidence_bytes

    entries = []
    for name, data in sorted(payloads.items()):
        entries.append(
            {
                "path": name,
                "sha256": _sha256_bytes(data),
                "bytes": len(data),
            }
        )

    bundle_manifest = {
        "schema_version": 1,
        "bundle_type": "aegis_qec_research_bundle",
        "entries": entries,
    }

    with zipfile.ZipFile(bundle, "w") as archive:
        for name, data in sorted(payloads.items()):
            _zip_write_bytes(archive, name, data)
        _zip_write_bytes(
            archive,
            "BUNDLE_MANIFEST.json",
            (
                json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8"),
        )

    return {
        "path": str(bundle),
        "sha256": _sha256_file(bundle),
        "bytes": int(bundle.stat().st_size),
        "entries": entries,
    }



def verify_research_bundle(bundle_path: str) -> dict[str, Any]:
    """Verify the complete research bundle without ambiguous ZIP membership."""
    from .archive_integrity import verify_manifested_payload

    bundle = Path(bundle_path).resolve()
    failures = verify_manifested_payload(
        bundle,
        manifest_name="BUNDLE_MANIFEST.json",
        manifest_type="aegis_qec_research_bundle",
        inventory_name="entries",
        required_names=("experiment.json", "aegis-run.json"),
        has_checksums=False,
    )
    if not failures:
        # Membership and hashes alone cannot prove that a bundle contains
        # *all* evidence cited by its embedded scientific run record.
        try:
            with zipfile.ZipFile(bundle, "r") as archive:
                manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
                run_record = json.loads(archive.read("aegis-run.json"))
                inventory = {
                    item["path"]: item["sha256"]
                    for item in manifest["entries"]
                }
                if run_record.get("manifest", {}).get("sha256") != _sha256_bytes(
                    archive.read("experiment.json")
                ):
                    failures.append("run record experiment manifest mismatch")
                for category in ("inputs", "artifacts"):
                    records = run_record.get(category, {})
                    if not isinstance(records, dict):
                        failures.append(f"run record has invalid {category} inventory")
                        continue
                    for logical_name, record in records.items():
                        if not isinstance(record, dict) or not record.get("path"):
                            failures.append(
                                f"run record has invalid {category}: {logical_name}"
                            )
                            continue
                        base_name = str(record["path"]).replace("\\\\", "/").rsplit("/", 1)[-1]
                        bundle_name = (
                            f"{category}/{_safe_name(str(logical_name))}/{base_name}"
                        )
                        if inventory.get(bundle_name) != record.get("sha256"):
                            failures.append(
                                f"missing or inconsistent referenced evidence: {bundle_name}"
                            )
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            failures.append(f"invalid research run record: {exc}")
    return {
        "schema_version": 1,
        "bundle_type": "aegis_qec_research_bundle_verification",
        "path": str(bundle),
        "bundle_sha256": _sha256_file(bundle),
        "valid": not failures,
        "failures": failures,
    }
