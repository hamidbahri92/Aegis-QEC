from __future__ import annotations

import numpy as np
import pytest

from aegis_qec.dataset import (
    extract_dem_mechanisms,
    generate_qec_dataset,
    inspect_qec_dataset,
)

h5py = pytest.importorskip("h5py")
stim = pytest.importorskip("stim")


def test_dem_mechanism_export_preserves_detector_and_observable_incidence():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D2 L0
        error(0.2) D1
        """
    )
    mechanisms = extract_dem_mechanisms(dem)

    np.testing.assert_allclose(
        mechanisms["probabilities"],
        np.asarray([0.1, 0.2]),
    )
    np.testing.assert_array_equal(
        mechanisms["detector_indptr"],
        np.asarray([0, 2, 3]),
    )
    np.testing.assert_array_equal(
        mechanisms["detector_indices"],
        np.asarray([0, 2, 1]),
    )
    np.testing.assert_array_equal(
        mechanisms["observable_indptr"],
        np.asarray([0, 1, 1]),
    )
    np.testing.assert_array_equal(
        mechanisms["observable_indices"],
        np.asarray([0]),
    )


def test_dataset_generation_resumes_deterministically(tmp_path):
    resumed_path = tmp_path / "resumed.h5"
    kwargs = {
        "shots": 40,
        "seed": 20261006,
        "chunk_size": 16,
        "distance": 3,
        "physical_error_rate": 0.02,
        "basis": "x",
    }

    first = generate_qec_dataset(
        str(resumed_path),
        max_chunks_per_run=1,
        **kwargs,
    )
    assert first["complete"] is False
    assert first["written_shots"] == 16

    second = generate_qec_dataset(
        str(resumed_path),
        max_chunks_per_run=1,
        **kwargs,
    )
    assert second["complete"] is False
    assert second["written_shots"] == 32

    third = generate_qec_dataset(str(resumed_path), **kwargs)
    assert third["complete"] is True
    assert third["written_shots"] == 40

    inspection = inspect_qec_dataset(str(resumed_path))
    assert inspection["valid"] is True
    assert sum(inspection["split_counts"].values()) == 40
    assert inspection["num_error_mechanisms"] > 0

    one_shot_path = tmp_path / "one-shot.h5"
    one_shot = generate_qec_dataset(str(one_shot_path), **kwargs)
    assert one_shot["complete"] is True
    assert one_shot["identity_sha256"] == third["identity_sha256"]
    assert one_shot["syndromes_sha256"] == third["syndromes_sha256"]
    assert one_shot["observables_sha256"] == third["observables_sha256"]
    assert one_shot["split_sha256"] == third["split_sha256"]


def test_dataset_resume_rejects_configuration_change(tmp_path):
    path = tmp_path / "dataset.h5"
    generate_qec_dataset(
        str(path),
        shots=20,
        seed=3,
        chunk_size=10,
        max_chunks_per_run=1,
        distance=3,
        physical_error_rate=0.02,
    )

    with pytest.raises(ValueError, match="configuration does not match"):
        generate_qec_dataset(
            str(path),
            shots=20,
            seed=4,
            chunk_size=10,
            distance=3,
            physical_error_rate=0.02,
        )


def test_completed_dataset_verifier_detects_sample_tampering(tmp_path):
    path = tmp_path / "dataset.h5"
    generate_qec_dataset(
        str(path),
        shots=24,
        seed=11,
        chunk_size=8,
        distance=3,
        physical_error_rate=0.02,
    )

    before = inspect_qec_dataset(str(path))
    assert before["valid"] is True

    with h5py.File(path, "r+") as handle:
        original = bool(handle["syndromes"][0, 0])
        handle["syndromes"][0, 0] = not original

    after = inspect_qec_dataset(str(path))
    assert after["valid"] is False
    assert any(
        "syndromes_sha256 mismatch" in failure
        for failure in after["failures"]
    )


def test_dense_mechanism_matrices_can_be_disabled(tmp_path):
    path = tmp_path / "sparse-only.h5"
    report = generate_qec_dataset(
        str(path),
        shots=8,
        seed=5,
        chunk_size=8,
        distance=3,
        physical_error_rate=0.02,
        dense_matrix_max_cells=0,
    )
    assert report["dense_mechanism_matrices"] is False

    with h5py.File(path, "r") as handle:
        assert "check_matrix" not in handle
        assert "obs_matrix" not in handle
        assert "error_mechanisms/detector_indices" in handle
        assert "error_mechanisms/observable_indices" in handle



def test_reopening_complete_dataset_cannot_recertify_tampered_samples(tmp_path):
    path = tmp_path / "signed.h5"
    params = dict(
        shots=24, seed=12941, chunk_size=8, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(str(path), **params)
    with h5py.File(path, "r+") as handle:
        originally_stored = str(handle.attrs["syndromes_sha256"])
        handle["syndromes"][0, 0] = not bool(handle["syndromes"][0, 0])
    with pytest.raises(ValueError, match="completed dataset integrity mismatch"):
        generate_qec_dataset(str(path), **params)
    with h5py.File(path, "r") as handle:
        assert str(handle.attrs["syndromes_sha256"]) == originally_stored
    assert inspect_qec_dataset(str(path))["valid"] is False


def test_partial_dataset_discards_uncommitted_append_after_interruption(tmp_path):
    path = tmp_path / "interrupted.h5"
    control = tmp_path / "uninterrupted.h5"
    params = dict(
        shots=40, seed=19847, chunk_size=8, distance=3,
        physical_error_rate=0.02,
    )
    first = generate_qec_dataset(
        str(path), max_chunks_per_run=1, **params
    )
    assert first["written_shots"] == 8
    with h5py.File(path, "r+") as handle:
        syndromes = handle["syndromes"]
        syndromes.resize((10, syndromes.shape[1]))
        syndromes[8:10, :] = False
        observables = handle["observables"]
        observables.resize((11, observables.shape[1]))
        observables[8:11, :] = True
        handle.flush()
    recovered = generate_qec_dataset(str(path), **params)
    reference = generate_qec_dataset(str(control), **params)
    assert recovered["complete"] is True
    for name in (
        "identity_sha256", "syndromes_sha256",
        "observables_sha256", "split_sha256",
    ):
        assert recovered[name] == reference[name]
    assert inspect_qec_dataset(str(path))["valid"] is True


def test_partial_dataset_does_not_invent_lost_committed_rows(tmp_path):
    path = tmp_path / "lost-committed.h5"
    params = dict(
        shots=24, seed=2498, chunk_size=8, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(
        str(path), max_chunks_per_run=1, **params
    )
    with h5py.File(path, "r+") as handle:
        handle["observables"].resize(
            (7, handle["observables"].shape[1])
        )
    with pytest.raises(RuntimeError, match="committed dataset rows are missing"):
        generate_qec_dataset(str(path), **params)
