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



def test_partial_dataset_tampering_is_detected_before_resumption(tmp_path):
    path = tmp_path / "partial-tampered.h5"
    params = dict(
        shots=32, chunk_size=8, seed=78623, distance=3,
        physical_error_rate=0.02,
    )
    first = generate_qec_dataset(
        str(path), max_chunks_per_run=2, **params
    )
    assert first["written_shots"] == 16
    assert inspect_qec_dataset(str(path))["valid"] is True
    with h5py.File(path, "r+") as handle:
        assert len(handle["chunk_sha256"]) == 2
        handle["syndromes"][0, 0] = not bool(handle["syndromes"][0, 0])
    verified = inspect_qec_dataset(str(path))
    assert verified["valid"] is False
    assert any("committed chunk integrity mismatch" in item
               for item in verified["failures"])
    with pytest.raises(ValueError, match="committed chunk integrity mismatch"):
        generate_qec_dataset(str(path), **params)
    assert inspect_qec_dataset(str(path))["valid"] is False


def test_legacy_partial_dataset_is_qualified_by_replay_before_resume(tmp_path):
    path = tmp_path / "legacy.h5"
    reference = tmp_path / "reference.h5"
    params = dict(
        shots=32, chunk_size=8, seed=97231, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(str(path), max_chunks_per_run=2, **params)
    with h5py.File(path, "r+") as handle:
        del handle["chunk_sha256"]
    assert inspect_qec_dataset(str(path))["valid"] is False
    completed = generate_qec_dataset(str(path), **params)
    expected = generate_qec_dataset(str(reference), **params)
    for item in ("syndromes_sha256", "observables_sha256", "split_sha256"):
        assert completed[item] == expected[item]
    assert inspect_qec_dataset(str(path))["valid"] is True


def test_legacy_replay_rejects_mutated_partial_file(tmp_path):
    path = tmp_path / "legacy-corrupt.h5"
    params = dict(
        shots=32, chunk_size=8, seed=732, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(str(path), max_chunks_per_run=1, **params)
    with h5py.File(path, "r+") as handle:
        del handle["chunk_sha256"]
        handle["observables"][0, 0] = not bool(handle["observables"][0, 0])
    with pytest.raises(ValueError, match="legacy dataset committed chunk replay mismatch"):
        generate_qec_dataset(str(path), **params)


def test_recovery_discards_uncommitted_chunk_journal(tmp_path):
    path = tmp_path / "journal-interrupt.h5"
    control = tmp_path / "journal-control.h5"
    params = dict(
        shots=32, chunk_size=8, seed=892, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(str(path), max_chunks_per_run=1, **params)
    with h5py.File(path, "r+") as handle:
        handle["syndromes"].resize((9, handle["syndromes"].shape[1]))
        journal = handle["chunk_sha256"]
        journal.resize((2,))
        journal[1] = b"0" * 64
    resumed = generate_qec_dataset(str(path), **params)
    baseline = generate_qec_dataset(str(control), **params)
    assert resumed["syndromes_sha256"] == baseline["syndromes_sha256"]
    assert inspect_qec_dataset(str(path))["valid"] is True



def test_final_short_chunk_recovers_after_interrupted_completion(tmp_path):
    path = tmp_path / "completed-but-unmarked.h5"
    reference = tmp_path / "reference-short.h5"
    params = dict(
        shots=22, seed=98204, chunk_size=8, distance=3,
        physical_error_rate=0.02,
    )
    generate_qec_dataset(str(path), **params)
    with h5py.File(path, "r+") as handle:
        # This is the durable state after the final chunk's commit but
        # before the overall complete=True metadata has been written.
        handle.attrs["complete"] = False
        for name in ("syndromes_sha256", "observables_sha256", "split_sha256"):
            del handle.attrs[name]
        assert int(handle.attrs["written_shots"]) == 22
        assert len(handle["chunk_sha256"]) == 3
    recovered = generate_qec_dataset(str(path), **params)
    expected = generate_qec_dataset(str(reference), **params)
    assert recovered["complete"] is True
    assert recovered["new_chunks"] == 0
    for name in ("syndromes_sha256", "observables_sha256", "split_sha256"):
        assert recovered[name] == expected[name]
    assert inspect_qec_dataset(str(path))["valid"] is True
