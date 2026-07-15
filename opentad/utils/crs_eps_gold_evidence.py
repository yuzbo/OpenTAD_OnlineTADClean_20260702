"""Data-byte and manifest bindings shared by the CRS-EPS G0 tools."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .crs_eps_sampling import sampling_specs_sha256, validate_epoch_manifest
from .full_petal_identity import build_data_identity, scientific_source_identity


class CrsEpsGoldEvidenceError(ValueError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind_manifest_to_loaded_dataset(
    manifest,
    dataset,
    *,
    cfg,
    config_path,
    repository_root,
    commit_sha,
    resolved_config_sha256,
    scientific_config_sha256,
):
    """Reconcile signed claims with the dataset object and source bytes in use."""

    validate_epoch_manifest(manifest)
    actual_specs_sha256 = sampling_specs_sha256(
        dataset.iter_crs_eps_sampling_specs()
    )
    if manifest.get("sampling_specs_sha256") != actual_specs_sha256:
        raise CrsEpsGoldEvidenceError(
            "G0 manifest sampling population differs from the loaded dataset"
        )
    data_identity = build_data_identity(cfg)
    source_tree_sha256 = scientific_source_identity(
        repository_root, scientific_config_sha256
    )
    expected_provenance = {
        "commit_sha": commit_sha,
        "config_sha256": sha256_file(config_path),
        "resolved_config_sha256": resolved_config_sha256,
        "scientific_config_sha256": scientific_config_sha256,
        "data_identity_sha256": data_identity["identity_sha256"],
        "sampling_specs_sha256": actual_specs_sha256,
        "annotation_sha256": sha256_file(dataset.ann_file),
        "feature_cache_manifest_sha256": dataset.cache_manifest_sha256,
        "split_manifest_sha256": dataset.split_manifest_sha256,
        "split_manifest_file_sha256": dataset.split_manifest_file_sha256,
        "split_role": dataset.split_role,
        "split_seed": dataset.split_seed,
    }
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise CrsEpsGoldEvidenceError("G0 manifest provenance is not an object")
    drifted = sorted(
        key for key, expected in expected_provenance.items()
        if provenance.get(key) != expected
    )
    if drifted:
        raise CrsEpsGoldEvidenceError(
            "G0 manifest provenance differs from loaded source bytes: "
            + ", ".join(drifted)
        )
    return {
        "commit_sha": commit_sha,
        "source_tree_sha256": source_tree_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "scientific_config_sha256": scientific_config_sha256,
        "data_identity_sha256": data_identity["identity_sha256"],
        "episode_manifest_sha256": manifest["manifest_sha256"],
        "sampling_specs_sha256": actual_specs_sha256,
    }


__all__ = [
    "CrsEpsGoldEvidenceError",
    "bind_manifest_to_loaded_dataset",
    "sha256_file",
]
