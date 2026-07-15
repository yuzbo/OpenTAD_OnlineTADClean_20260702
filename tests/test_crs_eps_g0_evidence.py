import json
from pathlib import Path

import pytest

import opentad.utils.crs_eps_gold_evidence as gold_evidence_module
import opentad.utils.full_petal_launch as launch_module
from opentad.utils.crs_eps_gold_evidence import (
    CrsEpsGoldEvidenceError,
    bind_manifest_to_loaded_dataset,
)
from opentad.utils.crs_eps_gold_gate import (
    AUDIT_SCHEMA_VERSION,
    MARGIN_SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
    evaluate_gold_audit,
)
from opentad.utils.crs_eps_sampling import (
    VideoSamplingSpec,
    build_epoch_manifest,
    sampling_specs_sha256,
)
from opentad.utils.full_petal_attestation import generate_private_key
from opentad.utils.full_petal_launch import FullPetalLaunchError
from opentad.utils.full_petal_role_signing import (
    sign_crs_eps_g0_audit,
    sign_crs_eps_g0_margins,
    sign_crs_eps_g0_selection,
)


COMMIT = "a" * 40
SOURCE = "b" * 64
RESOLVED = "c" * 64
SCIENTIFIC = "d" * 64
DATA = "e" * 64
CONFIG = "f" * 64


def _write(path, payload):
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path):
    return launch_module.sha256_file(path)


def _trace(mode, *, fail=False):
    values = {
        "dynamic_birth": (0.98, 0.98, 0.97, 0.60, 1.02),
        "fixed_192": (0.94, 0.95, 0.94, 0.50, 1.04),
        "reset": (0.88, 0.90, 0.85, 0.20, 1.10),
    }
    gradient, sign, runtime, ratio, loss = values[mode]
    if fail and mode == "dynamic_birth":
        gradient = 0.1
    gold_tokens = 100
    candidate_tokens = int(gold_tokens * ratio)
    return {
        "comparison_type": "replay_fidelity",
        "left": {
            "losses": {"cost": 1.0},
            "audit": {"replay_range": [0, gold_tokens]},
        },
        "right": {
            "losses": {"cost": loss},
            "audit": {
                "replay_range": [gold_tokens - candidate_tokens, gold_tokens]
            },
        },
        "gradient_comparison": {
            "cosine": gradient,
            "sign_agreement": sign,
        },
        "runtime_continuous_comparison": {"cosine": runtime},
        "runtime_discrete_equal": True,
    }


def _g0_bundle(tmp_path, *, fail=False):
    private_key = tmp_path / "g0.pem"
    public_key = generate_private_key(private_key)
    trust_root = {"key_id": "g0-test", "public_key": public_key}
    spec = VideoSamplingSpec("video", 100, ())
    manifest = build_epoch_manifest(
        [spec], epoch=0, seed=705, draws_per_video=1, provenance={}
    )
    manifest_path = _write(tmp_path / "manifest.json", manifest)
    bindings = {
        "commit_sha": COMMIT,
        "resolved_config_sha256": RESOLVED,
        "scientific_config_sha256": SCIENTIFIC,
        "data_identity_sha256": DATA,
        "episode_manifest_sha256": manifest["manifest_sha256"],
        "sampling_specs_sha256": manifest["sampling_specs_sha256"],
    }
    selection_path = _write(
        tmp_path / "selection.json",
        sign_crs_eps_g0_selection(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "status": "PREREGISTERED_BEFORE_G0_EXECUTION",
                **bindings,
                "samples": [{"video_id": "video", "draw_index": 0}],
            },
            private_key_path=private_key,
            key_id="g0-test",
        ),
    )
    margins = {
        "schema_version": MARGIN_SCHEMA_VERSION,
        "status": "PREREGISTERED_BEFORE_G0_EXECUTION",
        **bindings,
        "selection_artifact_sha256": _sha256(selection_path),
        "min_gradient_cosine": 0.90,
        "min_gradient_sign_agreement": 0.90,
        "min_runtime_continuous_cosine": 0.90,
        "max_relative_loss_error": 0.10,
        "require_runtime_discrete_equal": True,
        "max_mean_dynamic_replay_ratio": 0.80,
        "max_video_start_fallback_fraction": 0.25,
        "min_dynamic_minus_fixed_gradient_cosine": 0.02,
        "min_dynamic_minus_reset_gradient_cosine": 0.05,
    }
    margins_path = _write(
        tmp_path / "margins.json",
        sign_crs_eps_g0_margins(
            margins,
            private_key_path=private_key,
            key_id="g0-test",
        ),
    )
    rows = [
        {
            "video_id": "video",
            "draw_index": 0,
            "mode": mode,
            "trace": _trace(mode, fail=fail),
        }
        for mode in ("dynamic_birth", "fixed_192", "reset")
    ]
    gate = evaluate_gold_audit(rows, margins)
    audit_path = _write(
        tmp_path / "audit.json",
        sign_crs_eps_g0_audit(
            {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "status": gate["status"],
                "commit_sha": COMMIT,
                "source_tree_sha256": SOURCE,
                "resolved_config_sha256": RESOLVED,
                "scientific_config_sha256": SCIENTIFIC,
                "data_identity_sha256": DATA,
                "config": {"path": "config.py", "sha256": CONFIG},
                "checkpoint": {
                    "path": "checkpoint.pth",
                    "sha256": "1" * 64,
                    "state_key": "state_dict",
                },
                "episode_manifest": {
                    "path": manifest_path.name,
                    "file_sha256": _sha256(manifest_path),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "sampling_specs_sha256": manifest["sampling_specs_sha256"],
                },
                "selection": {
                    "path": selection_path.name,
                    "sha256": _sha256(selection_path),
                },
                "margins": {
                    "path": margins_path.name,
                    "sha256": _sha256(margins_path),
                },
                "rows": rows,
                "gate": gate,
            },
            private_key_path=private_key,
            key_id="g0-test",
        ),
    )
    return {
        "private_key": private_key,
        "trust_root": trust_root,
        "audit": audit_path,
        "margins": margins_path,
    }


def _validate(bundle, **overrides):
    values = {
        "commit_sha": COMMIT,
        "source_tree_sha256": SOURCE,
        "config_file_sha256": CONFIG,
        "resolved_config_sha256": RESOLVED,
        "scientific_config_sha256": SCIENTIFIC,
        "data_identity_sha256": DATA,
    }
    values.update(overrides)
    return launch_module._validate_g0_artifact(
        {"path": bundle["audit"].name, "sha256": _sha256(bundle["audit"])},
        bundle["audit"].parent,
        trust_root=bundle["trust_root"],
        **values,
    )


def test_signed_g0_pass_reproduces_and_binds_exact_launch(tmp_path):
    audit, path, digest = _validate(_g0_bundle(tmp_path))

    assert audit["status"] == "PASS"
    assert path.name == "audit.json"
    assert digest == _sha256(path)


def test_g0_kill_or_mismatched_launch_binding_cannot_authorize_profile(tmp_path):
    with pytest.raises(FullPetalLaunchError, match="has not reached PASS"):
        _validate(_g0_bundle(tmp_path / "kill", fail=True))

    valid = _g0_bundle(tmp_path / "mismatch")
    with pytest.raises(FullPetalLaunchError, match="bindings differ"):
        _validate(valid, data_identity_sha256="0" * 64)


def test_post_result_margin_file_substitution_breaks_the_signed_chain(tmp_path):
    bundle = _g0_bundle(tmp_path)
    payload = json.loads(bundle["margins"].read_text(encoding="utf-8"))
    payload["min_gradient_cosine"] = 0.0
    _write(bundle["margins"], payload)

    with pytest.raises(FullPetalLaunchError, match="hash mismatch"):
        _validate(bundle)


def test_loaded_dataset_bytes_override_self_consistent_false_provenance(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "config.py"
    config_path.write_text("fixture = True\n", encoding="utf-8")
    annotation = tmp_path / "annotations.json"
    annotation.write_text("{}\n", encoding="utf-8")
    spec = VideoSamplingSpec("video", 8, ())

    class Dataset:
        ann_file = annotation
        cache_manifest_sha256 = "2" * 64
        split_manifest_sha256 = "3" * 64
        split_manifest_file_sha256 = "4" * 64
        split_role = "fit_core"
        split_seed = 705

        @staticmethod
        def iter_crs_eps_sampling_specs():
            return (spec,)

    monkeypatch.setattr(
        gold_evidence_module,
        "build_data_identity",
        lambda cfg: {"identity_sha256": DATA},
    )
    monkeypatch.setattr(
        gold_evidence_module,
        "scientific_source_identity",
        lambda root, digest: SOURCE,
    )
    provenance = {
        "commit_sha": COMMIT,
        "config_sha256": _sha256(config_path),
        "resolved_config_sha256": RESOLVED,
        "scientific_config_sha256": SCIENTIFIC,
        "data_identity_sha256": DATA,
        "sampling_specs_sha256": sampling_specs_sha256([spec]),
        "annotation_sha256": "0" * 64,
        "feature_cache_manifest_sha256": Dataset.cache_manifest_sha256,
        "split_manifest_sha256": Dataset.split_manifest_sha256,
        "split_manifest_file_sha256": Dataset.split_manifest_file_sha256,
        "split_role": Dataset.split_role,
        "split_seed": Dataset.split_seed,
    }
    manifest = build_epoch_manifest(
        [spec], epoch=0, seed=705, draws_per_video=1, provenance=provenance
    )

    with pytest.raises(CrsEpsGoldEvidenceError, match="source bytes: annotation_sha256"):
        bind_manifest_to_loaded_dataset(
            manifest,
            Dataset(),
            cfg={},
            config_path=config_path,
            repository_root=tmp_path,
            commit_sha=COMMIT,
            resolved_config_sha256=RESOLVED,
            scientific_config_sha256=SCIENTIFIC,
        )
