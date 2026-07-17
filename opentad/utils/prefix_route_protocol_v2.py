"""Cryptographically bound, evidence-derived Prefix-Route Protocol V2 gates."""

from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_ssh_public_key

from .evidence_bundle import (
    read_stable_file_bytes,
    read_verified_bundle_bytes,
    strict_json_from_bytes,
)
from .prefix_route_r0_v2 import collect_r0_census, parse_class_map_bytes


PROTOCOL_SCHEMA = "prefix-route-identifiability-protocol-v2"
PROTOCOL_ID = "prefix-route-identifiability-20260717-v2"
MANIFEST_SCHEMA = "prefix-route-source-manifest-v2"
REVIEW_SCHEMA = "prefix-route-signed-review-attestation-v2"
POPULATION_REQUEST_SCHEMA = "prefix-route-population-request-v2"
ID_LIST_SCHEMA = "prefix-route-video-id-list-v2"
DIFFERENCE_REASON_SCHEMA = "prefix-route-population-reasons-v2"
R0_REQUEST_SCHEMA = "prefix-route-r0-request-v2"
R0_ENVELOPE_SCHEMA = "prefix-route-r0-evidence-envelope-v2"
EXPOSURE_LEDGER_SCHEMA = "prefix-route-exposure-ledger-entry-v2"

PROTOCOL_PASS = "PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION"
PROTOCOL_REVISE = "REVISE_PROTOCOL_BEFORE_COLLECTION"
REVIEW_SCOPE = "OUTCOME_BLIND_ZERO_GPU_PROTOCOL_ONLY"

REVIEWER_ID = "019f6f63-496d-75a0-a77b-91425a8e7ea1"
REVIEWER_PUBLIC_KEY = (
    "ssh-ed25519 "
    "AAAAC3NzaC1lZDI1NTE5AAAAIMAt+4hE9Lv4k0mMVWmkMckBmd2eIr4tTY8OHxEVulWV "
    "codex-reviewer:019f6f63-496d-75a0-a77b-91425a8e7ea1:protocol-v2"
)
REVIEWER_FINGERPRINT = "SHA256:g+0xrnbZxxkJtQl7TsiexfPeZMioIOX55arBxuJ8RrU"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_TOP_FIELDS = {
    "schema_version",
    "protocol_id",
    "status",
    "policy_lock_sha256",
    "task",
    "governance",
    "population",
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "evidence_package",
    "source_bindings",
}
_POLICY_SECTIONS = (
    "task",
    "governance",
    "population",
    "r0",
    "r1",
    "r2",
    "r3",
    "r4",
    "r5",
    "r6",
    "evidence_package",
)


class PrefixRouteProtocolV2Error(ValueError):
    """Raised when a V2 authorization or evidence derivation is not unique."""


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value, label):
    if not isinstance(value, dict):
        raise PrefixRouteProtocolV2Error(f"{label} must be an object")
    return value


def _sequence(value, label):
    if not isinstance(value, list):
        raise PrefixRouteProtocolV2Error(f"{label} must be an array")
    return value


def _exact(value, expected, label):
    value = _mapping(value, label)
    actual = set(value)
    expected = set(expected)
    if actual != expected:
        raise PrefixRouteProtocolV2Error(
            f"{label} fields differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise PrefixRouteProtocolV2Error(f"{label} must be non-empty text")
    return value


def _sha256(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PrefixRouteProtocolV2Error(f"{label} must be lowercase SHA-256")
    return value


def _git_sha1(value, label):
    if not isinstance(value, str) or not _GIT_SHA1.fullmatch(value):
        raise PrefixRouteProtocolV2Error(f"{label} must be lowercase 40-hex")
    return value


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteProtocolV2Error(f"{label} must be a positive integer")
    return value


def _timestamp(value, label):
    value = _nonempty(value, label)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PrefixRouteProtocolV2Error(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PrefixRouteProtocolV2Error(f"{label} lacks a timezone")
    return value


def _unique_strings(value, label, *, allow_empty=False):
    values = _sequence(value, label)
    if not allow_empty and not values:
        raise PrefixRouteProtocolV2Error(f"{label} may not be empty")
    if any(not isinstance(item, str) or not item for item in values):
        raise PrefixRouteProtocolV2Error(f"{label} contains invalid text")
    if len(values) != len(set(values)):
        raise PrefixRouteProtocolV2Error(f"{label} contains duplicates")
    return values


def _relative_repo_path(value, label):
    value = _nonempty(value, label)
    if "\\" in value:
        raise PrefixRouteProtocolV2Error(f"{label} must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PrefixRouteProtocolV2Error(f"{label} is not a canonical relative path")
    return value


def _reject_placeholders(value, path="protocol"):
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_placeholders(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_placeholders(nested, f"{path}[{index}]")
    elif isinstance(value, str):
        upper = value.upper()
        if any(token in upper for token in ("TODO", "TBD", "PLACEHOLDER")):
            raise PrefixRouteProtocolV2Error(f"{path} contains a placeholder")


def _require_canonical_json(payload, label):
    value = strict_json_from_bytes(payload, label, require_object=True)
    expected = canonical_json_bytes(value)
    if payload != expected:
        raise PrefixRouteProtocolV2Error(
            f"{label} must use canonical sorted UTF-8 JSON plus one LF"
        )
    return value


def _public_key_fingerprint(public_key):
    parts = public_key.split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        raise PrefixRouteProtocolV2Error("reviewer public key is not Ed25519")
    try:
        wire = base64.b64decode(parts[1], validate=True)
    except ValueError as exc:
        raise PrefixRouteProtocolV2Error("reviewer public key is invalid") from exc
    digest = base64.b64encode(hashlib.sha256(wire).digest()).decode("ascii").rstrip("=")
    return f"SHA256:{digest}"


def _validate_task(protocol):
    task = _exact(
        protocol["task"],
        {
            "name",
            "decision_input",
            "future_frame_access",
            "formal_emission_fields",
            "field_mapping",
            "commit_semantics",
            "task_scope_exclusions",
        },
        "task",
    )
    if task["name"] != (
        "fully_supervised_strict_causal_online_temporal_action_localization"
    ):
        raise PrefixRouteProtocolV2Error("task identity drifted")
    if task["future_frame_access"] is not False:
        raise PrefixRouteProtocolV2Error("future frames must remain forbidden")
    if task["formal_emission_fields"] != [
        "emission_id",
        "stream_key",
        "sequence_id",
        "start",
        "end",
        "class",
        "score",
        "source_frame",
        "emit_frame",
    ]:
        raise PrefixRouteProtocolV2Error("formal emission fields differ")
    if task["field_mapping"] != {
        "internal_decision_frame": "source_frame",
        "immutable_commit_frame": "emit_frame",
    }:
        raise PrefixRouteProtocolV2Error("formal evaluator field mapping differs")
    if task["commit_semantics"] != (
        "append_only_no_revision_no_merge_no_nms_no_offline_cleanup"
    ):
        raise PrefixRouteProtocolV2Error("commit semantics differ")
    if "future_ground_truth" not in task["task_scope_exclusions"]:
        raise PrefixRouteProtocolV2Error("future ground truth is not explicitly excluded")


def _validate_governance(protocol):
    governance = _exact(
        protocol["governance"],
        {
            "current_authorization",
            "blocked_before_signed_pass",
            "post_pass_scope",
            "pass_verdict",
            "revise_verdict",
            "no_automatic_escalation",
            "reviewer_trust",
        },
        "governance",
    )
    if governance["current_authorization"] != [
        "EDIT_PROTOCOL_FILES",
        "RUN_ZERO_GPU_PROTOCOL_TESTS",
        "REQUEST_SAME_SOLE_INDEPENDENT_REVIEWER",
    ]:
        raise PrefixRouteProtocolV2Error("current authorization differs")
    mandatory_blocks = {
        "NEW_R0_COLLECTION",
        "NEW_R1_COLLECTION",
        "MODEL_OUTCOME_INSPECTION",
        "MODEL_CONTRACT_FREEZE",
        "MODEL_IMPLEMENTATION",
        "GPU_PROFILE",
        "FORMAL_TRAINING",
        "VISUAL_FINETUNING",
        "RAW_VIDEO_TRAINING",
    }
    blocked = set(
        _unique_strings(
            governance["blocked_before_signed_pass"],
            "blocked_before_signed_pass",
        )
    )
    if blocked != mandatory_blocks:
        raise PrefixRouteProtocolV2Error("pre-PASS block set differs")
    if governance["post_pass_scope"] != [
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    ]:
        raise PrefixRouteProtocolV2Error("post-PASS scope differs")
    if governance["pass_verdict"] != PROTOCOL_PASS:
        raise PrefixRouteProtocolV2Error("PASS token differs")
    if governance["revise_verdict"] != PROTOCOL_REVISE:
        raise PrefixRouteProtocolV2Error("REVISE token differs")
    if governance["no_automatic_escalation"] is not True:
        raise PrefixRouteProtocolV2Error("automatic escalation must be forbidden")
    trust = _exact(
        governance["reviewer_trust"],
        {
            "reviewer_id",
            "review_task_id",
            "public_key_openssh",
            "public_key_fingerprint",
            "signature_algorithm",
            "signature_encoding",
            "signed_bytes",
            "minimum_reviewer_count",
        },
        "reviewer_trust",
    )
    if (
        trust["reviewer_id"] != REVIEWER_ID
        or trust["review_task_id"] != REVIEWER_ID
        or trust["public_key_openssh"] != REVIEWER_PUBLIC_KEY
        or trust["public_key_fingerprint"] != REVIEWER_FINGERPRINT
        or trust["signature_algorithm"] != "Ed25519"
        or trust["signature_encoding"] != "raw_signature_base64_lf"
        or trust["signed_bytes"] != "canonical_review_attestation_json_bytes"
        or trust["minimum_reviewer_count"] != 1
    ):
        raise PrefixRouteProtocolV2Error("reviewer trust root differs")
    if _public_key_fingerprint(trust["public_key_openssh"]) != (
        trust["public_key_fingerprint"]
    ):
        raise PrefixRouteProtocolV2Error("reviewer public-key fingerprint differs")


def _validate_population_policy(protocol):
    population = _exact(
        protocol["population"],
        {"dataset", "development", "reporting", "exposure_policy"},
        "population",
    )
    if population["dataset"] != "THUMOS14":
        raise PrefixRouteProtocolV2Error("dataset must remain THUMOS14")
    development = _exact(
        population["development"],
        {
            "universe_count",
            "fit_core_count",
            "calibration_count",
            "split_seed",
            "disjoint",
        },
        "development population",
    )
    if development != {
        "universe_count": 200,
        "fit_core_count": 160,
        "calibration_count": 40,
        "split_seed": 20260713,
        "disjoint": True,
    }:
        raise PrefixRouteProtocolV2Error("development population differs")
    reporting = _exact(
        population["reporting"],
        {
            "canonical_role",
            "canonical_count",
            "historical_role",
            "historical_count",
            "historical_primary_allowed",
            "required_status",
            "allowed_reason_codes",
            "unresolved_action",
        },
        "reporting population",
    )
    if (
        reporting["canonical_role"] != "canonical_reporting_213"
        or reporting["canonical_count"] != 213
        or reporting["historical_role"] != "historical_reporting_211_audit_only"
        or reporting["historical_count"] != 211
        or reporting["historical_primary_allowed"] is not False
        or reporting["required_status"] != "EXPLAINED_MISMATCH"
        or reporting["unresolved_action"] != "BLOCK_R0_POPULATION_UNRESOLVED"
    ):
        raise PrefixRouteProtocolV2Error("reporting population policy differs")
    _unique_strings(reporting["allowed_reason_codes"], "allowed reason codes")
    exposure = _exact(
        population["exposure_policy"],
        {
            "thumos_reporting_status",
            "annotation_unseen_claim_allowed",
            "model_outcome_blind_required",
            "final_confirmatory_claim_requires",
            "ledger_path",
            "ledger_append_only",
        },
        "exposure policy",
    )
    if (
        exposure["thumos_reporting_status"]
        != "DESIGN_EXPOSED_ROUTE_SELECTION_AND_BENCHMARK"
        or exposure["annotation_unseen_claim_allowed"] is not False
        or exposure["model_outcome_blind_required"] is not True
        or exposure["final_confirmatory_claim_requires"]
        != "SEPARATELY_FROZEN_ANNOTATION_UNSEEN_DATASET_BEFORE_MODEL_OUTCOMES"
        or exposure["ledger_append_only"] is not True
    ):
        raise PrefixRouteProtocolV2Error("exposure policy differs")
    _relative_repo_path(exposure["ledger_path"], "exposure ledger path")


def _validate_r0_r1(protocol):
    r0 = _exact(
        protocol["r0"],
        {
            "collector",
            "feature_stride_frames",
            "interval_preprocessing",
            "decision_coordinates",
            "pair_rules",
            "bootstrap",
            "claim_eligibility",
            "author_output",
            "status_derivation",
        },
        "r0",
    )
    if r0["collector"] != "opentad.utils.prefix_route_r0_v2.collect_r0_census":
        raise PrefixRouteProtocolV2Error("R0 collector binding differs")
    if r0["feature_stride_frames"] != 8:
        raise PrefixRouteProtocolV2Error("R0 feature stride differs")
    preprocessing = r0["interval_preprocessing"]
    required_preprocessing = {
        "interval": "half_open_[start_sec,end_sec)",
        "bounds": "require_0_le_start_lt_end_le_video_duration_no_clipping",
        "ambiguous_label": "exclude_and_report",
        "unknown_legal_label": "fail_source_identity",
        "malformed_or_nonfinite": "exclude_and_report_ambiguous",
        "exact_duplicate": "exclude_all_duplicate_group_members_as_ambiguous",
        "sort_key": [
            "start_frame",
            "end_observation_count",
            "class_index",
            "source_index",
        ],
    }
    if preprocessing != required_preprocessing:
        raise PrefixRouteProtocolV2Error("R0 interval preprocessing differs")
    if r0["decision_coordinates"] != {
        "first_previous_observation_count": 0,
        "start_frame": "floor(frame_count*start_sec/duration_sec)",
        "end_observation_count": "ceil(frame_count*end_sec/duration_sec)",
        "terminal_endpoint": "end_observation_count_equals_frame_count",
        "start_bin": "start_frame//8",
        "end_bin": "(end_observation_count-1)//8",
    }:
        raise PrefixRouteProtocolV2Error("R0 decision coordinates differ")
    if r0["pair_rules"] != {
        "overlap": "positive_half_open_intersection",
        "touching_is_overlap": False,
        "concurrency_tie": "end_before_start",
        "pair_counting": "each_unordered_pair_exactly_once",
        "same_bin_relation": (
            "earlier_end_instance_to_other_start_with_deterministic_tie_break"
        ),
        "same_bin_subtypes": ["handoff_or_touching", "overlap_transition"],
        "sequential_same_class": "consecutive_start_sorted_instances_only",
        "sequential_gap": "next_start-current_end",
    }:
        raise PrefixRouteProtocolV2Error("R0 pair rules differ")
    if r0["bootstrap"] != {
        "unit": "video",
        "method": "nonparametric_video_cluster_percentile_type7",
        "resamples": 10000,
        "seed": 2026071701,
        "confidence_level": 0.95,
        "same_indices_for_all_targets": True,
        "zero_denominator_action": "FAIL_INCOMPLETE",
    }:
        raise PrefixRouteProtocolV2Error("R0 bootstrap differs")
    eligibility = r0["claim_eligibility"]
    if eligibility != {
        "families": [
            "same_class_repetition",
            "same_class_overlap",
            "same_bin_end_start",
            "direct_complete",
        ],
        "primary_minimums": {
            "independent_videos": 30,
            "gt_instances": 100,
            "fraction_positive_videos": 0.05,
            "classes": 3,
            "power": 0.8,
        },
        "secondary_minimums": {
            "independent_videos": 10,
            "gt_instances": 30,
        },
        "power": {
            "family_wise_alpha": 0.05,
            "family_count": 4,
            "two_sided_alpha_each": 0.0125,
            "baseline_rate": 0.5,
            "target_rate": 0.6,
            "absolute_effect": 0.1,
            "intracluster_correlation": 0.2,
            "cluster_vector_includes_zero_videos": True,
            "cluster_cv_ddof": 0,
        },
    }:
        raise PrefixRouteProtocolV2Error("R0 claim eligibility differs")
    if (
        r0["author_output"] != "aggregate_only_no_video_ids"
        or r0["status_derivation"] != "recompute_from_source_bytes_never_trust_status"
    ):
        raise PrefixRouteProtocolV2Error("R0 disclosure or derivation differs")

    r1 = _exact(
        protocol["r1"],
        {
            "validator",
            "claim_scope",
            "excluded_scope",
            "required_source_objects",
            "support_rule",
            "dynamic_selection",
            "dynamic_mutations",
            "cache_linkage",
            "status_derivation",
        },
        "r1",
    )
    if r1["validator"] != (
        "opentad.utils.prefix_route_r1_v2.validate_r1_bundle"
    ):
        raise PrefixRouteProtocolV2Error("R1 validator binding differs")
    if r1["claim_scope"] != "decoded_rgb_frames_to_feature_token_only":
        raise PrefixRouteProtocolV2Error("R1 claim scope differs")
    if set(r1["excluded_scope"]) != {
        "compressed_bitstream_pts_dts_causality",
        "b_frame_decode_buffering",
        "wall_clock_decode_latency",
    }:
        raise PrefixRouteProtocolV2Error("R1 excluded scope differs")
    required_sources = set(r1["required_source_objects"])
    if required_sources != {
        "extractor_source",
        "resolved_command",
        "environment_lock",
        "software_versions",
        "hf_snapshot_revision",
        "model_config",
        "processor_config",
        "weight_shards",
        "annotation",
        "cache_manifest",
        "support_map",
        "dynamic_audit",
        "all_raw_videos",
        "all_feature_arrays",
    }:
        raise PrefixRouteProtocolV2Error("R1 source object set differs")
    if r1["support_rule"] != {
        "general": "max(support_frames)<=decision_frame",
        "current_path": "support_frames=[decision_frame]",
        "source_frame": "equals_decision_frame",
        "metadata_only_sufficient": False,
    }:
        raise PrefixRouteProtocolV2Error("R1 support rule differs")
    if r1["dynamic_selection"] != {
        "seed": 2026071702,
        "rank": "SHA256(video_id||token_index||seed||category)",
        "categories": [
            "first_token",
            "interior_token",
            "last_complete_bin",
            "last_partial_bin",
        ],
        "tokens_per_category": 32,
        "minimum_unique_videos": 16,
        "shortfall": "select_all_and_record",
    }:
        raise PrefixRouteProtocolV2Error("R1 dynamic selection differs")
    if r1["dynamic_mutations"] != {
        "future_frames": (
            "invert_native_future_suffix_else_"
            "paired_synthetic_continuation_after_eos"
        ),
        "future_suffix_modes": [
            "native_future_suffix",
            "paired_synthetic_continuation_after_eos",
        ],
        "other_batch_image": "invert_one_non_target_batch_image",
        "batch_order": "deterministic_reverse_permutation",
        "support_frame": "invert_uint8_rgb_at_decision_frame",
        "invariance": "exact_target_token_float32_bytes",
        "sensitivity": "target_token_linf_strictly_positive",
        "no_op_mutation": "FAIL_UNVERIFIABLE",
    }:
        raise PrefixRouteProtocolV2Error("R1 dynamic mutations differ")
    if r1["cache_linkage"] != {
        "key_sets": "raw_equals_feature_equals_support_equals_canonical_213",
        "read_and_hash_every_raw_video": True,
        "read_and_hash_every_feature_array": True,
        "load_npy_and_validate_geometry_dtype": True,
        "cross_environment_gpu_fp16_byte_match_required": False,
        "replacement_cache_automatically_authorized": False,
    }:
        raise PrefixRouteProtocolV2Error("R1 cache linkage differs")
    if r1["status_derivation"] != "computed_from_evidence_no_boolean_pass_inputs":
        raise PrefixRouteProtocolV2Error("R1 status derivation differs")


def _validate_r2_to_r6(protocol):
    r2 = _exact(
        protocol["r2"],
        {
            "arms",
            "b2_canonical_attack",
            "fairness_validator",
            "fairness",
            "threshold_calibration",
        },
        "r2",
    )
    if set(r2["arms"]) != {"B0", "B1", "B2", "B3", "B4"}:
        raise PrefixRouteProtocolV2Error("B0-B4 arm set differs")
    expected_arms = {
        "B0": {
            "identity": "fresh_queries_each_decision",
            "shared_causal_visual_memory": True,
            "training_assignment": "per_prefix_set_matching",
        },
        "B1": {
            "identity": "all_queries_always_persistent",
            "lifecycle": "ordinary_score_based_persistence",
            "route_specific_d1_d2": False,
        },
        "B2": {
            "identity": "newborn_plus_propagated_track_queries",
            "role": "faithful_temporal_motr_attack",
            "route_specific_d1_d2": False,
        },
        "B3": {
            "identity": "hard_model_only_lifecycle",
            "training_assignment": "hungarian_plus_dustbin",
        },
        "B4": {
            "identity": "prefix_shared_latent_event_filter",
            "route_specific_deltas": [
                "D1_ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
                "D2_CONTINUOUS_MASS_EPHEMERAL_NONCANONICAL_UOT",
            ],
        },
    }
    if r2["arms"] != expected_arms:
        raise PrefixRouteProtocolV2Error("B0-B4 arm definitions differ")
    if r2["arms"]["B4"]["route_specific_deltas"] != [
        "D1_ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
        "D2_CONTINUOUS_MASS_EPHEMERAL_NONCANONICAL_UOT",
    ]:
        raise PrefixRouteProtocolV2Error("B4 delta set differs")
    b2 = r2["b2_canonical_attack"]
    if b2 != {
        "query_pools": "newborn_object_queries_plus_propagated_track_queries",
        "newborn_availability": "present_at_every_decision_not_predecision_free_only",
        "propagation": "surviving_track_queries_propagate_across_decisions",
        "assignment": "temporal_tracklet_aware_one_to_one",
        "termination": "model_generated_track_score_and_endpoint_policy",
        "shared_heads_and_ledger": True,
        "route_specific_d1_d2_forbidden": True,
        "implementation_gate": (
            "source_cited_reconstruction_tests_pass_before_B4_comparison"
        ),
    }:
        raise PrefixRouteProtocolV2Error("B2 canonical attack differs")
    if r2["fairness_validator"] != (
        "opentad.utils.prefix_route_fairness_v2.derive_fairness_audit"
    ):
        raise PrefixRouteProtocolV2Error("fairness validator differs")
    fairness = r2["fairness"]
    if fairness != {
        "capacity_anchor": "B2",
        "trainable_parameter_relative_tolerance": 0.05,
        "train_macs_relative_tolerance": 0.1,
        "inference_macs_relative_tolerance": 0.1,
        "live_state_bytes_relative_tolerance": 0.1,
        "peak_training_memory_relative_tolerance": 0.1,
        "peak_inference_memory_relative_tolerance": 0.1,
        "latency_median_and_p95_upper_ratio": 1.1,
        "exact_equal": [
            "optimizer_event_count",
            "effective_token_count",
            "gradient_accumulation_steps",
            "hyperparameter_trial_count",
            "calibration_video_count",
            "seed_count",
        ],
        "unused_or_dummy_trainable_parameters": "FORBIDDEN",
        "every_trainable_parameter_requires_forward_use_and_smoke_gradient": True,
        "early_stopping": "FORBIDDEN",
        "nonfinite_or_skipped_update": "INVALIDATE_PAIRED_RUN_SET",
    }:
        raise PrefixRouteProtocolV2Error("arm fairness contract differs")
    calibration = r2["threshold_calibration"]
    if (
        calibration["population"] != "calibration_40"
        or calibration["model_selection_use"] is not False
        or calibration["selection_rule"]
        != (
            "highest_precision_subject_to_recall_ge_0.70_else_"
            "highest_recall_then_precision_then_higher_threshold"
        )
    ):
        raise PrefixRouteProtocolV2Error("threshold calibration differs")

    r3 = _exact(
        protocol["r3"],
        {"controls", "semantic_margin", "kill_rules"},
        "r3",
    )
    if set(r3["controls"]) != {
        "count_only",
        "template_timing",
        "feature_time_shuffle",
        "semantic_derangement",
        "ledger_only",
        "history_off",
    }:
        raise PrefixRouteProtocolV2Error("negative-control set differs")
    expected_nonsemantic_controls = {
        "count_only": {
            "feature_access": False,
            "sources": [
                "fit_core_event_count_prior",
                "fit_core_class_prior",
                "decision_bin_index",
            ],
        },
        "template_timing": {
            "feature_access": False,
            "sources": [
                "fit_core_duration_distribution",
                "fit_core_gap_distribution",
                "decision_bin_index",
            ],
        },
        "feature_time_shuffle": {
            "algorithm": "sha256_rank_then_positive_circular_token_rotation",
            "length_one": "exclude_and_report",
            "seed": 2026071703,
        },
        "ledger_only": {
            "input": "same_raw_B0_candidates",
            "learning": False,
            "operation": "shared_deterministic_ledger",
        },
        "history_off": {
            "clear_cross_decision_model_state": True,
            "keep_current_causal_visual_memory": True,
        },
    }
    for control, expected in expected_nonsemantic_controls.items():
        if r3["controls"][control] != expected:
            raise PrefixRouteProtocolV2Error(
                f"negative control differs: {control}"
            )
    semantic = r3["controls"]["semantic_derangement"]
    if semantic != {
        "scope": "within_one_frozen_evaluation_split",
        "seed": 2026071704,
        "algorithm": (
            "sha256_rank_then_max_frequency_multiset_rotation_with_non_global_swap"
        ),
        "preserve_global_label_multiset": True,
        "change_every_instance_label": True,
        "global_consistent_class_rename": "FORBIDDEN",
        "unconstructible": "FAIL_CONTROL_UNCONSTRUCTIBLE",
        "canonical_serialization": "sorted_compact_ascii_json_lf",
    }:
        raise PrefixRouteProtocolV2Error("semantic derangement differs")
    if r3["semantic_margin"] != {
        "metric": "class_ap",
        "required_absolute_drop": 0.05,
        "failure": "KILL_SEMANTIC_IDENTIFIABILITY_CLAIM",
    }:
        raise PrefixRouteProtocolV2Error("semantic control margin differs")
    if r3["kill_rules"] != {
        "feature_shuffle_no_timing_drop": "KILL_TEMPORAL_IDENTIFIABILITY_CLAIM",
        "semantic_drop_below_margin": "KILL_SEMANTIC_IDENTIFIABILITY_CLAIM",
        "simple_control_equivalent_to_strongest_arm": (
            "KILL_BENCHMARK_NOT_IDENTIFIABLE"
        ),
    }:
        raise PrefixRouteProtocolV2Error("negative-control kill rules differ")

    r4 = _exact(
        protocol["r4"],
        {"deletions", "d1_test", "d2_test", "necessity_rule"},
        "r4",
    )
    if set(r4["deletions"]) != {"A1", "A2", "A3", "A4", "A5", "A1_A2_JOINT"}:
        raise PrefixRouteProtocolV2Error("mechanism deletion set differs")
    if r4["deletions"] != {
        "A1": {
            "deleted": "uot_to_hungarian_plus_dustbin",
            "linked_delta": "D2",
        },
        "A2": {
            "deleted": "continuous_mass_to_binary_validity",
            "linked_delta": "D2",
        },
        "A3": {
            "deleted": "adjacent_prefix_consistency",
            "linked_delta": "D2_SUPPORT_ONLY",
        },
        "A4": {
            "deleted": "same_bin_reuse",
            "linked_delta": "D1_REQUIRED_ISOLATION",
        },
        "A5": {
            "deleted": "neural_transition_latch",
            "linked_delta": "D1_SUPPORT_ONLY",
        },
        "A1_A2_JOINT": {
            "deleted": "uot_and_continuous_mass",
            "linked_delta": "D2_REQUIRED_ISOLATION",
        },
    }:
        raise PrefixRouteProtocolV2Error("mechanism deletion definitions differ")
    if r4["d1_test"] != {
        "contrast": "B4_vs_A4_NO_SAME_BIN_REUSE",
        "target": "same_bin_end_start_recall",
    }:
        raise PrefixRouteProtocolV2Error("D1 isolation differs")
    if r4["d2_test"] != {
        "contrast": "B4_vs_A1_A2_JOINT",
        "targets": [
            "same_class_repetition_recall",
            "same_class_overlap_recall",
        ],
        "individual_A1_A2": "diagnostic_not_sufficient",
    }:
        raise PrefixRouteProtocolV2Error("D2 isolation differs")
    if r4["necessity_rule"] != (
        "paired_simultaneous_interval_beyond_margin_with_all_global_metrics_noninferior"
    ):
        raise PrefixRouteProtocolV2Error("mechanism necessity rule differs")

    r5 = _exact(
        protocol["r5"],
        {
            "generator",
            "sequence",
            "factor_levels",
            "sets",
            "valid_cross",
            "hidden_compound",
            "disjointness",
            "failure_actions",
        },
        "r5",
    )
    if r5["generator"] != (
        "opentad.utils.prefix_route_ood_v2.generate_sequence"
    ):
        raise PrefixRouteProtocolV2Error("R5 generator binding differs")
    if r5["sequence"] != {
        "length_bins": 64,
        "feature_dim": 16,
        "class_count": 4,
        "interval": "half_open_bins",
        "observation_equation": (
            "active_prototype_sum+birth_impulse+delayed_completion_cue+"
            "normalized_time+active_count_then_frozen_distribution"
        ),
        "float_serialization": "round_8_decimal_canonical_json",
    }:
        raise PrefixRouteProtocolV2Error("R5 sequence contract differs")
    required_families = {
        "event_topology",
        "temporal_geometry",
        "semantic_mapping",
        "observation_distribution",
    }
    if set(r5["factor_levels"]) != required_families:
        raise PrefixRouteProtocolV2Error("R5 factor families differ")
    if r5["factor_levels"] != {
        "event_topology": {
            "iid": [
                "empty",
                "single",
                "disjoint_pair",
                "disjoint_triple",
                "nested_pair",
                "partial_overlap_pair",
                "same_bin_handoff",
            ],
            "shift": [
                "disjoint_four",
                "disjoint_six",
                "chain_three",
                "clique_three",
            ],
        },
        "temporal_geometry": {
            "iid": [
                "duration_2_gap_0_delay_0",
                "duration_4_gap_2_delay_0",
                "duration_8_gap_8_delay_1",
                "duration_16_gap_2_delay_1",
            ],
            "shift": [
                "duration_1_gap_16_delay_2",
                "duration_32_gap_16_delay_4",
                "duration_8_gap_neg2_delay_2",
                "duration_16_gap_neg8_delay_4",
            ],
        },
        "semantic_mapping": {
            "iid": ["identity_prototype_to_class"],
            "shift": ["heldout_prototype_to_class_derangement"],
        },
        "observation_distribution": {
            "iid": ["identity", "gaussian_noise_sigma_0.05"],
            "shift": [
                "seeded_orthogonal_signed_permutation",
                "two_component_basis_mixture",
                "student_t_noise_df_3_scaled_0.05",
            ],
        },
    }:
        raise PrefixRouteProtocolV2Error("R5 factor levels differ")
    if r5["sets"] != {
        "TRAIN": {"count": 1000, "seed": 2026071705, "support": "iid"},
        "IID_HOLDOUT": {"count": 400, "seed": 2026071706, "support": "iid"},
        "SINGLE_SHIFT_OOD": {
            "count_per_family": 400,
            "seed_by_family": {
                "event_topology": 2026071708,
                "temporal_geometry": 2026071709,
                "semantic_mapping": 2026071710,
                "observation_distribution": 2026071711,
            },
            "support": "exactly_one_shifted_family",
        },
        "COMPOUND_OOD": {
            "count": 800,
            "combination_count": 8,
            "support": "at_least_two_shifted_families",
        },
    }:
        raise PrefixRouteProtocolV2Error("R5 set contract differs")
    if r5["valid_cross"] != {
        "enumerate_frozen_cartesian_cells": True,
        "reject_events_outside_64_bins": True,
        "round_robin_balancing": True,
        "minimum_and_maximum_cell_counts_reported": True,
        "no_posthoc_resampling": True,
    }:
        raise PrefixRouteProtocolV2Error("R5 valid-cross rules differ")
    if r5["disjointness"] != (
        "scientific_content_sha256_sets_pairwise_disjoint_excluding_set_metadata"
    ):
        raise PrefixRouteProtocolV2Error("R5 disjointness differs")
    if r5["hidden_compound"] != {
        "owner": "same_independent_reviewer",
        "seed_commitment": "SHA256(protocol_sha256||reviewer_secret||COMPOUND_OOD)",
        "signed_commitment_before_model_implementation": True,
        "combination_table_reveal": "after_all_arm_artifacts_hashed",
    }:
        raise PrefixRouteProtocolV2Error("R5 hidden compound contract differs")
    if r5["failure_actions"] != {
        "invalid_cross": "FAIL_R5_GENERATION",
        "cell_balance_mismatch": "FAIL_R5_GENERATION",
        "set_hash_overlap": "FAIL_R5_GENERATION",
    }:
        raise PrefixRouteProtocolV2Error("R5 failure actions differ")

    r6 = _exact(
        protocol["r6"],
        {
            "implementation",
            "metric_sources",
            "global_metrics",
            "stress_metrics",
            "registered_arms",
            "registered_contrasts",
            "paired_inference",
            "margins",
            "b4_survival",
            "terminal_statuses",
        },
        "r6",
    )
    if r6["implementation"] != (
        "opentad.evaluations.prefix_route_r6_v2.paired_crossed_bootstrap"
    ):
        raise PrefixRouteProtocolV2Error("R6 implementation differs")
    if r6["metric_sources"] != {
        "field_adapter": (
            "opentad.evaluations.prefix_route_r6_v2.canonical_emission"
        ),
        "identity_and_lifecycle": (
            "opentad.evaluations.full_petal_metrics.compute_full_petal_metrics"
        ),
        "mOnlineAP": (
            "opentad.evaluations.online_budgeted_map.OnlineAPBudgeted"
        ),
        "per_video_cell": (
            "opentad.evaluations.prefix_route_r6_v2.derive_per_video_cell"
        ),
        "stress_membership": "source_derived_R0_ground_truth_ID_membership",
    }:
        raise PrefixRouteProtocolV2Error("R6 metric source bindings differ")
    if r6["global_metrics"] != [
        "mOnlineAP",
        "event_recall",
        "false_emission_per_video",
        "duplicate_per_gt",
        "fragmentation_per_gt",
        "endpoint_latency_bins",
    ]:
        raise PrefixRouteProtocolV2Error("R6 global metrics differ")
    if r6["stress_metrics"] != [
        "same_class_repetition_recall",
        "same_class_overlap_recall",
        "same_bin_end_start_recall",
        "direct_complete_recall",
    ]:
        raise PrefixRouteProtocolV2Error("R6 stress metrics differ")
    if r6["registered_arms"] != [
        "B2",
        "B3",
        "B4",
        "A1_NO_UOT",
        "A2_BINARY_MASS",
        "A1_A2_JOINT",
        "A3_NO_CONSISTENCY",
        "A4_NO_SAME_BIN_REUSE",
        "A5_NO_NEURAL_LATCH",
    ]:
        raise PrefixRouteProtocolV2Error("R6 registered arms differ")
    if r6["registered_contrasts"] != [
        "B4_vs_B2",
        "B4_vs_B3",
        "B4_vs_A1_NO_UOT",
        "B4_vs_A2_BINARY_MASS",
        "B4_vs_A1_A2_JOINT",
        "B4_vs_A3_NO_CONSISTENCY",
        "B4_vs_A4_NO_SAME_BIN_REUSE",
        "B4_vs_A5_NO_NEURAL_LATCH",
    ]:
        raise PrefixRouteProtocolV2Error("R6 registered contrasts differ")
    if r6["paired_inference"] != {
        "fixed_reporting_population_estimand": True,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 2026071707,
        "video_resample": "global_paired_indices_shared_by_all_arms",
        "seed_resample": (
            "global_paired_seed_indices_shared_across_all_videos_and_arms"
        ),
        "mOnlineAP_unit": "paired_global_training_seed",
        "additive_metric_units": ["video", "paired_global_training_seed"],
        "simultaneous_interval": "bonferroni_paired_percentile_type7",
        "family_wise_alpha": 0.05,
        "multiplicity_denominator": "all_registered_contrasts_times_all_eligible_metrics",
    }:
        raise PrefixRouteProtocolV2Error("R6 inference contract differs")
    if r6["b4_survival"] != {
        "B4_vs_B2": "all_global_metrics_noninferior",
        "D1": (
            "B4_vs_A4_same_bin_recall_improves_beyond_margin_"
            "and_global_noninferior"
        ),
        "D2": (
            "B4_vs_A1_A2_joint_repetition_or_overlap_recall_improves_"
            "beyond_margin_and_global_noninferior"
        ),
        "mechanism_requirement": "D1_or_D2_must_be_established",
        "controls": "negative_controls_not_equivalent",
    }:
        raise PrefixRouteProtocolV2Error("B4 survival rule differs")
    if r6["margins"] != {
        "mOnlineAP": 0.005,
        "event_recall": "max(0.01,2/N_legal_gt)",
        "false_emission_per_video": "max(0.01,2/N_reporting_videos)",
        "duplicate_per_gt": "max(0.01,2/N_legal_gt)",
        "fragmentation_per_gt": "max(0.01,2/N_legal_gt)",
        "eligible_stress_recall": "max(0.02,2/N_stress_gt)",
        "endpoint_latency_bins": 1.0,
    }:
        raise PrefixRouteProtocolV2Error("R6 practical margins differ")
    if set(r6["terminal_statuses"]) != {
        "PASS_B4_ROUTE_SURVIVES",
        "KILL_TEMPORAL_MOTR_INFERIOR",
        "KILL_D1_D2_NOT_ESTABLISHED",
        "INDETERMINATE_NO_ROUTE_CLAIM",
        "KILL_BENCHMARK_NOT_IDENTIFIABLE",
    }:
        raise PrefixRouteProtocolV2Error("R6 terminal statuses differ")


def validate_protocol(protocol):
    """Validate all protected semantics and the policy lock."""

    protocol = _exact(protocol, _TOP_FIELDS, "protocol")
    if protocol["schema_version"] != PROTOCOL_SCHEMA:
        raise PrefixRouteProtocolV2Error("protocol schema differs")
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise PrefixRouteProtocolV2Error("protocol ID differs")
    if protocol["status"] != "PROTOCOL_REVIEW_PENDING":
        raise PrefixRouteProtocolV2Error("protocol status differs")
    _reject_placeholders(protocol)
    locked = {section: protocol[section] for section in _POLICY_SECTIONS}
    if canonical_sha256(locked) != _sha256(
        protocol["policy_lock_sha256"],
        "policy lock",
    ):
        raise PrefixRouteProtocolV2Error("policy lock differs")
    _validate_task(protocol)
    _validate_governance(protocol)
    _validate_population_policy(protocol)
    _validate_r0_r1(protocol)
    _validate_r2_to_r6(protocol)
    evidence = _exact(
        protocol["evidence_package"],
        {
            "required_before_model_contract",
            "forbidden_before_model_contract",
            "route_review_verdicts",
            "route_pass_authorizes_gpu",
        },
        "evidence package",
    )
    if evidence["route_pass_authorizes_gpu"] is not False:
        raise PrefixRouteProtocolV2Error("route PASS must not authorize GPU")
    if evidence["required_before_model_contract"] != [
        "signed_protocol_review_attestation",
        "source_derived_reporting_population",
        "source_derived_r0_aggregate_report",
        "source_derived_r1_cache_audit",
        "frozen_b0_b4_fairness_contract",
        "executable_negative_controls",
        "executable_structural_ood_generator",
        "executable_r6_inference_and_terminal_decision",
    ]:
        raise PrefixRouteProtocolV2Error("pre-model evidence package differs")
    if evidence["forbidden_before_model_contract"] != [
        "model_predictions",
        "checkpoints",
        "effectiveness_tables",
        "gpu_profiles",
        "formal_training_outputs",
    ]:
        raise PrefixRouteProtocolV2Error("pre-model forbidden object set differs")
    if evidence["route_review_verdicts"] != [
        "PASS_AUTHORIZE_MODEL_P0_CONTRACT_FREEZE",
        "REVISE_ROUTE_AND_REVIEW_AGAIN",
        "KILL_PERSISTENT_CARRIER_ROUTE",
    ]:
        raise PrefixRouteProtocolV2Error("route review verdict set differs")
    sources = _exact(
        protocol["source_bindings"],
        {
            "source_manifest_path",
            "required_paths",
            "v1_revise_review_path",
            "v1_revise_review_sha256",
        },
        "source bindings",
    )
    _relative_repo_path(sources["source_manifest_path"], "source manifest path")
    required_paths = _unique_strings(
        sources["required_paths"],
        "required source paths",
    )
    if required_paths != sorted(required_paths):
        raise PrefixRouteProtocolV2Error("required source paths must be sorted")
    mandatory_sources = {
        ".gitattributes",
        "opentad/utils/evidence_bundle.py",
        "opentad/utils/prefix_route_protocol_v2.py",
        "opentad/utils/prefix_route_r0_v2.py",
        "opentad/utils/prefix_route_r1_v2.py",
        "opentad/utils/prefix_route_controls_v2.py",
        "opentad/utils/prefix_route_fairness_v2.py",
        "opentad/utils/prefix_route_ood_v2.py",
        "opentad/evaluations/prefix_route_r6_v2.py",
        "opentad/evaluations/full_petal_metrics.py",
        "opentad/evaluations/online_budgeted_map.py",
        "tools/cache_ontad_features.py",
        "tests/test_prefix_route_protocol_v2.py",
    }
    if not mandatory_sources.issubset(required_paths):
        raise PrefixRouteProtocolV2Error("validator dependencies are not all bound")
    _relative_repo_path(sources["v1_revise_review_path"], "V1 review path")
    _sha256(sources["v1_revise_review_sha256"], "V1 review hash")
    return protocol


def load_protocol(path):
    resolved, payload = read_stable_file_bytes(path, "Protocol V2")
    protocol = _require_canonical_json(payload, "Protocol V2")
    validate_protocol(protocol)
    return {
        "path": resolved,
        "bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "protocol": protocol,
    }


def load_source_manifest(path, *, protocol_record, repo_root, check_worktree=True):
    resolved, payload = read_stable_file_bytes(path, "source manifest")
    manifest = _require_canonical_json(payload, "source manifest")
    _exact(
        manifest,
        {"schema_version", "protocol_id", "protocol_sha256", "entries"},
        "source manifest",
    )
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise PrefixRouteProtocolV2Error("source manifest schema differs")
    if manifest["protocol_id"] != protocol_record["protocol"]["protocol_id"]:
        raise PrefixRouteProtocolV2Error("source manifest protocol ID differs")
    if manifest["protocol_sha256"] != protocol_record["sha256"]:
        raise PrefixRouteProtocolV2Error("source manifest protocol hash differs")
    entries = _mapping(manifest["entries"], "source manifest entries")
    required = protocol_record["protocol"]["source_bindings"]["required_paths"]
    if sorted(entries) != required:
        raise PrefixRouteProtocolV2Error("source manifest path set differs")
    repo_root = Path(repo_root).resolve()
    for relative_path, digest in entries.items():
        _relative_repo_path(relative_path, "manifest source path")
        _sha256(digest, f"manifest source hash {relative_path}")
        if check_worktree:
            path = repo_root.joinpath(*PurePosixPath(relative_path).parts)
            _, source_bytes = read_stable_file_bytes(
                path,
                f"manifest source {relative_path}",
            )
            actual = hashlib.sha256(source_bytes).hexdigest()
            if actual != digest:
                raise PrefixRouteProtocolV2Error(
                    f"worktree source hash differs: {relative_path}"
                )
    return {
        "path": resolved,
        "bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "manifest": manifest,
    }


def _git(repo_root, *args):
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrefixRouteProtocolV2Error(
            f"git command failed: git {' '.join(args)}"
        ) from exc


def verify_frozen_git_tree(
    *,
    repo_root,
    commit,
    tree_sha1,
    protocol_record,
    manifest_record,
    require_head=True,
):
    """Verify exact commit, tree, manifest, protocol, paths, hashes, and blob modes."""

    repo_root = Path(repo_root).resolve()
    commit = _git_sha1(commit, "review commit")
    tree_sha1 = _git_sha1(tree_sha1, "review tree")
    resolved_commit = _git(repo_root, "rev-parse", commit).decode().strip()
    if resolved_commit != commit:
        raise PrefixRouteProtocolV2Error("review commit does not resolve exactly")
    if require_head:
        head = _git(repo_root, "rev-parse", "HEAD").decode().strip()
        if head != commit:
            raise PrefixRouteProtocolV2Error("review commit is not current HEAD")
    actual_tree = _git(repo_root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    if actual_tree != tree_sha1:
        raise PrefixRouteProtocolV2Error("review tree hash differs")
    protocol_relative = protocol_record["path"].relative_to(repo_root).as_posix()
    manifest_relative = manifest_record["path"].relative_to(repo_root).as_posix()
    for relative, expected, label in (
        (protocol_relative, protocol_record["sha256"], "protocol"),
        (manifest_relative, manifest_record["sha256"], "source manifest"),
    ):
        payload = _git(repo_root, "show", f"{commit}:{relative}")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise PrefixRouteProtocolV2Error(
                f"{label} bytes differ at reviewed commit"
            )
    for relative, expected in manifest_record["manifest"]["entries"].items():
        listing = _git(repo_root, "ls-tree", commit, "--", relative).decode().strip()
        if not listing:
            raise PrefixRouteProtocolV2Error(
                f"manifest path is absent at reviewed commit: {relative}"
            )
        mode = listing.split(maxsplit=1)[0]
        if mode not in {"100644", "100755"}:
            raise PrefixRouteProtocolV2Error(
                f"manifest path is not a regular blob: {relative}"
            )
        payload = _git(repo_root, "show", f"{commit}:{relative}")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise PrefixRouteProtocolV2Error(
                f"manifest source differs at reviewed commit: {relative}"
            )
    return True


def load_signed_review(
    *,
    attestation_path,
    signature_path,
    protocol_record,
    manifest_record,
    repo_root,
    require_head=True,
):
    """Verify the sole reviewer's detached signature and every frozen binding."""

    attestation_resolved, attestation_bytes = read_stable_file_bytes(
        attestation_path,
        "signed review attestation",
    )
    attestation = _require_canonical_json(
        attestation_bytes,
        "signed review attestation",
    )
    _exact(
        attestation,
        {
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "protocol_commit",
            "protocol_tree_sha1",
            "source_manifest_sha256",
            "reviewer_id",
            "review_task_id",
            "reviewed_at",
            "verdict",
            "scope",
            "findings",
            "commands",
            "test_summary",
            "no_prohibited_access",
        },
        "signed review attestation",
    )
    if attestation["schema_version"] != REVIEW_SCHEMA:
        raise PrefixRouteProtocolV2Error("review attestation schema differs")
    if attestation["protocol_id"] != protocol_record["protocol"]["protocol_id"]:
        raise PrefixRouteProtocolV2Error("review protocol ID differs")
    if attestation["protocol_sha256"] != protocol_record["sha256"]:
        raise PrefixRouteProtocolV2Error("review protocol hash differs")
    if attestation["source_manifest_sha256"] != manifest_record["sha256"]:
        raise PrefixRouteProtocolV2Error("review source manifest hash differs")
    if (
        attestation["reviewer_id"] != REVIEWER_ID
        or attestation["review_task_id"] != REVIEWER_ID
    ):
        raise PrefixRouteProtocolV2Error("reviewer identity differs")
    _timestamp(attestation["reviewed_at"], "reviewed_at")
    if attestation["verdict"] not in {PROTOCOL_PASS, PROTOCOL_REVISE}:
        raise PrefixRouteProtocolV2Error("review verdict differs")
    if attestation["scope"] != REVIEW_SCOPE:
        raise PrefixRouteProtocolV2Error("review scope differs")
    if not isinstance(attestation["findings"], list):
        raise PrefixRouteProtocolV2Error("review findings must be an array")
    if not isinstance(attestation["commands"], list):
        raise PrefixRouteProtocolV2Error("review commands must be an array")
    if not isinstance(attestation["test_summary"], dict):
        raise PrefixRouteProtocolV2Error("review test summary must be an object")
    if attestation["no_prohibited_access"] is not True:
        raise PrefixRouteProtocolV2Error("review accessed prohibited evidence")
    if attestation["verdict"] == PROTOCOL_REVISE and not attestation["findings"]:
        raise PrefixRouteProtocolV2Error("REVISE review lacks findings")
    commit = _git_sha1(attestation["protocol_commit"], "review commit")
    tree = _git_sha1(attestation["protocol_tree_sha1"], "review tree")

    signature_resolved, signature_bytes = read_stable_file_bytes(
        signature_path,
        "review signature",
    )
    if not signature_bytes.endswith(b"\n") or signature_bytes.count(b"\n") != 1:
        raise PrefixRouteProtocolV2Error("review signature encoding differs")
    try:
        signature = base64.b64decode(signature_bytes[:-1], validate=True)
    except ValueError as exc:
        raise PrefixRouteProtocolV2Error("review signature is not base64") from exc
    if base64.b64encode(signature) + b"\n" != signature_bytes:
        raise PrefixRouteProtocolV2Error("review signature is not canonical base64")
    if len(signature) != 64:
        raise PrefixRouteProtocolV2Error("Ed25519 signature length differs")
    public_key = load_ssh_public_key(REVIEWER_PUBLIC_KEY.encode("ascii"))
    if not isinstance(public_key, Ed25519PublicKey):
        raise PrefixRouteProtocolV2Error("review trust root is not Ed25519")
    try:
        public_key.verify(signature, attestation_bytes)
    except InvalidSignature as exc:
        raise PrefixRouteProtocolV2Error("review signature verification failed") from exc
    verify_frozen_git_tree(
        repo_root=repo_root,
        commit=commit,
        tree_sha1=tree,
        protocol_record=protocol_record,
        manifest_record=manifest_record,
        require_head=require_head,
    )
    return {
        "attestation_path": attestation_resolved,
        "attestation_bytes": attestation_bytes,
        "attestation_sha256": hashlib.sha256(attestation_bytes).hexdigest(),
        "signature_path": signature_resolved,
        "signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
        "attestation": attestation,
    }


def authorize_collection(
    *,
    protocol_path,
    manifest_path,
    repo_root,
    attestation_path=None,
    signature_path=None,
):
    """The only public authorization path; it never accepts prevalidated dictionaries."""

    protocol_record = load_protocol(protocol_path)
    manifest_record = load_source_manifest(
        manifest_path,
        protocol_record=protocol_record,
        repo_root=repo_root,
        check_worktree=True,
    )
    if attestation_path is None or signature_path is None:
        return {
            "authorized": False,
            "status": "BLOCKED_PENDING_SIGNED_INDEPENDENT_PROTOCOL_REVIEW",
            "allowed": [],
        }
    review = load_signed_review(
        attestation_path=attestation_path,
        signature_path=signature_path,
        protocol_record=protocol_record,
        manifest_record=manifest_record,
        repo_root=repo_root,
        require_head=True,
    )
    if review["attestation"]["verdict"] != PROTOCOL_PASS:
        return {
            "authorized": False,
            "status": PROTOCOL_REVISE,
            "allowed": [],
        }
    return {
        "authorized": True,
        "status": "AUTHORIZED_OUTCOME_BLIND_R0_R1_COLLECTION_ONLY",
        "allowed": list(
            protocol_record["protocol"]["governance"]["post_pass_scope"]
        ),
        "review_attestation_sha256": review["attestation_sha256"],
        "protocol_sha256": protocol_record["sha256"],
        "protocol_commit": review["attestation"]["protocol_commit"],
        "blocked": sorted(
            set(
                protocol_record["protocol"]["governance"][
                    "blocked_before_signed_pass"
                ]
            )
            - {"NEW_R0_COLLECTION", "NEW_R1_COLLECTION"}
        ),
    }


def _read_canonical_bundle_json(reference, bundle_root, label):
    _, payload = read_verified_bundle_bytes(reference, bundle_root, label)
    return payload, _require_canonical_json(payload, label)


def _parse_id_list(value, role, expected_count):
    _exact(value, {"schema_version", "role", "ids"}, f"{role} ID list")
    if value["schema_version"] != ID_LIST_SCHEMA or value["role"] != role:
        raise PrefixRouteProtocolV2Error(f"{role} ID-list identity differs")
    ids = _unique_strings(value["ids"], f"{role} IDs")
    if ids != sorted(ids):
        raise PrefixRouteProtocolV2Error(f"{role} IDs must be sorted")
    if len(ids) != expected_count:
        raise PrefixRouteProtocolV2Error(f"{role} ID count differs")
    return ids


def validate_population_bundle(
    request,
    *,
    bundle_root,
    protocol_record,
    review_record,
):
    """Re-read ID lists and reason sources, then derive the 211/213 decision."""

    _exact(
        request,
        {
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "review_attestation_sha256",
            "historical_ids",
            "canonical_ids",
            "difference_reasons",
            "reason_sources",
        },
        "population request",
    )
    if request["schema_version"] != POPULATION_REQUEST_SCHEMA:
        raise PrefixRouteProtocolV2Error("population request schema differs")
    if request["protocol_id"] != protocol_record["protocol"]["protocol_id"]:
        raise PrefixRouteProtocolV2Error("population protocol ID differs")
    if request["protocol_sha256"] != protocol_record["sha256"]:
        raise PrefixRouteProtocolV2Error("population protocol hash differs")
    if request["review_attestation_sha256"] != (
        review_record["attestation_sha256"]
    ):
        raise PrefixRouteProtocolV2Error("population review binding differs")
    reporting = protocol_record["protocol"]["population"]["reporting"]
    historical_bytes, historical_value = _read_canonical_bundle_json(
        request["historical_ids"],
        bundle_root,
        "historical ID list",
    )
    canonical_bytes, canonical_value = _read_canonical_bundle_json(
        request["canonical_ids"],
        bundle_root,
        "canonical ID list",
    )
    historical = _parse_id_list(
        historical_value,
        reporting["historical_role"],
        reporting["historical_count"],
    )
    canonical = _parse_id_list(
        canonical_value,
        reporting["canonical_role"],
        reporting["canonical_count"],
    )
    reason_bytes, reason_value = _read_canonical_bundle_json(
        request["difference_reasons"],
        bundle_root,
        "population difference reasons",
    )
    _exact(
        reason_value,
        {"schema_version", "reasons"},
        "population difference reasons",
    )
    if reason_value["schema_version"] != DIFFERENCE_REASON_SCHEMA:
        raise PrefixRouteProtocolV2Error("population reason schema differs")
    historical_only = sorted(set(historical) - set(canonical))
    canonical_only = sorted(set(canonical) - set(historical))
    difference_ids = historical_only + canonical_only
    if not difference_ids:
        raise PrefixRouteProtocolV2Error("211/213 mismatch has no differing IDs")
    reason_rows = reason_value["reasons"]
    if not isinstance(reason_rows, list):
        raise PrefixRouteProtocolV2Error("population reasons must be an array")
    if [row.get("video_id") for row in reason_rows if isinstance(row, dict)] != (
        sorted(
            row.get("video_id")
            for row in reason_rows
            if isinstance(row, dict)
        )
    ):
        raise PrefixRouteProtocolV2Error(
            "population reason rows must be sorted by video ID"
        )
    reasons = {}
    source_keys = set()
    allowed_codes = set(reporting["allowed_reason_codes"])
    for row in reason_rows:
        _exact(
            row,
            {"video_id", "side", "reason_code", "source_key"},
            "population reason row",
        )
        video_id = _nonempty(row["video_id"], "population reason video ID")
        if video_id in reasons:
            raise PrefixRouteProtocolV2Error("duplicate population reason")
        expected_side = (
            "historical_only"
            if video_id in historical_only
            else "canonical_only"
            if video_id in canonical_only
            else None
        )
        if row["side"] != expected_side:
            raise PrefixRouteProtocolV2Error("population reason side differs")
        if row["reason_code"] not in allowed_codes:
            raise PrefixRouteProtocolV2Error("population reason code is forbidden")
        source_key = _nonempty(row["source_key"], "reason source key")
        source_keys.add(source_key)
        reasons[video_id] = dict(row)
    if sorted(reasons) != sorted(difference_ids):
        raise PrefixRouteProtocolV2Error("population reasons do not cover differences")
    reason_sources = _mapping(request["reason_sources"], "reason sources")
    if set(reason_sources) != source_keys:
        raise PrefixRouteProtocolV2Error("population reason-source set differs")
    reason_source_hashes = {}
    for source_key, reference in sorted(reason_sources.items()):
        _, payload = read_verified_bundle_bytes(
            reference,
            bundle_root,
            f"population reason source {source_key}",
        )
        if not payload:
            raise PrefixRouteProtocolV2Error("population reason source is empty")
        reason_source_hashes[source_key] = hashlib.sha256(payload).hexdigest()
    if len(historical) - len(historical_only) + len(canonical_only) != len(canonical):
        raise PrefixRouteProtocolV2Error("211/213 cardinality does not reconcile")
    result = {
        "schema_version": "prefix-route-population-derived-v2",
        "status": "EXPLAINED_MISMATCH",
        "historical_role": reporting["historical_role"],
        "canonical_role": reporting["canonical_role"],
        "historical_count": len(historical),
        "canonical_count": len(canonical),
        "historical_ids_sha256": hashlib.sha256(historical_bytes).hexdigest(),
        "canonical_ids_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "difference_reasons_sha256": hashlib.sha256(reason_bytes).hexdigest(),
        "reason_source_sha256": reason_source_hashes,
        "historical_only": historical_only,
        "canonical_only": canonical_only,
        "canonical_ids": canonical,
        "reasons": reasons,
    }
    result["derived_sha256"] = canonical_sha256(result)
    return result


def validate_exposure_ledger_bytes(payload, *, frozen_prefix=None):
    """Validate canonical JSONL hash chaining and append-only prefix preservation."""

    if not isinstance(payload, bytes) or not payload:
        raise PrefixRouteProtocolV2Error("exposure ledger is empty")
    if frozen_prefix is not None and not payload.startswith(frozen_prefix):
        raise PrefixRouteProtocolV2Error("exposure ledger changed its frozen prefix")
    if not payload.endswith(b"\n"):
        raise PrefixRouteProtocolV2Error("exposure ledger lacks final LF")
    lines = payload.splitlines(keepends=True)
    previous = "0" * 64
    entries = []
    for index, line in enumerate(lines):
        value = _require_canonical_json(line, f"exposure ledger line {index}")
        _exact(
            value,
            {
                "schema_version",
                "index",
                "previous_entry_sha256",
                "recorded_at",
                "actor_id",
                "dataset",
                "population",
                "exposure_type",
                "scope",
                "source_commit",
                "details",
            },
            f"exposure ledger line {index}",
        )
        if value["schema_version"] != EXPOSURE_LEDGER_SCHEMA:
            raise PrefixRouteProtocolV2Error("exposure ledger schema differs")
        if value["index"] != index:
            raise PrefixRouteProtocolV2Error("exposure ledger index differs")
        if value["previous_entry_sha256"] != previous:
            raise PrefixRouteProtocolV2Error("exposure ledger hash chain differs")
        _timestamp(value["recorded_at"], "exposure recorded_at")
        _nonempty(value["actor_id"], "exposure actor")
        _nonempty(value["dataset"], "exposure dataset")
        _nonempty(value["population"], "exposure population")
        _nonempty(value["exposure_type"], "exposure type")
        _nonempty(value["scope"], "exposure scope")
        _git_sha1(value["source_commit"], "exposure source commit")
        _mapping(value["details"], "exposure details")
        previous = hashlib.sha256(line).hexdigest()
        entries.append(value)
    if not any(
        entry["exposure_type"] == "PLANNED_R0_AGGREGATE_CENSUS"
        for entry in entries
    ):
        raise PrefixRouteProtocolV2Error(
            "exposure ledger lacks planned R0 disclosure"
        )
    return {
        "entry_count": len(entries),
        "ledger_sha256": hashlib.sha256(payload).hexdigest(),
        "last_entry_sha256": previous,
        "entries": entries,
    }


def derive_r0_envelope(
    *,
    annotation_bytes,
    class_map_bytes,
    exposure_ledger_bytes,
    population_record,
    protocol_record,
    review_record,
    frozen_ledger_prefix=None,
):
    annotation = strict_json_from_bytes(
        annotation_bytes,
        "R0 annotation",
        require_object=True,
    )
    class_names = parse_class_map_bytes(class_map_bytes)
    validate_exposure_ledger_bytes(
        exposure_ledger_bytes,
        frozen_prefix=frozen_ledger_prefix,
    )
    report, _ = collect_r0_census(
        annotation,
        class_names,
        population_record["canonical_ids"],
        feature_stride_frames=8,
        bootstrap_resamples=10000,
        bootstrap_seed=2026071701,
    )
    envelope = {
        "schema_version": R0_ENVELOPE_SCHEMA,
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": review_record["attestation_sha256"],
        "population_derived_sha256": population_record["derived_sha256"],
        "source_sha256": {
            "annotation": hashlib.sha256(annotation_bytes).hexdigest(),
            "class_map": hashlib.sha256(class_map_bytes).hexdigest(),
            "exposure_ledger": hashlib.sha256(exposure_ledger_bytes).hexdigest(),
        },
        "report": report,
        "status": "PASS_R0_COMPLETE",
    }
    envelope["derived_sha256"] = canonical_sha256(envelope)
    return envelope


def validate_r0_bundle(
    request,
    *,
    bundle_root,
    protocol_record,
    review_record,
    population_record,
    frozen_ledger_prefix=None,
):
    """Recompute the entire R0 envelope and compare canonical bytes exactly."""

    _exact(
        request,
        {
            "schema_version",
            "protocol_id",
            "protocol_sha256",
            "review_attestation_sha256",
            "population_derived_sha256",
            "annotation",
            "class_map",
            "exposure_ledger",
            "author_report",
        },
        "R0 request",
    )
    if request["schema_version"] != R0_REQUEST_SCHEMA:
        raise PrefixRouteProtocolV2Error("R0 request schema differs")
    expected_bindings = {
        "protocol_id": protocol_record["protocol"]["protocol_id"],
        "protocol_sha256": protocol_record["sha256"],
        "review_attestation_sha256": review_record["attestation_sha256"],
        "population_derived_sha256": population_record["derived_sha256"],
    }
    for field, expected in expected_bindings.items():
        if request[field] != expected:
            raise PrefixRouteProtocolV2Error(f"R0 {field} differs")
    _, annotation_bytes = read_verified_bundle_bytes(
        request["annotation"],
        bundle_root,
        "R0 annotation",
    )
    _, class_map_bytes = read_verified_bundle_bytes(
        request["class_map"],
        bundle_root,
        "R0 class map",
    )
    _, ledger_bytes = read_verified_bundle_bytes(
        request["exposure_ledger"],
        bundle_root,
        "R0 exposure ledger",
    )
    _, supplied_report_bytes = read_verified_bundle_bytes(
        request["author_report"],
        bundle_root,
        "R0 author report",
    )
    _require_canonical_json(supplied_report_bytes, "R0 author report")
    derived = derive_r0_envelope(
        annotation_bytes=annotation_bytes,
        class_map_bytes=class_map_bytes,
        exposure_ledger_bytes=ledger_bytes,
        population_record=population_record,
        protocol_record=protocol_record,
        review_record=review_record,
        frozen_ledger_prefix=frozen_ledger_prefix,
    )
    derived_bytes = canonical_json_bytes(derived)
    if supplied_report_bytes != derived_bytes:
        raise PrefixRouteProtocolV2Error(
            "R0 author report differs from source-derived envelope"
        )
    return derived


__all__ = [
    "DIFFERENCE_REASON_SCHEMA",
    "EXPOSURE_LEDGER_SCHEMA",
    "ID_LIST_SCHEMA",
    "MANIFEST_SCHEMA",
    "POPULATION_REQUEST_SCHEMA",
    "PROTOCOL_ID",
    "PROTOCOL_PASS",
    "PROTOCOL_REVISE",
    "PROTOCOL_SCHEMA",
    "R0_ENVELOPE_SCHEMA",
    "R0_REQUEST_SCHEMA",
    "REVIEWER_FINGERPRINT",
    "REVIEWER_ID",
    "REVIEWER_PUBLIC_KEY",
    "REVIEW_SCHEMA",
    "PrefixRouteProtocolV2Error",
    "authorize_collection",
    "canonical_json_bytes",
    "canonical_sha256",
    "derive_r0_envelope",
    "load_protocol",
    "load_signed_review",
    "load_source_manifest",
    "validate_exposure_ledger_bytes",
    "validate_population_bundle",
    "validate_protocol",
    "validate_r0_bundle",
    "verify_frozen_git_tree",
]
