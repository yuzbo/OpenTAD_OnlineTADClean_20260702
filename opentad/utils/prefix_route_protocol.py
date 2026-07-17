"""Fail-closed validation for the prefix-route evidence protocol."""

from __future__ import annotations

from datetime import datetime
import hashlib
import math
from pathlib import Path
import re
import subprocess

from .evidence_bundle import read_stable_file_bytes, strict_json_from_bytes


PROTOCOL_SCHEMA_VERSION = "prefix-route-identifiability-protocol-v1"
PROTOCOL_ID = "prefix-route-identifiability-20260717-v1"
PROTOCOL_REVIEW_SCHEMA_VERSION = "prefix-route-protocol-review-certificate-v1"
PROTOCOL_PASS = "PASS_PROTOCOL_TO_OUTCOME_BLIND_EVIDENCE_COLLECTION"
PROTOCOL_REVISE = "REVISE_PROTOCOL_BEFORE_COLLECTION"
R0_SCHEMA_VERSION = "prefix-route-r0-evidence-v1"
R1_SCHEMA_VERSION = "prefix-route-r1-certificate-v1"
POPULATION_SCHEMA_VERSION = "prefix-route-reporting-population-v1"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "protocol_id",
    "status",
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
_SECTION_FIELDS = {
    "task": {
        "name",
        "input_at_decision",
        "future_frame_access",
        "formal_output",
        "commit_semantics",
        "excluded_task_changes",
    },
    "governance": {
        "current_authorization",
        "blocked_before_protocol_pass",
        "protocol_pass_verdict",
        "protocol_revise_verdict",
        "post_pass_scope",
        "no_automatic_escalation",
        "review_certificate",
    },
    "population": {"dataset", "development", "reporting"},
    "r0": {
        "name",
        "definitions",
        "required_aggregates",
        "statistical_unit",
        "cluster_uncertainty",
        "claim_eligibility",
        "prior_exposure_ledger",
        "execution_isolation",
        "artifact_contract",
    },
    "r1": {
        "name",
        "claim_scope",
        "excluded_scope",
        "static_code_hypothesis",
        "certificate_schema",
        "support_rules",
        "dynamic_audit",
        "artifact_linkage",
    },
    "r2": {"name", "arms", "fairness", "threshold_calibration"},
    "r3": {
        "name",
        "controls",
        "benchmark_kill_rule",
        "corruption_kill_rule",
    },
    "r4": {
        "name",
        "required_deletions",
        "conditional_not_core",
        "necessity_rule",
        "internal_only_evidence_forbidden",
    },
    "r5": {
        "name",
        "unit",
        "factor_families",
        "sets",
        "hidden_generation",
        "disjointness",
        "benchmark_kill_rule",
    },
    "r6": {
        "name",
        "exact_delta_table",
        "delta_failure_rule",
        "primary_metrics",
        "eligible_stress_metrics",
        "equivalence_margins",
        "paired_inference",
        "b4_survival_rule",
        "kill_rules",
    },
    "evidence_package": {
        "required_objects_before_route_re_review",
        "forbidden_objects",
        "route_re_review_verdicts",
        "route_pass_does_not_authorize_gpu",
    },
    "source_bindings": {
        "parent_commit",
        "protocol_review_sha256",
        "protocol_review_absorption_sha256",
        "code_sources",
    },
}
_EXPECTED_ARMS = {"B0", "B1", "B2", "B3", "B4"}
_EXPECTED_CONTROLS = {
    "count_only",
    "template_timing",
    "feature_time_shuffle",
    "semantic_derangement",
    "ledger_only",
    "history_off",
}
_EXPECTED_ABLATIONS = {"A1", "A2", "A3", "A4", "A5"}
_EXPECTED_FACTOR_FAMILIES = {
    "event_topology",
    "temporal_geometry",
    "semantic_mapping",
    "observation_distribution",
}
_EXPECTED_OOD_SETS = {
    "TRAIN",
    "IID_HOLDOUT",
    "SINGLE_SHIFT_OOD",
    "COMPOUND_OOD",
}
_EXPECTED_DELTAS = {
    "D1_ATOMIC_RELEASE_BEFORE_SAME_BIN_RESEED",
    "D2_CONTINUOUS_MASS_EPHEMERAL_NONCANONICAL_UOT",
}
_EXPECTED_GLOBAL_METRICS = {
    "mOnlineAP",
    "event_recall",
    "false_emission_per_video",
    "duplicate_per_gt",
    "fragmentation_per_gt",
    "endpoint_latency_bins",
}
_EXPECTED_STRESS_METRICS = {
    "same_class_repetition_recall",
    "same_class_overlap_recall",
    "same_bin_end_start_recall",
    "direct_complete_recall",
}


class PrefixRouteProtocolError(ValueError):
    """Raised when a protocol or certificate is not uniquely executable."""


def _mapping(value, label):
    if not isinstance(value, dict):
        raise PrefixRouteProtocolError(f"{label} must be a JSON object")
    return value


def _sequence(value, label):
    if not isinstance(value, list):
        raise PrefixRouteProtocolError(f"{label} must be a JSON array")
    return value


def _exact_fields(value, expected, label):
    value = _mapping(value, label)
    actual = set(value)
    if actual != set(expected):
        raise PrefixRouteProtocolError(
            f"{label} fields differ; missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )
    return value


def _nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise PrefixRouteProtocolError(f"{label} must be non-empty text")
    return value


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrefixRouteProtocolError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise PrefixRouteProtocolError(f"{label} must be a finite number")
    return value


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PrefixRouteProtocolError(f"{label} must be a positive integer")
    return value


def _sha256_value(value, label):
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PrefixRouteProtocolError(f"{label} must be lowercase SHA-256")
    return value


def _commit_value(value, label):
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise PrefixRouteProtocolError(f"{label} must be a lowercase 40-hex commit")
    return value


def _unique_strings(value, label):
    values = _sequence(value, label)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise PrefixRouteProtocolError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise PrefixRouteProtocolError(f"{label} contains duplicate values")
    return values


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
            raise PrefixRouteProtocolError(f"{path} contains a placeholder")


def _validate_timestamp(value, label):
    value = _nonempty_string(value, label)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PrefixRouteProtocolError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise PrefixRouteProtocolError(f"{label} must include a timezone")
    return value


def _validate_governance(protocol):
    governance = protocol["governance"]
    if governance["protocol_pass_verdict"] != PROTOCOL_PASS:
        raise PrefixRouteProtocolError("protocol PASS verdict differs")
    if governance["protocol_revise_verdict"] != PROTOCOL_REVISE:
        raise PrefixRouteProtocolError("protocol REVISE verdict differs")
    if governance["no_automatic_escalation"] is not True:
        raise PrefixRouteProtocolError("automatic escalation must be forbidden")
    blocked = set(_unique_strings(
        governance["blocked_before_protocol_pass"],
        "governance.blocked_before_protocol_pass",
    ))
    required_blocks = {
        "NEW_R0_COLLECTION",
        "NEW_R1_COLLECTION",
        "MODEL_OUTCOME_INSPECTION",
        "B0_B4_MODEL_IMPLEMENTATION",
        "MODEL_P0",
        "GPU_PROFILE",
        "FORMAL_TRAINING",
    }
    if not required_blocks.issubset(blocked):
        raise PrefixRouteProtocolError("governance is missing mandatory blocks")
    if set(governance["post_pass_scope"]) != {
        "READ_ONLY_R0_ANNOTATION_CENSUS",
        "READ_ONLY_R1_CACHE_CAUSALITY_AUDIT",
    }:
        raise PrefixRouteProtocolError("protocol PASS scope differs")
    review = _exact_fields(
        governance["review_certificate"],
        {
            "schema_version",
            "required_fields",
            "allowed_verdicts",
            "minimum_distinct_reviewer_count",
            "author_may_not_self_certify",
        },
        "governance.review_certificate",
    )
    if review["schema_version"] != PROTOCOL_REVIEW_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("review certificate schema differs")
    if set(review["allowed_verdicts"]) != {PROTOCOL_PASS, PROTOCOL_REVISE}:
        raise PrefixRouteProtocolError("review certificate verdict set differs")
    if review["minimum_distinct_reviewer_count"] != 1:
        raise PrefixRouteProtocolError("exactly one independent reviewer is required")
    if review["author_may_not_self_certify"] is not True:
        raise PrefixRouteProtocolError("author self-certification must be forbidden")


def _validate_population(protocol):
    population = protocol["population"]
    if population["dataset"] != "THUMOS14":
        raise PrefixRouteProtocolError("dataset must remain THUMOS14")
    development = _exact_fields(
        population["development"],
        {
            "universe_role",
            "expected_count",
            "fit_core_role",
            "fit_core_expected_count",
            "calibration_role",
            "calibration_expected_count",
            "split_seed",
            "fit_and_calibration_must_be_disjoint",
        },
        "population.development",
    )
    if (
        development["expected_count"],
        development["fit_core_expected_count"],
        development["calibration_expected_count"],
        development["split_seed"],
    ) != (200, 160, 40, 20260713):
        raise PrefixRouteProtocolError("development population constants differ")
    if development["fit_and_calibration_must_be_disjoint"] is not True:
        raise PrefixRouteProtocolError("development splits must be disjoint")
    reporting = _exact_fields(
        population["reporting"],
        {
            "primary_role",
            "primary_expected_count",
            "historical_role",
            "historical_expected_count",
            "historical_population_may_be_primary",
            "selection_rule",
            "allowed_difference_reason_codes",
            "required_certificate_fields",
            "unresolved_action",
            "no_dual_primary_reporting",
            "remote_manifest_access_at_freeze",
        },
        "population.reporting",
    )
    if (
        reporting["primary_role"],
        reporting["primary_expected_count"],
        reporting["historical_expected_count"],
    ) != ("canonical_reporting_213", 213, 211):
        raise PrefixRouteProtocolError("reporting population rule differs")
    if (
        reporting["historical_population_may_be_primary"] is not False
        or reporting["no_dual_primary_reporting"] is not True
    ):
        raise PrefixRouteProtocolError("historical 211 must remain audit-only")
    if reporting["unresolved_action"] != "BLOCK_R0_POPULATION_UNRESOLVED":
        raise PrefixRouteProtocolError("unresolved population must block R0")
    _unique_strings(
        reporting["allowed_difference_reason_codes"],
        "population.reporting.allowed_difference_reason_codes",
    )
    _unique_strings(
        reporting["required_certificate_fields"],
        "population.reporting.required_certificate_fields",
    )


def _validate_r0(protocol):
    r0 = protocol["r0"]
    definitions = _mapping(r0["definitions"], "r0.definitions")
    required_values = {
        "feature_stride_frames": 8,
        "interval_semantics": "half_open_[start,end)",
        "touching_is_overlap": False,
        "sequential_gap": "next_start-current_end",
        "ambiguous_policy": (
            "report_separately_then_exclude_from_legal_action_statistics"
        ),
    }
    for field, expected in required_values.items():
        if definitions.get(field) != expected:
            raise PrefixRouteProtocolError(f"r0.definitions.{field} differs")
    if set(definitions.get("gap_sign", {})) != {"positive", "zero", "negative"}:
        raise PrefixRouteProtocolError("r0 gap sign contract differs")
    if r0["statistical_unit"] != "video_cluster":
        raise PrefixRouteProtocolError("R0 statistical unit must be video")
    uncertainty = _mapping(r0["cluster_uncertainty"], "r0.cluster_uncertainty")
    if (
        uncertainty.get("method"),
        uncertainty.get("resamples"),
        uncertainty.get("seed"),
        uncertainty.get("confidence_level"),
    ) != (
        "nonparametric_video_cluster_bootstrap",
        10000,
        2026071701,
        0.95,
    ):
        raise PrefixRouteProtocolError("R0 uncertainty contract differs")
    eligibility = _mapping(r0["claim_eligibility"], "r0.claim_eligibility")
    if set(eligibility.get("primary_stress_families", ())) != {
        "same_class_repetition",
        "same_class_overlap",
        "same_bin_end_start",
        "direct_complete",
    }:
        raise PrefixRouteProtocolError("R0 stress families differ")
    minimums = eligibility.get("primary_minimums", {})
    if minimums != {
        "independent_videos": 30,
        "gt_instances": 100,
        "fraction_of_positive_videos": 0.05,
        "classes": 3,
        "power": 0.8,
    }:
        raise PrefixRouteProtocolError("R0 primary eligibility thresholds differ")
    power = _mapping(eligibility.get("power_design"), "r0.power_design")
    if not math.isclose(
        _finite_number(power.get("per_family_two_sided_alpha"), "R0 alpha"),
        power.get("family_wise_alpha") / power.get("primary_family_count"),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise PrefixRouteProtocolError("R0 family-wise alpha allocation differs")
    if (
        power.get("absolute_effect"),
        power.get("intracluster_correlation"),
        power.get("required_power"),
    ) != (0.1, 0.2, 0.8):
        raise PrefixRouteProtocolError("R0 power constants differ")
    exposure = _mapping(r0["prior_exposure_ledger"], "r0.prior_exposure_ledger")
    if (
        exposure.get("annotation_unseen_claim_allowed") is not False
        or exposure.get("model_outcome_blind_required") is not True
    ):
        raise PrefixRouteProtocolError("R0 prior-exposure disclosure differs")
    isolation = _mapping(r0["execution_isolation"], "r0.execution_isolation")
    if isolation.get("author_output") != "aggregate_only":
        raise PrefixRouteProtocolError("R0 author disclosure must be aggregate-only")
    if isolation.get("author_may_see_new_hard_case_ids") is not False:
        raise PrefixRouteProtocolError("R0 hard-case IDs must remain reviewer-only")
    if isolation.get("existing_output_root_action") != "REFUSE_OVERWRITE":
        raise PrefixRouteProtocolError("R0 output must fail closed on overwrite")
    artifact = _mapping(r0["artifact_contract"], "r0.artifact_contract")
    if artifact.get("schema_version") != R0_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("R0 artifact schema differs")


def _validate_r1(protocol):
    r1 = protocol["r1"]
    if r1["claim_scope"] != "decoded_rgb_frames_to_feature_token_only":
        raise PrefixRouteProtocolError("R1 claim scope differs")
    schema = _mapping(r1["certificate_schema"], "r1.certificate_schema")
    if schema.get("schema_version") != R1_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("R1 certificate schema differs")
    required_fields = set(schema.get("required_top_level_fields", ()))
    mandatory = {
        "hf_snapshot_revision",
        "processor_config_sha256",
        "weight_shard_sha256",
        "raw_video_sha256_by_video",
        "support_map_sha256",
        "dynamic_perturbation_audit",
        "existing_cache_linkage",
        "status",
    }
    if not mandatory.issubset(required_fields):
        raise PrefixRouteProtocolError("R1 certificate is missing identity fields")
    support = _mapping(r1["support_rules"], "r1.support_rules")
    if (
        support.get("general") != "max(support_frames)<=decision_frame"
        or support.get("current_single_frame_path")
        != "support_frames=[decision_frame]"
        or support.get("monotonic_source_metadata_alone_is_sufficient") is not False
    ):
        raise PrefixRouteProtocolError("R1 support rules differ")
    dynamic = _mapping(r1["dynamic_audit"], "r1.dynamic_audit")
    if set(dynamic.get("categories", ())) != {
        "first_token",
        "interior_token",
        "last_complete_bin",
        "last_partial_bin",
    }:
        raise PrefixRouteProtocolError("R1 dynamic audit categories differ")
    if (
        dynamic.get("tokens_per_category"),
        dynamic.get("minimum_unique_videos_overall"),
        dynamic.get("audit_seed"),
    ) != (32, 16, 2026071702):
        raise PrefixRouteProtocolError("R1 dynamic audit sample contract differs")
    if dynamic.get("historical_gpu_fp16_byte_match_required") is not False:
        raise PrefixRouteProtocolError("R1 must not require cross-environment byte parity")
    linkage = _mapping(r1["artifact_linkage"], "r1.artifact_linkage")
    if linkage.get("missing_any_required_identity_action") != "FAIL_UNVERIFIABLE":
        raise PrefixRouteProtocolError("R1 missing identity must fail unverifiable")
    if linkage.get("replacement_cache_automatically_authorized") is not False:
        raise PrefixRouteProtocolError("R1 must not authorize replacement cache creation")


def _validate_r2_to_r6(protocol):
    r2 = protocol["r2"]
    arms = _mapping(r2["arms"], "r2.arms")
    if set(arms) != _EXPECTED_ARMS:
        raise PrefixRouteProtocolError("B0-B4 arm set differs")
    b4_deltas = set(arms["B4"].get("allowed_route_specific_deltas", ()))
    if b4_deltas != _EXPECTED_DELTAS:
        raise PrefixRouteProtocolError("B4 route-specific delta set differs")
    if arms["B0"].get("visual_memory") != "same_causal_memory_as_all_arms":
        raise PrefixRouteProtocolError("B0 may not be deprived of causal visual memory")
    if arms["B2"].get("training_assignment") != (
        "temporal_tracklet_aware_one_to_one_assignment"
    ):
        raise PrefixRouteProtocolError("B2 is not an exact temporal MOTR attack")
    fairness = _mapping(r2["fairness"], "r2.fairness")
    if fairness.get("comparison_view") != (
        "BOTH_CAPACITY_AND_RESOURCE_MATCHED_REQUIRED"
    ):
        raise PrefixRouteProtocolError("both fairness views are required")
    if (
        fairness.get("trainable_parameter_relative_tolerance"),
        fairness.get("training_mac_relative_tolerance"),
    ) != (0.05, 0.1):
        raise PrefixRouteProtocolError("B0-B4 matching tolerances differ")
    if fairness.get("model_p0_must_bind_numeric_budgets_before_implementation") is not True:
        raise PrefixRouteProtocolError("future numeric budgets must be preregistered")
    if fairness.get("comparison_seeds") != [705, 706, 707]:
        raise PrefixRouteProtocolError("comparison seeds differ")
    calibration = _mapping(r2["threshold_calibration"], "r2.threshold_calibration")
    if calibration.get("calibration_population") != "calibration_40":
        raise PrefixRouteProtocolError("threshold calibration population differs")
    if calibration.get("threshold_sweep_is_sensitivity_not_negative_control") is not True:
        raise PrefixRouteProtocolError("threshold sweep classification differs")

    controls = _mapping(protocol["r3"]["controls"], "r3.controls")
    if set(controls) != _EXPECTED_CONTROLS:
        raise PrefixRouteProtocolError("negative control set differs")
    if controls["semantic_derangement"].get(
        "global_consistent_class_rename_forbidden"
    ) is not True:
        raise PrefixRouteProtocolError("global class renaming is not a valid control")

    deletions = _sequence(protocol["r4"]["required_deletions"], "r4.required_deletions")
    deletion_ids = {item.get("id") for item in deletions if isinstance(item, dict)}
    if deletion_ids != _EXPECTED_ABLATIONS or len(deletions) != len(_EXPECTED_ABLATIONS):
        raise PrefixRouteProtocolError("core ablation set differs")

    r5 = protocol["r5"]
    if set(_mapping(r5["factor_families"], "r5.factor_families")) != (
        _EXPECTED_FACTOR_FAMILIES
    ):
        raise PrefixRouteProtocolError("structural OOD factor families differ")
    if set(_mapping(r5["sets"], "r5.sets")) != _EXPECTED_OOD_SETS:
        raise PrefixRouteProtocolError("structural OOD sets differ")
    hidden = _mapping(r5["hidden_generation"], "r5.hidden_generation")
    if hidden.get("owner") != "independent_reviewer":
        raise PrefixRouteProtocolError("hidden OOD must be reviewer-owned")

    r6 = protocol["r6"]
    rows = _sequence(r6["exact_delta_table"], "r6.exact_delta_table")
    row_by_element = {
        row.get("element"): row for row in rows if isinstance(row, dict)
    }
    if not _EXPECTED_DELTAS.issubset(row_by_element):
        raise PrefixRouteProtocolError("R6 exact-delta table omits D1 or D2")
    if row_by_element["persistent_carriers"].get("verdict") != "NO_DELTA":
        raise PrefixRouteProtocolError("persistent carriers alone are not a delta")
    if row_by_element["immutable_ledger"].get("verdict") != "NO_DELTA":
        raise PrefixRouteProtocolError("the shared ledger is not a delta")
    if set(r6["primary_metrics"]) != _EXPECTED_GLOBAL_METRICS:
        raise PrefixRouteProtocolError("R6 global primary metric set differs")
    if set(r6["eligible_stress_metrics"]) != _EXPECTED_STRESS_METRICS:
        raise PrefixRouteProtocolError("R6 stress metric set differs")
    margins = _mapping(r6["equivalence_margins"], "r6.equivalence_margins")
    if margins.get("mOnlineAP_absolute") != 0.005:
        raise PrefixRouteProtocolError("mOnlineAP equivalence margin differs")
    if margins.get("endpoint_latency_bins") != 1:
        raise PrefixRouteProtocolError("latency equivalence margin differs")
    paired = _mapping(r6["paired_inference"], "r6.paired_inference")
    if (
        paired.get("bootstrap_resamples"),
        paired.get("bootstrap_seed"),
        paired.get("family_wise_alpha"),
    ) != (10000, 2026071707, 0.05):
        raise PrefixRouteProtocolError("paired inference contract differs")
    if paired.get("simultaneous_interval") != (
        "bonferroni_adjusted_paired_percentile"
    ):
        raise PrefixRouteProtocolError("simultaneous CI method differs")


def _validate_sources(protocol, repo_root=None):
    sources = protocol["source_bindings"]
    _commit_value(sources["parent_commit"], "source_bindings.parent_commit")
    _sha256_value(
        sources["protocol_review_sha256"],
        "source_bindings.protocol_review_sha256",
    )
    _sha256_value(
        sources["protocol_review_absorption_sha256"],
        "source_bindings.protocol_review_absorption_sha256",
    )
    code_sources = _mapping(sources["code_sources"], "source_bindings.code_sources")
    if not code_sources:
        raise PrefixRouteProtocolError("source_bindings.code_sources is empty")
    for relative_path, expected_sha256 in code_sources.items():
        _nonempty_string(relative_path, "source path")
        _sha256_value(expected_sha256, f"source hash {relative_path}")
        if repo_root is None:
            continue
        path = Path(repo_root).resolve() / relative_path
        if not path.is_file():
            raise PrefixRouteProtocolError(f"bound source is missing: {relative_path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise PrefixRouteProtocolError(
                f"bound source hash differs for {relative_path}: "
                f"expected {expected_sha256}, found {actual}"
            )


def validate_protocol(protocol, *, repo_root=None):
    """Validate that the protocol closes the reviewed R0-R6 choices."""

    protocol = _exact_fields(protocol, _TOP_LEVEL_FIELDS, "protocol")
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("protocol schema version differs")
    if protocol["protocol_id"] != PROTOCOL_ID:
        raise PrefixRouteProtocolError("protocol ID differs")
    if protocol["status"] != "PROTOCOL_REVIEW_PENDING":
        raise PrefixRouteProtocolError("frozen protocol status differs")
    for section, fields in _SECTION_FIELDS.items():
        _exact_fields(protocol[section], fields, section)
    _reject_placeholders(protocol)
    task = protocol["task"]
    if task["future_frame_access"] is not False:
        raise PrefixRouteProtocolError("future frame access must be forbidden")
    if task["commit_semantics"] != (
        "append_only_no_revision_no_merge_no_nms_no_offline_cleanup"
    ):
        raise PrefixRouteProtocolError("commit semantics differ")
    _validate_governance(protocol)
    _validate_population(protocol)
    _validate_r0(protocol)
    _validate_r1(protocol)
    _validate_r2_to_r6(protocol)
    _validate_sources(protocol, repo_root=repo_root)
    evidence = protocol["evidence_package"]
    if evidence["route_pass_does_not_authorize_gpu"] is not True:
        raise PrefixRouteProtocolError("route PASS must not authorize GPU use")
    return protocol


def load_protocol(path, *, repo_root=None):
    """Read, hash, and validate one immutable protocol file."""

    resolved, payload = read_stable_file_bytes(path, "prefix-route protocol")
    protocol = strict_json_from_bytes(
        payload,
        "prefix-route protocol",
        require_object=True,
    )
    validate_protocol(protocol, repo_root=repo_root)
    return {
        "path": resolved,
        "bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "protocol": protocol,
    }


def validate_review_certificate(
    certificate,
    *,
    protocol,
    protocol_sha256,
    author_id,
    expected_commit=None,
):
    """Bind one independent reviewer verdict to exact protocol bytes."""

    required = set(
        protocol["governance"]["review_certificate"]["required_fields"]
    )
    certificate = _exact_fields(certificate, required, "review certificate")
    if certificate["schema_version"] != PROTOCOL_REVIEW_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("review certificate schema differs")
    if certificate["protocol_id"] != protocol["protocol_id"]:
        raise PrefixRouteProtocolError("review certificate protocol ID differs")
    if certificate["protocol_sha256"] != protocol_sha256:
        raise PrefixRouteProtocolError("review certificate protocol hash differs")
    commit = _commit_value(certificate["protocol_commit"], "review protocol commit")
    if expected_commit is not None and commit != expected_commit:
        raise PrefixRouteProtocolError("review certificate commit differs")
    reviewer_id = _nonempty_string(certificate["reviewer_id"], "reviewer_id")
    _nonempty_string(certificate["review_task_id"], "review_task_id")
    if reviewer_id == _nonempty_string(author_id, "author_id"):
        raise PrefixRouteProtocolError("author may not self-certify the protocol")
    _validate_timestamp(certificate["reviewed_at"], "reviewed_at")
    if certificate["verdict"] not in {PROTOCOL_PASS, PROTOCOL_REVISE}:
        raise PrefixRouteProtocolError("review certificate verdict differs")
    _sha256_value(
        certificate["review_artifact_sha256"],
        "review_artifact_sha256",
    )
    return certificate


def load_review_certificate(
    path,
    *,
    protocol,
    protocol_sha256,
    author_id,
    expected_commit=None,
    review_artifact_path=None,
):
    resolved, payload = read_stable_file_bytes(path, "protocol review certificate")
    certificate = strict_json_from_bytes(
        payload,
        "protocol review certificate",
        require_object=True,
    )
    validate_review_certificate(
        certificate,
        protocol=protocol,
        protocol_sha256=protocol_sha256,
        author_id=author_id,
        expected_commit=expected_commit,
    )
    artifact_record = None
    if review_artifact_path is not None:
        artifact_path, artifact_bytes = read_stable_file_bytes(
            review_artifact_path,
            "independent protocol review artifact",
        )
        artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_sha256 != certificate["review_artifact_sha256"]:
            raise PrefixRouteProtocolError(
                "independent review artifact hash differs from certificate"
            )
        artifact_record = {
            "path": artifact_path,
            "bytes": artifact_bytes,
            "sha256": artifact_sha256,
        }
    return {
        "path": resolved,
        "bytes": payload,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "certificate": certificate,
        "review_artifact": artifact_record,
    }


def validate_population_certificate(
    certificate,
    *,
    protocol,
    protocol_sha256,
    review_certificate_sha256,
):
    """Validate the fail-closed 211-versus-213 population decision."""

    reporting = protocol["population"]["reporting"]
    certificate = _exact_fields(
        certificate,
        set(reporting["required_certificate_fields"]),
        "population certificate",
    )
    if certificate["schema_version"] != POPULATION_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("population certificate schema differs")
    if certificate["protocol_id"] != protocol["protocol_id"]:
        raise PrefixRouteProtocolError("population certificate protocol ID differs")
    if certificate["protocol_sha256"] != protocol_sha256:
        raise PrefixRouteProtocolError("population certificate protocol hash differs")
    if certificate["protocol_review_certificate_sha256"] != (
        review_certificate_sha256
    ):
        raise PrefixRouteProtocolError(
            "population certificate review binding differs"
        )
    if (
        certificate["historical_count"],
        certificate["canonical_count"],
    ) != (
        reporting["historical_expected_count"],
        reporting["primary_expected_count"],
    ):
        raise PrefixRouteProtocolError("population certificate counts differ")
    _sha256_value(
        certificate["historical_ids_sha256"],
        "historical_ids_sha256",
    )
    _sha256_value(
        certificate["canonical_ids_sha256"],
        "canonical_ids_sha256",
    )
    missing = _unique_strings(
        certificate["missing_from_canonical"],
        "population.missing_from_canonical",
    ) if certificate["missing_from_canonical"] else []
    extra = _unique_strings(
        certificate["extra_in_canonical"],
        "population.extra_in_canonical",
    ) if certificate["extra_in_canonical"] else []
    difference_ids = set(missing) | set(extra)
    if set(missing) & set(extra):
        raise PrefixRouteProtocolError("population difference sets overlap")
    reasons = _mapping(certificate["difference_reasons"], "difference_reasons")
    if set(reasons) != difference_ids:
        raise PrefixRouteProtocolError("population difference reasons are incomplete")
    allowed = set(reporting["allowed_difference_reason_codes"])
    if any(reason not in allowed for reason in reasons.values()):
        raise PrefixRouteProtocolError("population difference reason is not allowed")
    if certificate["unexplained_ids"] != []:
        raise PrefixRouteProtocolError("population certificate has unexplained IDs")
    if certificate["status"] != "EXPLAINED_MISMATCH":
        raise PrefixRouteProtocolError("population status must be EXPLAINED_MISMATCH")
    if not difference_ids:
        raise PrefixRouteProtocolError("population mismatch certificate has no difference")
    if (
        certificate["historical_count"] - len(missing) + len(extra)
        != certificate["canonical_count"]
    ):
        raise PrefixRouteProtocolError(
            "population difference cardinality does not reconcile 211 to 213"
        )
    sources = _mapping(certificate["source_artifacts"], "source_artifacts")
    if not sources:
        raise PrefixRouteProtocolError("population source artifacts are missing")
    for name, digest in sources.items():
        _nonempty_string(name, "population source artifact name")
        _sha256_value(digest, f"population source artifact {name}")
    return certificate


def validate_r0_evidence(
    evidence,
    *,
    protocol,
    protocol_sha256,
    review_certificate_sha256,
    population_certificate_sha256,
):
    """Validate R0 structure without interpreting uncollected outcomes."""

    contract = protocol["r0"]["artifact_contract"]
    evidence = _exact_fields(
        evidence,
        set(contract["required_top_level_fields"]),
        "R0 evidence",
    )
    if evidence["schema_version"] != R0_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("R0 evidence schema differs")
    if evidence["protocol_id"] != protocol["protocol_id"]:
        raise PrefixRouteProtocolError("R0 protocol ID differs")
    bindings = {
        "protocol_sha256": protocol_sha256,
        "protocol_review_certificate_sha256": review_certificate_sha256,
        "population_certificate_sha256": population_certificate_sha256,
    }
    for field, expected in bindings.items():
        if evidence[field] != expected:
            raise PrefixRouteProtocolError(f"R0 {field} differs")
    if evidence["definitions"] != protocol["r0"]["definitions"]:
        raise PrefixRouteProtocolError("R0 definitions differ from protocol")
    if evidence["status"] not in set(contract["allowed_statuses"]):
        raise PrefixRouteProtocolError("R0 status differs")
    _sha256_value(
        evidence["reviewer_detail_commitment_sha256"],
        "R0 reviewer detail commitment",
    )
    if evidence["author_disclosure"] != "aggregate_only":
        raise PrefixRouteProtocolError("R0 author disclosure differs")
    return evidence


def validate_r1_certificate(
    certificate,
    *,
    protocol,
    protocol_sha256,
    review_certificate_sha256,
):
    """Validate a future R1 certificate and its existing-cache disposition."""

    contract = protocol["r1"]["certificate_schema"]
    certificate = _exact_fields(
        certificate,
        set(contract["required_top_level_fields"]),
        "R1 certificate",
    )
    if certificate["schema_version"] != R1_SCHEMA_VERSION:
        raise PrefixRouteProtocolError("R1 certificate schema differs")
    if certificate["protocol_id"] != protocol["protocol_id"]:
        raise PrefixRouteProtocolError("R1 protocol ID differs")
    if certificate["protocol_sha256"] != protocol_sha256:
        raise PrefixRouteProtocolError("R1 protocol hash differs")
    if certificate["protocol_review_certificate_sha256"] != (
        review_certificate_sha256
    ):
        raise PrefixRouteProtocolError("R1 review certificate binding differs")
    if certificate["status"] not in set(contract["allowed_statuses"]):
        raise PrefixRouteProtocolError("R1 status differs")
    if certificate["motion_branch_enabled"] is not False:
        raise PrefixRouteProtocolError("R1 motion branch must be disabled")
    status = certificate["status"]
    if status == "PASS_R1_EXISTING_CACHE_CERTIFIED":
        for field in (
            "extractor_source_sha256",
            "resolved_command_sha256",
            "environment_lock_sha256",
            "model_config_sha256",
            "processor_config_sha256",
            "annotation_raw_sha256",
            "cache_manifest_sha256",
            "support_map_sha256",
        ):
            _sha256_value(certificate[field], f"R1 {field}")
        _commit_value(certificate["repository_commit"], "R1 repository_commit")
        _nonempty_string(certificate["hf_snapshot_revision"], "R1 hf revision")
        weight_hashes = certificate["weight_shard_sha256"]
        if not isinstance(weight_hashes, list) or not weight_hashes:
            raise PrefixRouteProtocolError("R1 weight shard hashes are missing")
        for index, digest in enumerate(weight_hashes):
            _sha256_value(digest, f"R1 weight shard {index}")
        for field in (
            "raw_video_sha256_by_video",
            "feature_array_sha256_by_video",
        ):
            values = _mapping(certificate[field], f"R1 {field}")
            if len(values) != (
                protocol["population"]["reporting"]["primary_expected_count"]
            ):
                raise PrefixRouteProtocolError(
                    f"R1 {field} must cover canonical reporting population"
                )
            for video_id, digest in values.items():
                _nonempty_string(video_id, f"R1 {field} video ID")
                _sha256_value(digest, f"R1 {field} {video_id}")
        static = _mapping(certificate["static_support_audit"], "R1 static audit")
        dynamic = _mapping(
            certificate["dynamic_perturbation_audit"],
            "R1 dynamic audit",
        )
        linkage = _mapping(
            certificate["existing_cache_linkage"],
            "R1 cache linkage",
        )
        if static.get("all_support_le_decision") is not True:
            raise PrefixRouteProtocolError("R1 static future-support audit failed")
        if static.get("all_support_equals_decision") is not True:
            raise PrefixRouteProtocolError("R1 single-frame support audit failed")
        required_dynamic_passes = {
            "future_mutation_invariant": True,
            "other_batch_image_invariant": True,
            "batch_order_invariant": True,
            "support_frame_sensitive": True,
        }
        if any(dynamic.get(key) is not value for key, value in required_dynamic_passes.items()):
            raise PrefixRouteProtocolError("R1 dynamic audit did not pass")
        required_linkage = {
            "exact_weight_snapshot_bound": True,
            "raw_video_hashes_bound": True,
            "extraction_environment_bound": True,
            "feature_arrays_bound": True,
        }
        if any(linkage.get(key) is not value for key, value in required_linkage.items()):
            raise PrefixRouteProtocolError("R1 existing-cache linkage did not pass")
    else:
        linkage = _mapping(
            certificate["existing_cache_linkage"],
            "R1 cache linkage",
        )
        reasons = linkage.get("failure_reasons")
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise PrefixRouteProtocolError(
                "failed R1 certificate requires non-empty failure reasons"
            )
    return certificate


def verify_protocol_git_binding(
    *,
    protocol_path,
    repo_root,
    commit,
    expected_sha256,
):
    """Verify that a certificate names a commit containing the exact protocol."""

    repo_root = Path(repo_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    try:
        relative = protocol_path.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise PrefixRouteProtocolError("protocol path is outside repository") from exc
    _commit_value(commit, "protocol commit")
    _sha256_value(expected_sha256, "protocol sha256")
    try:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrefixRouteProtocolError(
            "cannot read protocol bytes from certified commit"
        ) from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise PrefixRouteProtocolError(
            f"certified commit protocol hash differs: expected "
            f"{expected_sha256}, found {actual}"
        )
    return True


def collection_authorization(
    *,
    protocol_record,
    review_record=None,
):
    """Return the only legal collection scope, or a fail-closed block."""

    if review_record is None:
        return {
            "authorized": False,
            "status": "BLOCKED_PENDING_INDEPENDENT_PROTOCOL_REVIEW",
            "allowed": [],
        }
    verdict = review_record["certificate"]["verdict"]
    if verdict != PROTOCOL_PASS:
        return {
            "authorized": False,
            "status": PROTOCOL_REVISE,
            "allowed": [],
        }
    return {
        "authorized": True,
        "status": "AUTHORIZED_READ_ONLY_R0_R1_ONLY",
        "allowed": list(
            protocol_record["protocol"]["governance"]["post_pass_scope"]
        ),
        "blocked": [
            action
            for action in protocol_record["protocol"]["governance"][
                "blocked_before_protocol_pass"
            ]
            if action not in {"NEW_R0_COLLECTION", "NEW_R1_COLLECTION"}
        ],
    }
