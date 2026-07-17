"""Deterministic structural-OOD sequence generator for Protocol V2."""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math


GENERATOR_SCHEMA = "prefix-route-structural-ood-sequence-v2"
GENERATOR_VERSION = "20260717.2"
SEQUENCE_LENGTH_BINS = 64
FEATURE_DIM = 16
CLASS_COUNT = 4

IID_TOPOLOGIES = (
    "empty",
    "single",
    "disjoint_pair",
    "disjoint_triple",
    "nested_pair",
    "partial_overlap_pair",
    "same_bin_handoff",
)
OOD_TOPOLOGIES = (
    "disjoint_four",
    "disjoint_six",
    "chain_three",
    "clique_three",
)
IID_TEMPORAL = (
    "duration_2_gap_0_delay_0",
    "duration_4_gap_2_delay_0",
    "duration_8_gap_8_delay_1",
    "duration_16_gap_2_delay_1",
)
OOD_TEMPORAL = (
    "duration_1_gap_16_delay_2",
    "duration_32_gap_16_delay_4",
    "duration_8_gap_neg2_delay_2",
    "duration_16_gap_neg8_delay_4",
)
IID_SEMANTIC = ("identity_prototype_to_class",)
OOD_SEMANTIC = ("heldout_prototype_to_class_derangement",)
IID_OBSERVATION = ("identity", "gaussian_noise_sigma_0.05")
OOD_OBSERVATION = (
    "seeded_orthogonal_signed_permutation",
    "two_component_basis_mixture",
    "student_t_noise_df_3_scaled_0.05",
)
FACTOR_FAMILIES = (
    "event_topology",
    "temporal_geometry",
    "semantic_mapping",
    "observation_distribution",
)


class PrefixRouteOODError(ValueError):
    pass


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


def _digest_bytes(*parts):
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _uniform(*parts):
    integer = int.from_bytes(_digest_bytes(*parts)[:8], "big")
    return (integer + 0.5) / (2**64)


def _normal(*parts):
    left = max(_uniform(*parts, "u1"), 2.0**-53)
    right = _uniform(*parts, "u2")
    return math.sqrt(-2.0 * math.log(left)) * math.cos(2.0 * math.pi * right)


def _student_t_df3(*parts):
    numerator = _normal(*parts, "z")
    chi_square = sum(_normal(*parts, "chi", index) ** 2 for index in range(3))
    return numerator / math.sqrt(max(chi_square / 3.0, 2.0**-52))


def _temporal_values(profile):
    parts = profile.split("_")
    if len(parts) != 6 or parts[0] != "duration" or parts[2] != "gap" or parts[4] != "delay":
        raise PrefixRouteOODError(f"invalid temporal profile: {profile}")
    duration = int(parts[1])
    gap_text = parts[3]
    gap = -int(gap_text[3:]) if gap_text.startswith("neg") else int(gap_text)
    delay = int(parts[5])
    return duration, gap, delay


def _event_count(topology):
    return {
        "empty": 0,
        "single": 1,
        "disjoint_pair": 2,
        "nested_pair": 2,
        "partial_overlap_pair": 2,
        "same_bin_handoff": 2,
        "disjoint_triple": 3,
        "chain_three": 3,
        "clique_three": 3,
        "disjoint_four": 4,
        "disjoint_six": 6,
    }[topology]


def _fit_events(topology, duration, gap):
    count = _event_count(topology)
    if count == 0:
        return []
    if topology == "single":
        starts = [max(1, (SEQUENCE_LENGTH_BINS - duration) // 2)]
        durations = [duration]
    elif topology.startswith("disjoint"):
        effective_gap = max(0, gap)
        total = count * duration + (count - 1) * effective_gap
        if total > SEQUENCE_LENGTH_BINS - 2:
            raise PrefixRouteOODError("invalid cross: disjoint events do not fit")
        first = max(1, (SEQUENCE_LENGTH_BINS - total) // 2)
        starts = [first + index * (duration + effective_gap) for index in range(count)]
        durations = [duration] * count
    elif topology == "nested_pair":
        if duration < 2:
            raise PrefixRouteOODError("invalid cross: nested pair requires duration >= 2")
        outer_duration = min(SEQUENCE_LENGTH_BINS - 4, max(4, duration * 2))
        inner_duration = max(1, min(duration, outer_duration - 2))
        first = max(1, (SEQUENCE_LENGTH_BINS - outer_duration) // 2)
        starts = [first, first + (outer_duration - inner_duration) // 2]
        durations = [outer_duration, inner_duration]
    elif topology == "partial_overlap_pair":
        if duration < 2:
            raise PrefixRouteOODError(
                "invalid cross: partial overlap requires duration >= 2"
            )
        first = max(1, (SEQUENCE_LENGTH_BINS - (duration + duration // 2)) // 2)
        starts = [first, first + max(1, duration // 2)]
        durations = [duration, duration]
    elif topology == "same_bin_handoff":
        if duration * 2 > SEQUENCE_LENGTH_BINS - 2:
            raise PrefixRouteOODError("invalid cross: handoff events do not fit")
        first = max(1, (SEQUENCE_LENGTH_BINS - 2 * duration) // 2)
        starts = [first, first + duration]
        durations = [duration, duration]
    elif topology == "chain_three":
        if duration < 2:
            raise PrefixRouteOODError("invalid cross: chain requires duration >= 2")
        step = max(1, duration // 2)
        total = duration + 2 * step
        if total > SEQUENCE_LENGTH_BINS - 2:
            raise PrefixRouteOODError("invalid cross: chain does not fit")
        first = max(1, (SEQUENCE_LENGTH_BINS - total) // 2)
        starts = [first, first + step, first + 2 * step]
        durations = [duration] * 3
    elif topology == "clique_three":
        if duration < 3:
            raise PrefixRouteOODError("invalid cross: clique requires duration >= 3")
        step = max(1, duration // 4)
        total = duration + 2 * step
        if total > SEQUENCE_LENGTH_BINS - 2:
            raise PrefixRouteOODError("invalid cross: clique does not fit")
        first = max(1, (SEQUENCE_LENGTH_BINS - total) // 2)
        starts = [first, first + step, first + 2 * step]
        durations = [duration] * 3
    else:
        raise PrefixRouteOODError(f"unknown topology: {topology}")
    events = []
    for index, (start, event_duration) in enumerate(zip(starts, durations)):
        end = start + event_duration
        if not 0 <= start < end <= SEQUENCE_LENGTH_BINS:
            raise PrefixRouteOODError("generated event escapes sequence")
        events.append(
            {
                "event_id": index,
                "start_bin": start,
                "end_bin": end,
                "prototype_id": index % CLASS_COUNT,
            }
        )
    return events


def _semantic_class(prototype_id, semantic_mapping):
    if semantic_mapping == "identity_prototype_to_class":
        return prototype_id
    if semantic_mapping == "heldout_prototype_to_class_derangement":
        return (1, 0, 3, 2)[prototype_id]
    raise PrefixRouteOODError(f"unknown semantic mapping: {semantic_mapping}")


def _signed_permutation(seed, sequence_index):
    ranked = sorted(
        range(FEATURE_DIM),
        key=lambda index: (_digest_bytes(seed, sequence_index, "perm", index), index),
    )
    signs = [
        -1.0 if _uniform(seed, sequence_index, "sign", index) < 0.5 else 1.0
        for index in range(FEATURE_DIM)
    ]
    return ranked, signs


def _apply_observation_distribution(
    vector,
    distribution,
    *,
    seed,
    sequence_index,
    time_bin,
):
    if distribution == "identity":
        return vector
    if distribution == "gaussian_noise_sigma_0.05":
        return [
            value + 0.05 * _normal(seed, sequence_index, time_bin, dim, "gaussian")
            for dim, value in enumerate(vector)
        ]
    permutation, signs = _signed_permutation(seed, sequence_index)
    rotated = [
        signs[dimension] * vector[permutation[dimension]]
        for dimension in range(FEATURE_DIM)
    ]
    if distribution == "seeded_orthogonal_signed_permutation":
        return rotated
    if distribution == "two_component_basis_mixture":
        scale = 1.0 / math.sqrt(2.0)
        return [
            scale * (left + right) for left, right in zip(vector, rotated)
        ]
    if distribution == "student_t_noise_df_3_scaled_0.05":
        return [
            value + 0.05 * _student_t_df3(
                seed,
                sequence_index,
                time_bin,
                dim,
                "student",
            )
            for dim, value in enumerate(vector)
        ]
    raise PrefixRouteOODError(
        f"unknown observation distribution: {distribution}"
    )


def _maximum_concurrency(events):
    maximum = 0
    for time_bin in range(SEQUENCE_LENGTH_BINS):
        maximum = max(
            maximum,
            sum(event["start_bin"] <= time_bin < event["end_bin"] for event in events),
        )
    return maximum


def generate_sequence(spec, *, seed, sequence_index, set_name):
    """Generate one exact synthetic causal sequence and its canonical hash."""

    required = {
        "event_topology",
        "temporal_geometry",
        "semantic_mapping",
        "observation_distribution",
    }
    if not isinstance(spec, dict) or set(spec) != required:
        raise PrefixRouteOODError("factor specification fields differ")
    topology = spec["event_topology"]
    temporal = spec["temporal_geometry"]
    semantic = spec["semantic_mapping"]
    observation = spec["observation_distribution"]
    if topology not in IID_TOPOLOGIES + OOD_TOPOLOGIES:
        raise PrefixRouteOODError("event topology is outside frozen grammar")
    if temporal not in IID_TEMPORAL + OOD_TEMPORAL:
        raise PrefixRouteOODError("temporal geometry is outside frozen grammar")
    if semantic not in IID_SEMANTIC + OOD_SEMANTIC:
        raise PrefixRouteOODError("semantic mapping is outside frozen grammar")
    if observation not in IID_OBSERVATION + OOD_OBSERVATION:
        raise PrefixRouteOODError(
            "observation distribution is outside frozen grammar"
        )
    duration, gap, endpoint_delay = _temporal_values(temporal)
    events = _fit_events(topology, duration, gap)
    event_rows = [
        {
            **event,
            "class_id": _semantic_class(event["prototype_id"], semantic),
            "completion_cue_bin": min(
                SEQUENCE_LENGTH_BINS - 1,
                event["end_bin"] + endpoint_delay,
            ),
        }
        for event in events
    ]
    observations = []
    for time_bin in range(SEQUENCE_LENGTH_BINS):
        active = [
            event
            for event in event_rows
            if event["start_bin"] <= time_bin < event["end_bin"]
        ]
        vector = [0.0] * FEATURE_DIM
        for event in active:
            vector[event["prototype_id"]] += 1.0
        vector[4] = float(
            sum(event["start_bin"] == time_bin for event in event_rows)
        )
        vector[5] = float(
            sum(event["completion_cue_bin"] == time_bin for event in event_rows)
        )
        vector[6] = time_bin / (SEQUENCE_LENGTH_BINS - 1)
        vector[7] = len(active) / max(1, CLASS_COUNT)
        transformed = _apply_observation_distribution(
            vector,
            observation,
            seed=seed,
            sequence_index=sequence_index,
            time_bin=time_bin,
        )
        observations.append([round(value, 8) for value in transformed])
    payload = {
        "schema_version": GENERATOR_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "set_name": str(set_name),
        "sequence_index": int(sequence_index),
        "seed": int(seed),
        "sequence_length_bins": SEQUENCE_LENGTH_BINS,
        "feature_dim": FEATURE_DIM,
        "class_count": CLASS_COUNT,
        "factor_spec": dict(spec),
        "observation_equation": (
            "active_prototype_sum+birth_impulse+delayed_completion_cue+"
            "normalized_time+active_count_then_frozen_distribution"
        ),
        "events": event_rows,
        "observations": observations,
        "derived": {
            "event_count": len(event_rows),
            "maximum_concurrency": _maximum_concurrency(event_rows),
            "endpoint_delay_bins": endpoint_delay,
        },
    }
    payload["sequence_sha256"] = _scientific_content_sha256(payload)
    return payload


def _scientific_content_sha256(sequence):
    scientific_content = {
        key: sequence[key]
        for key in (
            "schema_version",
            "generator_version",
            "sequence_length_bins",
            "feature_dim",
            "class_count",
            "factor_spec",
            "observation_equation",
            "events",
            "observations",
            "derived",
        )
    }
    return hashlib.sha256(canonical_json_bytes(scientific_content)).hexdigest()


def _iid_specs():
    return tuple(
        {
            "event_topology": topology,
            "temporal_geometry": temporal,
            "semantic_mapping": semantic,
            "observation_distribution": observation,
        }
        for topology, temporal, semantic, observation in itertools.product(
            IID_TOPOLOGIES,
            IID_TEMPORAL,
            IID_SEMANTIC,
            IID_OBSERVATION,
        )
        if _valid_spec(
            {
                "event_topology": topology,
                "temporal_geometry": temporal,
                "semantic_mapping": semantic,
                "observation_distribution": observation,
            }
        )
    )


def _single_shift_specs(family):
    pools = {
        "event_topology": (OOD_TOPOLOGIES, IID_TEMPORAL, IID_SEMANTIC, IID_OBSERVATION),
        "temporal_geometry": (IID_TOPOLOGIES, OOD_TEMPORAL, IID_SEMANTIC, IID_OBSERVATION),
        "semantic_mapping": (IID_TOPOLOGIES, IID_TEMPORAL, OOD_SEMANTIC, IID_OBSERVATION),
        "observation_distribution": (
            IID_TOPOLOGIES,
            IID_TEMPORAL,
            IID_SEMANTIC,
            OOD_OBSERVATION,
        ),
    }
    if family not in pools:
        raise PrefixRouteOODError(f"unknown shifted family: {family}")
    return tuple(
        {
            "event_topology": topology,
            "temporal_geometry": temporal,
            "semantic_mapping": semantic,
            "observation_distribution": observation,
        }
        for topology, temporal, semantic, observation in itertools.product(*pools[family])
        if _valid_spec(
            {
                "event_topology": topology,
                "temporal_geometry": temporal,
                "semantic_mapping": semantic,
                "observation_distribution": observation,
            }
        )
    )


def _valid_spec(spec):
    try:
        duration, gap, _ = _temporal_values(spec["temporal_geometry"])
        _fit_events(spec["event_topology"], duration, gap)
    except PrefixRouteOODError:
        return False
    return True


def iter_balanced_public_set(set_name, count, *, seed, shifted_family=None):
    """Yield a deterministic round-robin set with exact auditable cell counts."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise PrefixRouteOODError("set count must be a positive integer")
    if set_name in {"TRAIN", "IID_HOLDOUT"}:
        if shifted_family is not None:
            raise PrefixRouteOODError("IID set may not name a shifted family")
        specs = _iid_specs()
    elif set_name == "SINGLE_SHIFT_OOD":
        specs = _single_shift_specs(shifted_family)
    else:
        raise PrefixRouteOODError(
            "compound OOD requires reviewer-owned explicit combination table"
        )
    if not specs:
        raise PrefixRouteOODError("no valid factor cells exist")
    offset = int.from_bytes(_digest_bytes(seed, set_name, shifted_family)[:8], "big")
    offset %= len(specs)
    for index in range(count):
        spec = specs[(offset + index) % len(specs)]
        yield generate_sequence(
            spec,
            seed=seed,
            sequence_index=index,
            set_name=(
                set_name
                if shifted_family is None
                else f"{set_name}:{shifted_family}"
            ),
        )


def audit_sequence_sets(named_sequences):
    """Check hashes, pairwise disjointness, and factor-cell accounting."""

    if not isinstance(named_sequences, dict) or not named_sequences:
        raise PrefixRouteOODError("named sequence sets must be a non-empty object")
    all_hashes = {}
    reports = {}
    for set_name, sequences in sorted(named_sequences.items()):
        hashes = []
        cells = Counter()
        for sequence in sequences:
            supplied = sequence.get("sequence_sha256")
            derived = _scientific_content_sha256(sequence)
            if supplied != derived:
                raise PrefixRouteOODError(
                    f"sequence hash differs in set {set_name}"
                )
            if supplied in all_hashes:
                raise PrefixRouteOODError(
                    f"sequence sets overlap: {set_name} and {all_hashes[supplied]}"
                )
            all_hashes[supplied] = set_name
            hashes.append(supplied)
            cell = tuple(
                sequence["factor_spec"][family] for family in FACTOR_FAMILIES
            )
            cells[cell] += 1
        reports[set_name] = {
            "sequence_count": len(hashes),
            "sequence_set_sha256": hashlib.sha256(
                canonical_json_bytes(sorted(hashes))
            ).hexdigest(),
            "factor_cell_count": len(cells),
            "minimum_cell_count": min(cells.values(), default=0),
            "maximum_cell_count": max(cells.values(), default=0),
        }
    return {
        "pairwise_disjoint": True,
        "total_sequence_count": len(all_hashes),
        "sets": reports,
    }
__all__ = [
    "CLASS_COUNT",
    "FACTOR_FAMILIES",
    "FEATURE_DIM",
    "GENERATOR_SCHEMA",
    "GENERATOR_VERSION",
    "IID_OBSERVATION",
    "IID_SEMANTIC",
    "IID_TEMPORAL",
    "IID_TOPOLOGIES",
    "OOD_OBSERVATION",
    "OOD_SEMANTIC",
    "OOD_TEMPORAL",
    "OOD_TOPOLOGIES",
    "PrefixRouteOODError",
    "SEQUENCE_LENGTH_BINS",
    "audit_sequence_sets",
    "canonical_json_bytes",
    "generate_sequence",
    "iter_balanced_public_set",
]
