"""Deterministic CPU mathematics for CRS-EPS episode manifests."""

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from typing import Mapping, Sequence


SCHEMA_VERSION = "full-petal-crs-eps-manifest-v1"
COMPONENTS = ("uniform", "start", "end", "ongoing", "hardbg")
DEFAULT_MIXTURE = {
    "uniform": 0.40,
    "start": 0.15,
    "end": 0.20,
    "ongoing": 0.10,
    "hardbg": 0.15,
}
EVENT_OFFSETS = (-4, -2, -1, 0, 1, 2, 4)
DEFAULT_SUFFIX_BINS = 8
DEFAULT_CONTEXT_BINS = 192
DEFAULT_DETACH_INTERVAL = 64
WEIGHT_TOLERANCE = 1e-12


class CrsEpsSamplingError(ValueError):
    pass


def canonical_json_sha256(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class InstanceTimeline:
    instance_id: int
    label: int
    birth_bin: int | None
    end_bin: int | None
    active_bins: tuple[int, ...]


@dataclass(frozen=True)
class VideoSamplingSpec:
    video_id: str
    num_bins: int
    instances: tuple[InstanceTimeline, ...]


def video_sampling_spec_from_schedule(video_id, schedule):
    schedule = tuple(schedule)
    if not schedule:
        raise CrsEpsSamplingError("a sampling schedule must contain at least one bin")
    records = {}
    for bin_index, step in enumerate(schedule):
        for target in step.births:
            record = records.setdefault(
                int(target.instance_id),
                {"label": int(target.label), "birth_bin": None, "end_bin": None, "active": []},
            )
            if record["birth_bin"] is not None:
                raise CrsEpsSamplingError("an instance has more than one birth crossing")
            record["birth_bin"] = bin_index
        for target in step.active:
            record = records.setdefault(
                int(target.instance_id),
                {"label": int(target.label), "birth_bin": None, "end_bin": None, "active": []},
            )
            record["active"].append(bin_index)
        for target in step.ends:
            record = records.setdefault(
                int(target.instance_id),
                {"label": int(target.label), "birth_bin": None, "end_bin": None, "active": []},
            )
            if record["end_bin"] is not None:
                raise CrsEpsSamplingError("an instance has more than one end crossing")
            record["end_bin"] = bin_index
    instances = tuple(
        InstanceTimeline(
            instance_id=instance_id,
            label=record["label"],
            birth_bin=record["birth_bin"],
            end_bin=record["end_bin"],
            active_bins=tuple(record["active"]),
        )
        for instance_id, record in sorted(records.items())
    )
    spec = VideoSamplingSpec(str(video_id), len(schedule), instances)
    validate_video_sampling_spec(spec)
    return spec


def validate_video_sampling_spec(spec):
    if not isinstance(spec, VideoSamplingSpec):
        raise CrsEpsSamplingError("video sampling spec has the wrong type")
    if not spec.video_id.strip() or spec.num_bins <= 0:
        raise CrsEpsSamplingError("video sampling spec requires an ID and positive bin count")
    seen = set()
    for instance in spec.instances:
        if instance.instance_id in seen:
            raise CrsEpsSamplingError("instance IDs must be unique within a video")
        seen.add(instance.instance_id)
        for value in (instance.birth_bin, instance.end_bin):
            if value is not None and not 0 <= value < spec.num_bins:
                raise CrsEpsSamplingError("instance crossing bin is outside the video")
        if tuple(sorted(set(instance.active_bins))) != instance.active_bins:
            raise CrsEpsSamplingError("instance active bins must be sorted and unique")
        if any(not 0 <= value < spec.num_bins for value in instance.active_bins):
            raise CrsEpsSamplingError("instance active bin is outside the video")
    return spec


def _crossing_bins(spec, field):
    return tuple(
        sorted(
            {
                value
                for instance in spec.instances
                if (value := getattr(instance, field)) is not None
            }
        )
    )


def _active_bins(spec):
    return {value for instance in spec.instances for value in instance.active_bins}


def _offset_candidates(crossings, num_bins):
    return tuple(
        sorted(
            {
                min(max(crossing + offset, 0), num_bins - 1)
                for crossing in crossings
                for offset in EVENT_OFFSETS
            }
        )
    )


def component_candidates(spec):
    validate_video_sampling_spec(spec)
    all_bins = tuple(range(spec.num_bins))
    births = _crossing_bins(spec, "birth_bin")
    ends = _crossing_bins(spec, "end_bin")
    crossings = tuple(sorted(set(births).union(ends)))
    active = _active_bins(spec)
    crossing_set = set(crossings)
    ongoing = tuple(
        bin_index
        for bin_index in all_bins
        if bin_index in active
        and bin_index not in crossing_set
        and all(abs(bin_index - crossing) >= 2 for crossing in crossings)
    )
    hardbg = tuple(
        bin_index
        for bin_index in all_bins
        if bin_index not in active
        and bin_index not in crossing_set
        and any(abs(bin_index - crossing) <= 4 for crossing in crossings)
    )
    raw = {
        "uniform": all_bins,
        "start": _offset_candidates(births, spec.num_bins),
        "end": _offset_candidates(ends, spec.num_bins),
        "ongoing": ongoing,
        "hardbg": hardbg,
    }
    return {
        component: {
            "candidate_bins": raw[component] or all_bins,
            "component_fallback_to_uniform": not bool(raw[component]),
        }
        for component in COMPONENTS
    }


def supervised_window(anchor_bin, num_bins, suffix_bins=DEFAULT_SUFFIX_BINS):
    anchor_bin = int(anchor_bin)
    num_bins = int(num_bins)
    suffix_bins = int(suffix_bins)
    if num_bins <= 0 or suffix_bins <= 0 or not 0 <= anchor_bin < num_bins:
        raise CrsEpsSamplingError("invalid supervised-window geometry")
    width = min(num_bins, suffix_bins)
    start = min(max(anchor_bin - 3, 0), num_bins - width)
    end = start + width
    if not start <= anchor_bin < end:
        raise CrsEpsSamplingError("edge-shifted window does not contain its anchor")
    return start, end


def _validated_mixture(mixture):
    mixture = dict(DEFAULT_MIXTURE if mixture is None else mixture)
    if set(mixture) != set(COMPONENTS):
        raise CrsEpsSamplingError("proposal mixture must define the five frozen components")
    fractions = {name: Fraction(str(mixture[name])) for name in COMPONENTS}
    if any(value <= 0 for value in fractions.values()) or sum(fractions.values()) != 1:
        raise CrsEpsSamplingError("proposal mixture must be positive and sum exactly to one")
    return {name: float(fractions[name]) for name in COMPONENTS}, fractions


def probability_table(spec, *, draws_per_video, suffix_bins=DEFAULT_SUFFIX_BINS, mixture=None):
    validate_video_sampling_spec(spec)
    if isinstance(draws_per_video, bool) or int(draws_per_video) <= 0:
        raise CrsEpsSamplingError("draws_per_video must be a positive integer")
    draws_per_video = int(draws_per_video)
    mixture_values, _ = _validated_mixture(mixture)
    proposals = component_candidates(spec)
    windows = {
        anchor: supervised_window(anchor, spec.num_bins, suffix_bins)
        for anchor in range(spec.num_bins)
    }
    q_marginal = [0.0] * spec.num_bins
    rho = [0.0] * spec.num_bins
    component_rows = {}
    for component in COMPONENTS:
        candidates = proposals[component]["candidate_bins"]
        conditional = 1.0 / len(candidates)
        alpha = mixture_values[component]
        component_rows[component] = {
            "alpha": alpha,
            "candidate_bins": list(candidates),
            "q_anchor_given_component": conditional,
            "component_fallback_to_uniform": proposals[component][
                "component_fallback_to_uniform"
            ],
        }
        for anchor in candidates:
            probability = alpha * conditional
            q_marginal[anchor] += probability
            start, end = windows[anchor]
            for bin_index in range(start, end):
                rho[bin_index] += probability
    raw_weights = []
    union_pi = []
    for bin_index, coverage_probability in enumerate(rho):
        if coverage_probability <= 0:
            raise CrsEpsSamplingError(f"proposal has zero support at bin {bin_index}")
        weight = (1.0 / spec.num_bins) / coverage_probability
        if weight > 2.5 + WEIGHT_TOLERANCE:
            raise CrsEpsSamplingError(
                f"raw IPW exceeds the uniform-floor bound at bin {bin_index}: {weight}"
            )
        raw_weights.append(weight)
        union_pi.append(1.0 - (1.0 - coverage_probability) ** draws_per_video)
    if not math.isclose(sum(q_marginal), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CrsEpsSamplingError("marginal anchor distribution is not normalized")
    return {
        "components": component_rows,
        "q_anchor_marginal": q_marginal,
        "rho_by_bin": rho,
        "union_pi_by_bin": union_pi,
        "raw_weight_by_bin": raw_weights,
    }


def _hash_parts(*parts):
    return hashlib.sha256(
        json.dumps(parts, allow_nan=False, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).digest()


def _randbelow(upper, *parts):
    if upper <= 0:
        raise CrsEpsSamplingError("deterministic sampler received an empty support")
    modulus = 1 << 256
    limit = modulus - (modulus % upper)
    counter = 0
    while True:
        value = int.from_bytes(_hash_parts(*parts, counter), "big")
        if value < limit:
            return value % upper
        counter += 1


def _integer_mixture(fractions):
    denominator = math.lcm(*(value.denominator for value in fractions.values()))
    counts = {
        name: fractions[name].numerator * (denominator // fractions[name].denominator)
        for name in COMPONENTS
    }
    return counts, sum(counts.values())


def _draw_component(fractions, *rng_parts):
    counts, total = _integer_mixture(fractions)
    choice = _randbelow(total, *rng_parts, "component")
    cumulative = 0
    for component in COMPONENTS:
        cumulative += counts[component]
        if choice < cumulative:
            return component
    raise CrsEpsSamplingError("component sampler escaped its support")


def _gradient_ranges(supervised_range, detach_interval):
    start, end = supervised_range
    gradient_start = (start // detach_interval) * detach_interval
    ranges = []
    cursor = gradient_start
    while cursor < end:
        next_boundary = min(end, ((cursor // detach_interval) + 1) * detach_interval)
        ranges.append([cursor, next_boundary])
        cursor = next_boundary
    return ranges


def episode_geometry(
    spec,
    anchor_bin,
    *,
    suffix_bins=DEFAULT_SUFFIX_BINS,
    context_bins=DEFAULT_CONTEXT_BINS,
    detach_interval=DEFAULT_DETACH_INTERVAL,
):
    supervised_range = supervised_window(anchor_bin, spec.num_bins, suffix_bins)
    supervised_bins = set(range(*supervised_range))
    relevant = []
    relevant_has_unobservable_birth = False
    for instance in spec.instances:
        is_relevant = bool(supervised_bins.intersection(instance.active_bins)) or (
            instance.end_bin in supervised_bins if instance.end_bin is not None else False
        )
        if is_relevant:
            relevant.append(instance)
            relevant_has_unobservable_birth = (
                relevant_has_unobservable_birth or instance.birth_bin is None
            )
    base_start = max(0, supervised_range[0] - int(context_bins))
    observable_births = [
        instance.birth_bin for instance in relevant if instance.birth_bin is not None
    ]
    replay_start = min([base_start, *observable_births]) if observable_births else base_start
    if relevant_has_unobservable_birth:
        replay_start = 0
    left_censored_ids = [
        instance.instance_id
        for instance in spec.instances
        if replay_start in instance.active_bins
        and (instance.birth_bin is None or instance.birth_bin < replay_start)
    ]
    extended_ids = [
        instance.instance_id
        for instance in relevant
        if instance.birth_bin is not None and instance.birth_bin < base_start
    ]
    return {
        "supervised_range": list(supervised_range),
        "replay_range": [replay_start, supervised_range[1]],
        "gradient_ranges": _gradient_ranges(supervised_range, int(detach_interval)),
        "true_left_censored": bool(left_censored_ids),
        "left_censored_instance_ids": left_censored_ids,
        "dynamic_extension_instance_ids": extended_ids,
        "video_start_fallback": replay_start == 0,
    }


def _effective_sample_size(weights):
    if not weights:
        return 0.0
    total = sum(weights)
    denominator = sum(value * value for value in weights)
    return total * total / denominator if denominator else 0.0


def _bin_diagnostics(spec):
    diagnostics = []
    for bin_index in range(spec.num_bins):
        births = [item for item in spec.instances if item.birth_bin == bin_index]
        ends = [item for item in spec.instances if item.end_bin == bin_index]
        active = [item for item in spec.instances if bin_index in item.active_bins]
        if births and ends:
            lifecycle = "birth_end"
        elif births:
            lifecycle = "birth"
        elif ends:
            lifecycle = "end"
        elif active:
            lifecycle = "ongoing"
        else:
            lifecycle = "background"
        labels = sorted({item.label for item in (*births, *active, *ends)})
        diagnostics.append((lifecycle, labels or ["background"]))
    return diagnostics


def build_video_manifest(
    spec,
    *,
    epoch,
    seed,
    draws_per_video,
    suffix_bins=DEFAULT_SUFFIX_BINS,
    context_bins=DEFAULT_CONTEXT_BINS,
    detach_interval=DEFAULT_DETACH_INTERVAL,
    mixture=None,
):
    validate_video_sampling_spec(spec)
    for value, label in ((epoch, "epoch"), (seed, "seed"), (draws_per_video, "draws")):
        if isinstance(value, bool) or int(value) < (1 if label == "draws" else 0):
            raise CrsEpsSamplingError(f"{label} must be a non-negative integer")
    epoch, seed, draws_per_video = int(epoch), int(seed), int(draws_per_video)
    mixture_values, fractions = _validated_mixture(mixture)
    table = probability_table(
        spec,
        draws_per_video=draws_per_video,
        suffix_bins=suffix_bins,
        mixture=mixture_values,
    )
    multiplicity = [0] * spec.num_bins
    exposures = []
    draws = []
    for draw_index in range(draws_per_video):
        rng_parts = (SCHEMA_VERSION, seed, epoch, spec.video_id, draw_index)
        component = _draw_component(fractions, *rng_parts)
        component_row = table["components"][component]
        candidates = component_row["candidate_bins"]
        anchor = candidates[_randbelow(len(candidates), *rng_parts, "anchor")]
        geometry = episode_geometry(
            spec,
            anchor,
            suffix_bins=suffix_bins,
            context_bins=context_bins,
            detach_interval=detach_interval,
        )
        bins = list(range(*geometry["supervised_range"]))
        weights = [table["raw_weight_by_bin"][value] for value in bins]
        for bin_index, weight in zip(bins, weights):
            multiplicity[bin_index] += 1
            exposures.append((bin_index, weight))
        rng_key = canonical_json_sha256(list(rng_parts))
        draws.append(
            {
                "draw_index": draw_index,
                "episode_id": f"{spec.video_id}:e{epoch}:d{draw_index}:{rng_key[:16]}",
                "proposal_component": component,
                "component_fallback_to_uniform": component_row[
                    "component_fallback_to_uniform"
                ],
                "anchor_bin": anchor,
                **geometry,
                "q_component": component_row["alpha"],
                "q_anchor_given_component": component_row[
                    "q_anchor_given_component"
                ],
                "q_anchor_marginal": table["q_anchor_marginal"][anchor],
                "rho_by_supervised_bin": [table["rho_by_bin"][value] for value in bins],
                "union_pi_by_supervised_bin": [
                    table["union_pi_by_bin"][value] for value in bins
                ],
                "raw_weight_by_bin": weights,
                "final_weight_by_bin": weights,
                "rng_key": rng_key,
            }
        )
    bin_diagnostics = _bin_diagnostics(spec)
    lifecycle_weights = {}
    class_weights = {}
    for bin_index, weight in exposures:
        lifecycle, labels = bin_diagnostics[bin_index]
        lifecycle_weights.setdefault(lifecycle, []).append(weight)
        for label in labels:
            class_weights.setdefault(str(label), []).append(weight)
    return {
        "video_id": spec.video_id,
        "num_bins": spec.num_bins,
        "instances": [asdict(instance) for instance in spec.instances],
        "draws_per_video": draws_per_video,
        "proposal_table": table,
        "draws": draws,
        "exposure_multiplicity_by_bin": multiplicity,
        "covered_unique_bins": sum(value > 0 for value in multiplicity),
        "coverage_fraction": sum(value > 0 for value in multiplicity) / spec.num_bins,
        "exposure_count": len(exposures),
        "effective_sample_size": _effective_sample_size(
            [weight for _, weight in exposures]
        ),
        "ipw_weight_sum": sum(weight for _, weight in exposures),
        "ipw_weight_squared_sum": sum(weight * weight for _, weight in exposures),
        "effective_sample_size_by_lifecycle": {
            key: _effective_sample_size(values)
            for key, values in sorted(lifecycle_weights.items())
        },
        "effective_sample_size_by_class": {
            key: _effective_sample_size(values)
            for key, values in sorted(class_weights.items())
        },
    }


def build_epoch_manifest(
    specs: Sequence[VideoSamplingSpec],
    *,
    epoch,
    seed,
    draws_per_video,
    provenance: Mapping | None = None,
    suffix_bins=DEFAULT_SUFFIX_BINS,
    context_bins=DEFAULT_CONTEXT_BINS,
    detach_interval=DEFAULT_DETACH_INTERVAL,
    mixture=None,
):
    specs = tuple(sorted(specs, key=lambda item: item.video_id))
    if not specs or len({item.video_id for item in specs}) != len(specs):
        raise CrsEpsSamplingError("epoch manifest requires unique non-empty videos")
    mixture_values, _ = _validated_mixture(mixture)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bin_index_convention": "zero_based_half_open_ranges",
        "target_risk": "video_uniform_then_decision_bin_uniform",
        "estimator": "hansen_hurwitz_repeated_exposure_uncapped_ipw",
        "epoch": int(epoch),
        "seed": int(seed),
        "draws_per_video": int(draws_per_video),
        "suffix_bins": int(suffix_bins),
        "context_bins": int(context_bins),
        "detach_interval": int(detach_interval),
        "mixture": mixture_values,
        "provenance": dict(provenance or {}),
        "videos": [
            build_video_manifest(
                spec,
                epoch=epoch,
                seed=seed,
                draws_per_video=draws_per_video,
                suffix_bins=suffix_bins,
                context_bins=context_bins,
                detach_interval=detach_interval,
                mixture=mixture_values,
            )
            for spec in specs
        ],
    }
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _spec_from_manifest(row):
    return VideoSamplingSpec(
        video_id=row["video_id"],
        num_bins=int(row["num_bins"]),
        instances=tuple(
            InstanceTimeline(
                instance_id=int(item["instance_id"]),
                label=int(item["label"]),
                birth_bin=None if item["birth_bin"] is None else int(item["birth_bin"]),
                end_bin=None if item["end_bin"] is None else int(item["end_bin"]),
                active_bins=tuple(int(value) for value in item["active_bins"]),
            )
            for item in row["instances"]
        ),
    )


def validate_epoch_manifest(manifest):
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CrsEpsSamplingError("unsupported CRS-EPS manifest schema")
    payload = dict(manifest)
    claimed_hash = payload.pop("manifest_sha256", None)
    if claimed_hash != canonical_json_sha256(payload):
        raise CrsEpsSamplingError("CRS-EPS manifest content hash does not verify")
    specs = tuple(_spec_from_manifest(row) for row in payload["videos"])
    rebuilt = build_epoch_manifest(
        specs,
        epoch=payload["epoch"],
        seed=payload["seed"],
        draws_per_video=payload["draws_per_video"],
        provenance=payload["provenance"],
        suffix_bins=payload["suffix_bins"],
        context_bins=payload["context_bins"],
        detach_interval=payload["detach_interval"],
        mixture=payload["mixture"],
    )
    if rebuilt != dict(manifest):
        raise CrsEpsSamplingError("CRS-EPS manifest does not reproduce exactly")
    return manifest


__all__ = [
    "COMPONENTS",
    "DEFAULT_CONTEXT_BINS",
    "DEFAULT_DETACH_INTERVAL",
    "DEFAULT_MIXTURE",
    "DEFAULT_SUFFIX_BINS",
    "CrsEpsSamplingError",
    "InstanceTimeline",
    "VideoSamplingSpec",
    "build_epoch_manifest",
    "build_video_manifest",
    "canonical_json_sha256",
    "component_candidates",
    "episode_geometry",
    "probability_table",
    "supervised_window",
    "validate_epoch_manifest",
    "video_sampling_spec_from_schedule",
]
