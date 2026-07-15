from copy import deepcopy
import json
import math

import pytest

from opentad.utils.crs_eps_sampling import (
    COMPONENTS,
    DEFAULT_MIXTURE,
    CrsEpsSamplingError,
    InstanceTimeline,
    VideoSamplingSpec,
    build_epoch_manifest,
    component_candidates,
    episode_geometry,
    probability_table,
    supervised_window,
    validate_epoch_manifest,
)


def _spec(num_bins=12):
    return VideoSamplingSpec(
        video_id="toy-video",
        num_bins=num_bins,
        instances=(
            InstanceTimeline(
                instance_id=0,
                label=3,
                birth_bin=2,
                end_bin=8,
                active_bins=tuple(range(2, 8)),
            ),
        ),
    )


def test_supervised_window_is_fixed_contains_anchor_and_shifts_at_edges():
    assert supervised_window(0, 12) == (0, 8)
    assert supervised_window(5, 12) == (2, 10)
    assert supervised_window(11, 12) == (4, 12)
    assert supervised_window(2, 5) == (0, 5)


def test_component_candidates_follow_frozen_risk_set_definitions():
    candidates = component_candidates(_spec())

    assert candidates["start"]["candidate_bins"] == (0, 1, 2, 3, 4, 6)
    assert candidates["end"]["candidate_bins"] == (4, 6, 7, 8, 9, 10, 11)
    assert candidates["ongoing"]["candidate_bins"] == (4, 5, 6)
    assert candidates["hardbg"]["candidate_bins"] == (0, 1, 9, 10, 11)
    assert not any(
        candidates[name]["component_fallback_to_uniform"] for name in COMPONENTS
    )


def test_empty_component_falls_back_to_uniform_without_losing_support():
    spec = VideoSamplingSpec("background-only", 6, ())
    candidates = component_candidates(spec)
    table = probability_table(spec, draws_per_video=3)

    for component in ("start", "end", "ongoing", "hardbg"):
        assert candidates[component]["candidate_bins"] == tuple(range(6))
        assert candidates[component]["component_fallback_to_uniform"] is True
    assert all(value > 0 for value in table["rho_by_bin"])
    assert max(table["raw_weight_by_bin"]) <= 2.5 + 1e-12


def _exhaustive_rho(spec, table):
    expected = [0.0] * spec.num_bins
    for component in COMPONENTS:
        row = table["components"][component]
        for anchor in row["candidate_bins"]:
            probability = row["alpha"] * row["q_anchor_given_component"]
            for bin_index in range(*supervised_window(anchor, spec.num_bins)):
                expected[bin_index] += probability
    return expected


def test_analytic_rho_and_union_pi_match_exhaustive_enumeration():
    spec = _spec()
    draws = 4
    table = probability_table(spec, draws_per_video=draws)

    assert table["rho_by_bin"] == pytest.approx(_exhaustive_rho(spec, table))
    for rho, union_pi in zip(table["rho_by_bin"], table["union_pi_by_bin"]):
        assert union_pi == pytest.approx(1.0 - (1.0 - rho) ** draws)


def test_hansen_hurwitz_expectation_recovers_video_uniform_bin_mean():
    spec = _spec()
    table = probability_table(spec, draws_per_video=5)
    losses = [0.1 + value * value / 17.0 for value in range(spec.num_bins)]
    expected = sum(losses) / spec.num_bins
    exhaustive_expectation = 0.0
    for component in COMPONENTS:
        row = table["components"][component]
        for anchor in row["candidate_bins"]:
            proposal = row["alpha"] * row["q_anchor_given_component"]
            start, end = supervised_window(anchor, spec.num_bins)
            exhaustive_expectation += proposal * sum(
                table["raw_weight_by_bin"][bin_index] * losses[bin_index]
                for bin_index in range(start, end)
            )

    assert exhaustive_expectation == pytest.approx(expected, abs=1e-12)


def test_uniform_floor_makes_2_5_a_derived_invariant_not_a_clip():
    table = probability_table(_spec(num_bins=101), draws_per_video=2)

    assert min(table["rho_by_bin"]) >= 0.40 / 101 - 1e-15
    assert max(table["raw_weight_by_bin"]) <= 2.5 + 1e-12
    with pytest.raises(CrsEpsSamplingError, match="sum exactly"):
        probability_table(
            _spec(),
            draws_per_video=2,
            mixture={**DEFAULT_MIXTURE, "uniform": 0.39},
        )


def test_dynamic_context_extends_to_relevant_long_action_birth_and_keeps_global_detach():
    spec = VideoSamplingSpec(
        "long-action",
        400,
        (
            InstanceTimeline(
                instance_id=7,
                label=4,
                birth_bin=10,
                end_bin=350,
                active_bins=tuple(range(10, 350)),
            ),
        ),
    )

    geometry = episode_geometry(spec, 347)

    assert geometry["supervised_range"] == [344, 352]
    assert geometry["replay_range"] == [10, 352]
    assert geometry["dynamic_extension_instance_ids"] == [7]
    assert geometry["gradient_ranges"] == [[320, 352]]
    assert geometry["true_left_censored"] is False
    assert geometry["left_censored_instance_ids"] == []


def test_epoch_manifest_is_deterministic_repeated_exposure_and_self_verifying():
    manifest = build_epoch_manifest(
        [_spec()],
        epoch=3,
        seed=705,
        draws_per_video=6,
        provenance={"commit_sha": "a" * 40},
    )
    repeated = build_epoch_manifest(
        [_spec()],
        epoch=3,
        seed=705,
        draws_per_video=6,
        provenance={"commit_sha": "a" * 40},
    )

    assert manifest == repeated
    validate_epoch_manifest(manifest)
    video = manifest["videos"][0]
    assert video["exposure_count"] == 6 * 8
    assert sum(video["exposure_multiplicity_by_bin"]) == 6 * 8
    assert 0 < video["effective_sample_size"] <= video["exposure_count"]
    assert all(draw["final_weight_by_bin"] == draw["raw_weight_by_bin"] for draw in video["draws"])

    tampered = deepcopy(manifest)
    tampered["videos"][0]["draws"][0]["anchor_bin"] += 1
    with pytest.raises(CrsEpsSamplingError, match="hash"):
        validate_epoch_manifest(tampered)


def test_epoch_manifest_reproduces_after_json_round_trip():
    manifest = build_epoch_manifest(
        [_spec()],
        epoch=3,
        seed=705,
        draws_per_video=6,
        provenance={"commit_sha": "a" * 40},
    )
    persisted = json.loads(json.dumps(manifest, allow_nan=False, sort_keys=True))

    assert persisted == manifest
    validate_epoch_manifest(persisted)


def test_manifest_union_probability_is_diagnostic_not_exposure_weight():
    manifest = build_epoch_manifest([_spec()], epoch=0, seed=1, draws_per_video=3)
    video = manifest["videos"][0]

    for draw in video["draws"]:
        assert draw["final_weight_by_bin"] == draw["raw_weight_by_bin"]
        assert any(
            not math.isclose(raw, union)
            for raw, union in zip(
                draw["raw_weight_by_bin"], draw["union_pi_by_supervised_bin"]
            )
        )
