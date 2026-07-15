from copy import deepcopy
from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from opentad.models.dense_heads.persistent_event_set_head import (
    SLOT_ACTIVE,
    SLOT_FREE,
    SLOT_REFRACTORY,
    PersistentEventSetHead,
)
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
)
from opentad.utils.crs_eps_audit import (
    _pair_vector_metrics,
    _runtime_record,
    CrsEpsAuditError,
    capture_rng_snapshot,
    rng_snapshot_digest,
    run_matched_crs_eps_pair,
)
from opentad.utils.crs_eps_sampling import (
    canonical_json_sha256,
    episode_payload_sha256,
)
from opentad.utils.prefix_instance_schedule import build_prefix_instance_schedule


def _model(binding):
    head = PersistentEventSetHead(
        in_channels=4,
        hidden_dim=8,
        num_classes=3,
        num_slots=2,
        memory_size=4,
        num_heads=2,
        dropout=0.35,
        query_mode="persistent",
        start_mode="scalar",
        endpoint_mode="binary",
        birth_threshold=1.1,
        alive_threshold=1.1,
        end_threshold=1.1,
        refractory_steps=1,
    )
    return PersistentTrajectoryOnlineDetector(
        head=head,
        trajectory_binding_mode=binding,
        detach_stream_state=True,
    ).train()


def _episode(stream_id="audit"):
    frames = tuple(range(8))
    payload = {
        "draw_index": 0,
        "episode_id": stream_id,
        "proposal_component": "uniform",
        "component_fallback_to_uniform": False,
        "anchor_bin": 4,
        "supervised_range": [0, 8],
        "replay_range": [0, 8],
        "gradient_ranges": [[0, 8]],
        "true_left_censored": False,
        "left_censored_instance_ids": [],
        "dynamic_extension_instance_ids": [],
        "video_start_fallback": True,
        "q_component": 0.4,
        "q_anchor_given_component": 0.125,
        "q_anchor_marginal": 0.125,
        "rho_by_supervised_bin": [1.0] * 8,
        "union_pi_by_supervised_bin": [1.0] * 8,
        "raw_weight_by_bin": [0.125] * 8,
        "final_weight_by_bin": [0.125] * 8,
        "rng_key": "rng-audit",
    }
    payload["episode_payload_sha256"] = episode_payload_sha256(payload)
    return {
        "inputs": torch.randn(1, 4, 8),
        "masks": torch.ones(1, 8, dtype=torch.bool),
        "model_meta": {
            "video_name": "video",
            "video_id": "video",
            "stream_id": stream_id,
            "input_format": "cached_features",
            "input_provenance_digest": "a" * 64,
            "feature_stride": 1,
            "fps": 30.0,
            "source_frames": frames,
            "current_frame": frames[-1],
            "packet_start_token": 0,
            "packet_end_token": 8,
        },
        "supervision_schedule": build_prefix_instance_schedule(
            [[1.0, 6.0]], [1], frames, previous_frame=-1
        ),
        "crs_eps_control": {
            **payload,
            "video_group_size": 1,
            "is_video_group_start": True,
            "is_video_group_end": True,
            "episode_manifest_sha256": "b" * 64,
            "episode_sequence_sha256": canonical_json_sha256(
                [payload["episode_payload_sha256"]]
            ),
        },
    }


def _runtime(slot_status, start_state):
    num_slots = len(slot_status)
    return SimpleNamespace(
        queries=torch.zeros(1, num_slots, 2),
        feature_memory=torch.zeros(1, 0, 2),
        slot_status=torch.tensor(slot_status, dtype=torch.long),
        refractory=torch.zeros(num_slots, dtype=torch.long),
        start_state=torch.tensor(start_state, dtype=torch.float32),
        score_state=torch.zeros(num_slots),
        label_state=torch.full((num_slots,), -1, dtype=torch.long),
        source_frames=(),
        last_decision_frame=-1,
    )


def test_binding_twin_restores_rng_and_closes_identical_single_instance_trace():
    torch.manual_seed(17)
    fixed = _model("fixed_birth_slot")
    rematch = _model("prefix_rematch_active_pool")
    rematch.load_state_dict(fixed.state_dict())
    episode = _episode()
    rng_before = rng_snapshot_digest(capture_rng_snapshot())

    trace = run_matched_crs_eps_pair(
        fixed,
        rematch,
        episode,
        comparison_type="binding_twin",
    )

    assert rng_snapshot_digest(capture_rng_snapshot()) == rng_before
    assert trace["logits_equal"] is True
    assert trace["runtime_discrete_equal"] is True
    assert trace["runtime_continuous_comparison"]["cosine"] == pytest.approx(1.0)
    assert trace["left"]["runtime"]["continuous_norm"] >= 0.0
    assert trace["gradient_comparison"]["cosine"] == pytest.approx(1.0)
    assert trace["gradient_comparison"]["sign_agreement"] == pytest.approx(1.0)
    assert trace["left"]["losses"] == pytest.approx(trace["right"]["losses"])
    assert len(trace["trace_sha256"]) == 64


def test_pair_vector_metrics_are_stable_for_large_values_and_reject_nonfinite():
    values = torch.tensor([1e30, -1e30], dtype=torch.float32)

    stable = _pair_vector_metrics(values, values.clone())
    invalid = _pair_vector_metrics(values, torch.tensor([float("nan"), 0.0]))

    assert stable["comparable"] is True
    assert stable["cosine"] == pytest.approx(1.0)
    assert stable["sign_agreement"] == pytest.approx(1.0)
    assert invalid == {
        "comparable": False,
        "cosine": None,
        "norm_ratio": None,
        "sign_agreement": None,
    }


def test_runtime_record_accepts_only_lifecycle_consistent_start_sentinels():
    runtime = _runtime(
        [SLOT_FREE, SLOT_REFRACTORY, SLOT_ACTIVE],
        [float("nan"), float("nan"), 2.0],
    )

    record = _runtime_record(runtime)

    assert record["discrete"]["start_state_nan_mask"] == [True, True, False]
    assert record["continuous_norm"] >= 0.0


@pytest.mark.parametrize(
    ("slot_status", "start_state", "active_nan", "inactive_finite"),
    [
        ([SLOT_ACTIVE], [float("nan")], 1, 0),
        ([SLOT_FREE], [1.0], 0, 1),
        ([SLOT_REFRACTORY], [1.0], 0, 1),
    ],
)
def test_runtime_record_rejects_start_state_lifecycle_mismatch(
    slot_status, start_state, active_nan, inactive_finite
):
    with pytest.raises(
        CrsEpsAuditError,
        match=(
            rf"start_state violates slot lifecycle: active_nan={active_nan}, "
            rf"inactive_finite={inactive_finite}"
        ),
    ):
        _runtime_record(_runtime(slot_status, start_state))


def test_runtime_record_rejects_invalid_slot_status_and_shape():
    with pytest.raises(CrsEpsAuditError, match="invalid lifecycle values"):
        _runtime_record(_runtime([3], [float("nan")]))

    malformed = _runtime([SLOT_FREE], [float("nan")])
    malformed.start_state = torch.tensor([float("nan"), float("nan")])
    with pytest.raises(CrsEpsAuditError, match="share one-dimensional shape"):
        _runtime_record(malformed)


def test_matched_pair_rejects_illegal_runtime_before_emitting_trace():
    torch.manual_seed(29)
    left = _model("fixed_birth_slot")
    right = _model("fixed_birth_slot")
    right.load_state_dict(left.state_dict())
    original = left.train_crs_eps_episode

    def corrupt_runtime(self, **kwargs):
        output = original(**kwargs)
        slot_status = output.runtime_state.slot_status.clone()
        start_state = output.runtime_state.start_state.clone()
        slot_status[0] = SLOT_ACTIVE
        start_state[0] = float("nan")
        return replace(
            output,
            runtime_state=replace(
                output.runtime_state,
                slot_status=slot_status,
                start_state=start_state,
            ),
        )

    left.train_crs_eps_episode = MethodType(corrupt_runtime, left)

    with pytest.raises(CrsEpsAuditError, match="start_state violates slot lifecycle"):
        run_matched_crs_eps_pair(
            left,
            right,
            _episode(),
            comparison_type="replay_fidelity",
        )


def test_matched_pair_rejects_parameter_drift_before_forward():
    torch.manual_seed(19)
    left = _model("fixed_birth_slot")
    right = _model("fixed_birth_slot")
    right.load_state_dict(left.state_dict())
    with torch.no_grad():
        next(right.parameters()).add_(1.0)

    with pytest.raises(CrsEpsAuditError, match="identical initial state"):
        run_matched_crs_eps_pair(
            left,
            right,
            _episode(),
            comparison_type="replay_fidelity",
        )


def test_replay_fidelity_pair_reports_state_shape_mismatch_without_false_cosine():
    torch.manual_seed(23)
    gold = _model("fixed_birth_slot")
    surrogate = _model("fixed_birth_slot")
    surrogate.load_state_dict(gold.state_dict())
    gold_episode = _episode("gold")
    surrogate_episode = deepcopy(gold_episode)
    surrogate_episode["model_meta"]["stream_id"] = "surrogate"
    surrogate_episode["crs_eps_control"]["episode_id"] = "surrogate"
    surrogate_episode["crs_eps_control"]["episode_payload_sha256"] = (
        episode_payload_sha256(surrogate_episode["crs_eps_control"])
    )
    surrogate_episode["crs_eps_control"]["episode_sequence_sha256"] = (
        canonical_json_sha256(
            [surrogate_episode["crs_eps_control"]["episode_payload_sha256"]]
        )
    )

    trace = run_matched_crs_eps_pair(
        gold,
        surrogate,
        gold_episode,
        surrogate_episode,
        comparison_type="replay_fidelity",
    )

    assert trace["gradient_comparison"]["comparable"] is True
    assert trace["runtime_continuous_comparison"]["comparable"] is True
