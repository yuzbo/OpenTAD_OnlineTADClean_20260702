"""Frozen-checkpoint, train-only owner counterfactual for EventMATR D1.5.

One GT-free MATR query stream is computed once for every physical batch.  Four
isolated lifecycle channels consume that stream:

* PF: predicted admission, free persistent owner;
* PR: predicted admission, post-forward oracle owner refresh;
* OF: one oracle-visible admission per training event, free owner;
* OR: oracle-visible admission and post-forward oracle owner refresh.

Each channel has an independent recurrent no-cancel shadow.  Ground truth is
consulted only after the query forward, is stored only in Python diagnostic
sidecars, and is never written into ``EventRecord.target_event_id``.  This
script emits structural diagnostics only; it never computes a paper metric.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset import THUMOS14Dataset  # noqa: E402
from models import build_model  # noqa: E402
from models.event_memory import DynamicEventMemory, EventRecord  # noqa: E402
from on_tal_task import (  # noqa: E402
    D1_RUNTIME_FORBIDDEN_MODEL_INFO,
    make_model_inputs,
    validate_d1_checkpoint_compatibility,
)
from scripts.run_eventmatr_d11_association_scan import (  # noqa: E402
    _git,
    _metadata_values,
    _sha256,
    _validate_inputs,
    _validate_ledger_snapshot,
)
from scripts.eventmatr_d15_contracts import (  # noqa: E402
    TargetView,
    deterministic_target_query_assignment,
    validate_parallel_window_causality,
)
from util.utils import memory_initialize, parrallel_collate_fn  # noqa: E402


PROTOCOL = "eventmatr_d15_owner_counterfactual_v1"
CHANNELS = {
    "PF": ("predicted", "free"),
    "PR": ("predicted", "refreshed"),
    "OF": ("oracle_visible", "free"),
    "OR": ("oracle_visible", "refreshed"),
}
ROUTES = ("formal", "shadow")
EXPECTED_D14_COUNTS = {
    "birth_count": 30002,
    "cancellation_count": 29974,
    "end_count": 0,
    "emit_count": 0,
    "reacquisition_count": 0,
    "capacity_exhaustion_count": 0,
}
EXPECTED_COMPLETE_CENSUS = {
    "parallel_window_causality_batch_count": 3270,
    "real_prefix_count": 203363,
    "padding_prefix_count": 5917,
    "visible_birth_target_count": 3003,
    "observed_eos_count": 200,
}
OFFICIAL_TRAIN_ARTIFACTS = {
    "annotation": {
        "bytes": 1575585,
        "sha256": "8eb3e61cc758bcc08aea1d17cfbf1acb2fed8c2a51ed884116d766e9e4c04e66",
    },
    "proposal_labels": {
        "bytes": 277325536,
        "sha256": "ba7dfb10614cfacd1b26f3d50c2ffa41ae2c7fbce62e946777c217adaeebd22f",
    },
    "train_features": {
        "bytes": 3331932341,
        "sha256": "d4660b31b8c6c00d48b590936b3574ab650423ede9b42016fa1d9dda5d45ac9b",
    },
    "video_len": {
        "bytes": 7063,
        "sha256": "0fcc70d555af6198e7b22850aebe56998e9184c2be3f81a16e7a56667f7b9fd8",
    },
}
QUERY_HASH_FIELDS = (
    "pred_cls",
    "pred_reg",
    "pred_stcls",
    "event_birth_logits",
    "event_alive_logits",
    "event_end_logits",
    "event_state_logits",
    "event_end_offsets",
    "event_query_features",
    "event_candidate_start_frames",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--options", required=True, type=Path)
    parser.add_argument("--training-source-identity", required=True, type=Path)
    parser.add_argument("--d14-structure-gate", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-training-source-commit", required=True)
    parser.add_argument("--expected-training-source-tree", required=True)
    parser.add_argument("--expected-d14-source-commit", required=True)
    parser.add_argument("--expected-d14-source-tree", required=True)
    parser.add_argument("--expected-diagnostic-source-commit", required=True)
    parser.add_argument("--expected-diagnostic-source-tree", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-options-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", default=0, type=int)
    args = parser.parse_args()
    for path_field in (
        "checkpoint",
        "options",
        "training_source_identity",
        "d14_structure_gate",
        "manifest",
    ):
        path = getattr(args, path_field).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{path_field} does not exist: {path}")
        setattr(args, path_field, path)
    for path_field in ("output", "trace_output"):
        path = getattr(args, path_field).expanduser().resolve()
        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            pass
        else:
            raise SystemExit(
                f"--{path_field.replace('_', '-')} must be outside the source repository"
            )
        if path.exists():
            raise SystemExit(
                f"--{path_field.replace('_', '-')} already exists; "
                "D1.5 evidence is append-only"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        setattr(args, path_field, path)
    if args.output == args.trace_output:
        raise SystemExit("--output and --trace-output must differ")
    if not args.trace_output.name.endswith(".jsonl.gz"):
        raise SystemExit("--trace-output must end with .jsonl.gz")
    if args.max_batches < 0:
        raise SystemExit("--max-batches must be non-negative")
    return args


def _load_json(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload


def _validate_registered_source_identities(
    manifest: dict,
    *,
    training_source_commit: str,
    training_source_tree: str,
    d14_source_commit: str,
    d14_source_tree: str,
) -> None:
    if manifest.get("protocol_id") != "eventmatr_d1_preexperiments_v8":
        raise RuntimeError("D1.5 manifest protocol is not the frozen v8 contract")
    expected_training = {
        "commit": training_source_commit,
        "tree": training_source_tree,
    }
    if manifest.get("base_training_source") != expected_training:
        raise RuntimeError("D1.5 registered training source identity mismatch")
    d15_rows = [
        gate
        for gate in manifest.get("gates", [])
        if gate.get("stage") == "d15_frozen_owner_counterfactual"
    ]
    if len(d15_rows) != 1:
        raise RuntimeError("D1.5 manifest must contain exactly one frozen D1.5 gate")
    expected_d14 = {
        "commit": d14_source_commit,
        "tree": d14_source_tree,
    }
    if d15_rows[0].get("d14_gate_source") != expected_d14:
        raise RuntimeError("D1.5 registered D1.4 source identity mismatch")


def _finite_number(value, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(numeric):
        raise RuntimeError(f"{label} is not finite: {value!r}")
    return numeric


def _nonnegative_integer(value, label: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{label} is not a count")
    numeric = _finite_number(value, label)
    if numeric < 0 or not numeric.is_integer():
        raise RuntimeError(f"{label} is not a finite non-negative integer")
    return int(numeric)


def _validate_d14_gate(
    gate: dict,
    *,
    checkpoint: Path,
    options: Path,
    checkpoint_sha256: str,
    options_sha256: str,
    d14_source_commit: str,
    d14_source_tree: str,
) -> dict:
    expected = {
        "protocol": "eventmatr_d14_cross_arm_structure_gate_v1",
        "status": "FAIL_STRUCTURE_GATE",
        "selected_variant": None,
        "test_access": False,
        "threshold_search": False,
        "threshold_lowering": False,
        "multi_seed": False,
        "raw_rgb_training": False,
        "strict_causal_paper_result_valid": False,
        "official_paper_performance_valid": False,
        "locked_test_release": False,
        "paper_claim_release": False,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise RuntimeError(
                f"D1.4 structure gate {key} mismatch: {gate.get(key)!r} != {value!r}"
            )
    source = gate.get("source_identity", {}).get("d14", {})
    if source.get("commit") != d14_source_commit:
        raise RuntimeError("D1.4 gate source commit mismatch")
    if source.get("tree") != d14_source_tree:
        raise RuntimeError("D1.4 gate source tree mismatch")
    official = gate.get("official_train_artifacts")
    if official != OFFICIAL_TRAIN_ARTIFACTS:
        raise RuntimeError("D1.4 gate official train artifact identity drifted")

    bag = gate.get("arms", {}).get("decision_aligned_bag")
    if not isinstance(bag, dict):
        raise RuntimeError("D1.4 gate omitted decision-aligned evidence")
    lifecycle = bag.get("terminal_lifecycle", {})
    expected_lifecycle = {
        "runtime_birth_count": EXPECTED_D14_COUNTS["birth_count"],
        "runtime_cancel_count": EXPECTED_D14_COUNTS["cancellation_count"],
        "runtime_end_count": EXPECTED_D14_COUNTS["end_count"],
        "runtime_emit_count": EXPECTED_D14_COUNTS["emit_count"],
        "runtime_reacquisition_count": EXPECTED_D14_COUNTS["reacquisition_count"],
        "runtime_capacity_exhaustion_count": EXPECTED_D14_COUNTS[
            "capacity_exhaustion_count"
        ],
    }
    for key, value in expected_lifecycle.items():
        if _nonnegative_integer(lifecycle.get(key), f"D1.4 {key}") != value:
            raise RuntimeError(f"D1.4 decision-aligned lifecycle drifted: {key}")
    if bag.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError("D1.4 gate checkpoint SHA-256 mismatch")

    linked = gate.get("linked_training_artifacts", {}).get("bag", {})
    expected_linked = {
        "checkpoint": (checkpoint, checkpoint_sha256),
        "options": (options, options_sha256),
    }
    for name, (path, digest) in expected_linked.items():
        row = linked.get(name)
        if not isinstance(row, dict):
            raise RuntimeError(f"D1.4 gate omitted linked {name}")
        if Path(row.get("path", "")).expanduser().resolve() != path:
            raise RuntimeError(f"D1.4 gate linked {name} path mismatch")
        if row.get("sha256") != digest or _sha256(path) != digest:
            raise RuntimeError(f"D1.4 gate linked {name} digest mismatch")
        if _nonnegative_integer(row.get("bytes"), f"linked {name}.bytes") != (
            path.stat().st_size
        ):
            raise RuntimeError(f"D1.4 gate linked {name} size mismatch")
    return {
        "path": str(checkpoint),
        "source_identity": {
            "commit": d14_source_commit,
            "tree": d14_source_tree,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "options_sha256": options_sha256,
        "terminal_lifecycle": expected_lifecycle,
    }


def _parse_targets(
    target_rows: torch.Tensor,
    valid_mask: torch.Tensor,
) -> list[TargetView]:
    result = []
    for row in target_rows[valid_mask]:
        states = tuple(bool(row[index].item()) for index in (5, 6, 7))
        if sum(states) != 1:
            raise RuntimeError(f"event target has non-exclusive lifecycle state: {states}")
        values = [float(row[index].item()) for index in range(5)]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("event target contains non-finite metadata")
        result.append(
            TargetView(
                event_id=int(row[0].item()),
                class_id=int(row[1].item()),
                start_frame=float(row[2].item()),
                end_frame=float(row[3].item()),
                admission_frame=float(row[4].item()),
                is_birth=states[0],
                is_alive=states[1],
                is_end=states[2],
            )
        )
    if len({target.event_id for target in result}) != len(result):
        raise RuntimeError("one prefix duplicated a target event id")
    return sorted(result, key=lambda target: target.event_id)


@dataclass(frozen=True)
class DiagnosticOwnerLink:
    target_event_id: int
    target_class_id: int
    target_start_frame: float
    target_end_frame: float
    association_frame: float
    birth_embedding: torch.Tensor
    late: bool


@dataclass
class StreamingStats:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    positive_count: int = 0

    def add(self, value: float) -> None:
        value = _finite_number(value, "streaming statistic")
        self.count += 1
        self.total += value
        self.total_square += value * value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.positive_count += int(value > 0.0)

    def summary(self) -> dict:
        if not self.count:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
            "positive_count": self.positive_count,
        }


@dataclass
class RouteState:
    channel: str
    route: str
    memory: DynamicEventMemory
    links: dict[tuple[str, int], DiagnosticOwnerLink] = field(default_factory=dict)
    birth_embeddings: dict[tuple[str, int], torch.Tensor] = field(
        default_factory=dict
    )
    created_frames: dict[tuple[str, int], float] = field(default_factory=dict)
    oracle_admitted_targets: set[tuple[str, int]] = field(default_factory=set)
    created_records: set[tuple[str, int]] = field(default_factory=set)
    first_decisions: set[tuple[str, int]] = field(default_factory=set)
    target_link_history: Counter = field(default_factory=Counter)
    counters: Counter = field(default_factory=Counter)
    created_by_source: Counter = field(default_factory=Counter)
    first_state_counts: Counter = field(default_factory=Counter)
    winner_counts: Counter = field(default_factory=Counter)
    transition_counts: Counter = field(default_factory=Counter)
    ledger_snapshots: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    query_consumption_digest: object = field(default_factory=hashlib.sha256)
    stats: dict[str, StreamingStats] = field(
        default_factory=lambda: {
            "end_margin": StreamingStats(),
            "owner_attention_entropy": StreamingStats(),
            "owner_to_birth_cosine": StreamingStats(),
            "owner_to_oracle_query_cosine": StreamingStats(),
            "lifetime_at_cancel": StreamingStats(),
            "lifetime_at_end": StreamingStats(),
            "lifetime_at_emit": StreamingStats(),
        }
    )


def _new_memory(reference: DynamicEventMemory) -> DynamicEventMemory:
    return DynamicEventMemory(
        birth_mode=reference.birth_mode,
        ownership_mode=reference.ownership_mode,
        birth_logit_threshold=reference.birth_logit_threshold,
        end_logit_threshold=reference.end_logit_threshold,
        min_duration_frames=reference.min_duration_frames,
        emit_delay_frames=reference.emit_delay_frames,
        resource_limit=reference.resource_limit,
        segment_size=reference.segment_size,
        enable_reacquisition=reference.enable_reacquisition,
        strict_causal_boundary=reference.strict_causal_boundary,
        owner_state_count=reference.owner_state_count,
    )


def _active_records(memory: DynamicEventMemory, video_name: str) -> list[EventRecord]:
    return [
        record
        for record in memory.records(video_name)
        if record.status == "active"
    ]


def _record_lookup(
    memory: DynamicEventMemory, video_name: str
) -> dict[int, EventRecord]:
    return {
        int(record.event_id): record
        for record in memory.records(video_name)
    }


def _assert_memory_has_no_gt_runtime_ids(
    memory: DynamicEventMemory, *, label: str
) -> None:
    for records in memory._records.values():
        for record in records:
            if record.target_event_id is not None:
                raise RuntimeError(f"{label} stored GT in EventRecord")
    for records in memory._cancelled_records.values():
        for record in records:
            if record.target_event_id is not None:
                raise RuntimeError(f"{label} archived GT in EventRecord")


def _assert_no_gt_runtime_ids(state: RouteState) -> None:
    _assert_memory_has_no_gt_runtime_ids(
        state.memory, label=f"{state.channel}/{state.route}"
    )


def _map_unlinked_records(
    state: RouteState,
    *,
    video_name: str,
    frame: float,
    targets: list[TargetView],
    target_query_map: dict[int, int],
    query_features: torch.Tensor,
) -> None:
    if CHANNELS[state.channel][1] != "refreshed" or not targets:
        return
    active = _active_records(state.memory, video_name)
    active_ids = {int(record.event_id) for record in active}
    claimed_targets = {
        link.target_event_id
        for (name, event_id), link in state.links.items()
        if name == video_name and event_id in active_ids
    }
    records = [
        record
        for record in active
        if (video_name, int(record.event_id)) not in state.links
    ]
    candidates_targets = [
        target
        for target in targets
        if target.event_id not in claimed_targets
        and target.event_id in target_query_map
    ]
    if not records or not candidates_targets:
        return

    candidates = []
    for record in records:
        for target in candidates_targets:
            query_index = target_query_map[target.event_id]
            target_query = query_features[query_index]
            embedding_score = F.cosine_similarity(
                record.owner_embedding.to(target_query).reshape(1, -1),
                target_query.reshape(1, -1),
            )[0]
            class_score = record.class_distribution[
                int(target.class_id)
            ].clamp_min(1e-8).log()
            start_score = -abs(record.start_frame - target.start_frame) / float(
                state.memory.segment_size
            )
            candidates.append(
                (
                    float((embedding_score + class_score + start_score).item()),
                    int(record.event_id),
                    int(target.event_id),
                )
            )

    available_records = {int(record.event_id) for record in records}
    available_targets = {target.event_id for target in candidates_targets}
    target_by_id = {target.event_id: target for target in candidates_targets}
    while available_records and available_targets:
        eligible = [
            candidate
            for candidate in candidates
            if candidate[1] in available_records
            and candidate[2] in available_targets
        ]
        best_score = max(candidate[0] for candidate in eligible)
        tied = [candidate for candidate in eligible if candidate[0] == best_score]
        if len(tied) > 1:
            state.counters["record_target_exact_tie_count"] += 1
        _, event_id, target_id = min(
            tied, key=lambda candidate: (candidate[1], candidate[2])
        )
        target = target_by_id[target_id]
        key = (video_name, event_id)
        birth_embedding = state.birth_embeddings.get(key)
        if birth_embedding is None:
            raise RuntimeError("diagnostic record lost its creation embedding")
        state.links[key] = DiagnosticOwnerLink(
            target_event_id=target.event_id,
            target_class_id=target.class_id,
            target_start_frame=target.start_frame,
            target_end_frame=target.end_frame,
            association_frame=float(frame),
            birth_embedding=birth_embedding.detach().clone(),
            late=not target.is_birth,
        )
        history_key = (video_name, target.event_id)
        state.target_link_history[history_key] += 1
        state.counters["association_count"] += 1
        state.counters["late_association_count"] += int(not target.is_birth)
        available_records.remove(event_id)
        available_targets.remove(target_id)


@dataclass
class PreparedOwnerBatch:
    owners: torch.Tensor
    owner_padding: torch.Tensor
    owner_ids: torch.Tensor
    decoder_input: torch.Tensor
    state_logits: Optional[torch.Tensor] = None
    end_offsets: Optional[torch.Tensor] = None
    class_logits: Optional[torch.Tensor] = None
    updated_embeddings: Optional[torch.Tensor] = None
    attention: Optional[torch.Tensor] = None


def _prepare_owner_batch(
    state: RouteState,
    *,
    video_name: str,
    frame: float,
    targets: list[TargetView],
    target_query_map: dict[int, int],
    event_query_features: torch.Tensor,
    class_logits: torch.Tensor,
) -> PreparedOwnerBatch:
    memory = state.memory
    if memory.ownership_mode == "fresh_rematch":
        memory.rematch_active_owners(
            video_name,
            event_query_features,
            class_logits,
        )
    _map_unlinked_records(
        state,
        video_name=video_name,
        frame=frame,
        targets=targets,
        target_query_map=target_query_map,
        query_features=event_query_features,
    )
    owners, owner_padding, owner_ids = memory.owner_batch(
        [video_name],
        device=event_query_features.device,
        dtype=event_query_features.dtype,
        embedding_dim=int(event_query_features.size(-1)),
    )
    decoder_input = owners.clone()
    for owner_index in range(int(owners.size(1))):
        event_id = int(owner_ids[0, owner_index].item())
        link = state.links.get((video_name, event_id))
        if (
            CHANNELS[state.channel][1] == "refreshed"
            and link is not None
            and link.target_event_id in target_query_map
        ):
            matched_query = int(target_query_map[link.target_event_id])
            decoder_input[0, owner_index] = event_query_features[
                matched_query
            ].clone()
            state.counters["owner_refresh_count"] += 1
    return PreparedOwnerBatch(
        owners=owners,
        owner_padding=owner_padding,
        owner_ids=owner_ids,
        decoder_input=decoder_input,
    )


def _decode_prepared_owner_batches(
    prepared: dict[tuple[str, str], PreparedOwnerBatch],
    *,
    owner_decoder,
    event_query_features: torch.Tensor,
) -> None:
    """Decode PF at its exact B=1 shape and batch the seven interventions."""

    ordered_keys = [
        (channel, route) for channel in CHANNELS for route in ROUTES
    ]
    if set(prepared) != set(ordered_keys):
        raise RuntimeError("D1.5 owner batch omitted a diagnostic route")

    def decode_keys(keys: list[tuple[str, str]]) -> None:
        max_width = max(
            int(prepared[key].owners.size(1)) for key in keys
        )
        if max_width == 0:
            return
        route_count = len(keys)
        embedding_dim = int(event_query_features.size(-1))
        combined_input = event_query_features.new_zeros(
            (route_count, max_width, embedding_dim)
        )
        combined_padding = torch.ones(
            (route_count, max_width),
            device=event_query_features.device,
            dtype=torch.bool,
        )
        for route_index, key in enumerate(keys):
            row = prepared[key]
            width = int(row.decoder_input.size(1))
            if width:
                combined_input[route_index, :width] = row.decoder_input[0]
                combined_padding[route_index, :width] = row.owner_padding[0]
        combined_queries = event_query_features.unsqueeze(0).expand(
            route_count, -1, -1
        )
        (
            state_logits,
            end_offsets,
            class_logits,
            updated_embeddings,
            attention,
        ) = owner_decoder(
            combined_input,
            combined_queries,
            combined_padding,
            return_attention=True,
        )
        for route_index, key in enumerate(keys):
            row = prepared[key]
            width = int(row.decoder_input.size(1))
            if not width:
                continue
            row.state_logits = state_logits[
                route_index : route_index + 1, :width
            ]
            row.end_offsets = end_offsets[
                route_index : route_index + 1, :width
            ]
            row.class_logits = class_logits[
                route_index : route_index + 1, :width
            ]
            row.updated_embeddings = updated_embeddings[
                route_index : route_index + 1, :width
            ]
            row.attention = attention[
                route_index : route_index + 1, :width
            ]

    # PF formal is the exact D1.4 control.  Preserving its one-route batch shape
    # avoids introducing a different GEMM reduction shape near owner-state ties.
    decode_keys([("PF", "formal")])
    decode_keys([key for key in ordered_keys if key != ("PF", "formal")])
    for channel in CHANNELS:
        formal = prepared[(channel, "formal")]
        shadow = prepared[(channel, "shadow")]
        same_preintervention_state = (
            formal.decoder_input.shape == shadow.decoder_input.shape
            and torch.equal(formal.decoder_input, shadow.decoder_input)
            and torch.equal(formal.owner_padding, shadow.owner_padding)
            and torch.equal(formal.owner_ids, shadow.owner_ids)
        )
        if not same_preintervention_state or formal.state_logits is None:
            continue
        # A shadow is definitionally a clone until the first formal CANCEL.
        # Reuse the exact formal decode, rather than accepting harmless but
        # protocol-visible GEMM batch-shape rounding differences.
        shadow.state_logits = formal.state_logits
        shadow.end_offsets = formal.end_offsets
        shadow.class_logits = formal.class_logits
        shadow.updated_embeddings = formal.updated_embeddings
        shadow.attention = formal.attention


def _oracle_birth_specs(
    state: RouteState,
    *,
    video_name: str,
    birth_targets: list[TargetView],
    birth_query_map: dict[int, int],
    candidate_start_frames: torch.Tensor,
) -> tuple[list[dict], dict[int, TargetView]]:
    if CHANNELS[state.channel][0] != "oracle_visible":
        return [], {}
    specs = []
    pending = {}
    for target in birth_targets:
        target_key = (video_name, target.event_id)
        if target_key in state.oracle_admitted_targets:
            raise RuntimeError("oracle-visible channel attempted duplicate admission")
        query_index = int(birth_query_map[target.event_id])
        specs.append(
            {
                "query_index": query_index,
                "target_event_id": None,
                "start_frame": float(candidate_start_frames[query_index].item()),
                "source": "diagnostic_oracle_visible",
                "association_status": "diagnostic_sidecar_only",
                "force_create": True,
                "merge_predicted": False,
            }
        )
        pending[query_index] = target
        state.oracle_admitted_targets.add(target_key)
    return specs, pending


def _register_created_records(
    state: RouteState,
    *,
    video_name: str,
    frame: float,
    pending_oracle: dict[int, TargetView],
) -> None:
    lookup = _record_lookup(state.memory, video_name)
    for event in state.memory.last_audit.get("lifecycle_events", ()):
        if event.get("transition") != "birth":
            continue
        event_id = int(event["event_id"])
        key = (video_name, event_id)
        record = lookup.get(event_id)
        if record is None:
            raise RuntimeError("birth audit has no live diagnostic record")
        if record.target_event_id is not None:
            raise RuntimeError("diagnostic birth wrote GT into EventRecord")
        if key in state.created_records:
            # Exact-ID reacquisition is forbidden because all runtime target IDs
            # are None; seeing the same event twice is therefore an error.
            raise RuntimeError("diagnostic runtime reused a formal event id")
        state.created_records.add(key)
        state.birth_embeddings[key] = record.owner_embedding.detach().clone()
        state.created_frames[key] = float(record.created_frame)
        state.created_by_source[record.source] += 1
        if pending_oracle:
            target = pending_oracle.get(int(record.owner_query_id))
            if target is None:
                raise RuntimeError("oracle-visible birth/query sidecar did not close")
            state.links[key] = DiagnosticOwnerLink(
                target_event_id=target.event_id,
                target_class_id=target.class_id,
                target_start_frame=target.start_frame,
                target_end_frame=target.end_frame,
                association_frame=float(frame),
                birth_embedding=record.owner_embedding.detach().clone(),
                late=False,
            )
            state.target_link_history[(video_name, target.event_id)] += 1
            state.counters["association_count"] += 1


def _state_margins(logits: torch.Tensor) -> list[float]:
    if logits.shape != (3,):
        raise RuntimeError("D1.5 owner state logits must have three states")
    margins = []
    for index in range(3):
        competitors = torch.cat((logits[:index], logits[index + 1 :]))
        margins.append(float((logits[index] - competitors.max()).item()))
    return margins


def _trace_transition(
    *,
    audit_events: dict[int, list[str]],
    event_id: int,
    state_winner: int,
    end_suppressed: bool,
) -> str:
    transitions = audit_events.get(event_id, [])
    if transitions:
        return "+".join(transitions)
    if end_suppressed:
        return "end_suppressed_min_duration"
    if state_winner == 1:
        return "continue"
    return "no_transition"


def _process_route_prefix(
    state: RouteState,
    *,
    video_name: str,
    frame: float,
    is_real: bool,
    is_eos: bool,
    targets: list[TargetView],
    target_query_map: dict[int, int],
    birth_targets: list[TargetView],
    birth_query_map: dict[int, int],
    outputs: dict,
    row_index: int,
    query_batch_sha256: str,
    prepared_owner: Optional[PreparedOwnerBatch],
    trace_handle,
) -> None:
    memory = state.memory
    state.query_consumption_digest.update(
        (
            f"{query_batch_sha256}|{row_index}|{video_name}|{frame:.17g}|"
            f"{int(is_real)}|{int(is_eos)}\n"
        ).encode("utf-8")
    )
    state.counters["query_prefix_consumption_count"] += 1
    row = slice(row_index, row_index + 1)
    event_query_features = outputs["event_query_features"][row]
    class_logits = outputs["pred_cls"][row]
    birth_logits = outputs["event_birth_logits"][row]
    candidate_starts = outputs["event_candidate_start_frames"][row]

    if not is_real:
        before = tuple(
            (
                record.event_id,
                record.status,
                float(record.start_frame),
                record.owner_embedding.detach().cpu().numpy().tobytes(),
            )
            for record in memory.records(video_name)
        )
        runtime = memory.step(
            video_names=[video_name],
            current_frames=[frame],
            birth_logits=birth_logits,
            alive_logits=outputs["event_alive_logits"][row],
            end_logits=outputs["event_end_logits"][row],
            end_offsets=outputs["event_end_offsets"][row],
            class_logits=class_logits,
            query_features=event_query_features,
            candidate_start_frames=candidate_starts,
            candidate_state_logits=outputs["event_state_logits"][row],
            is_real_prefix=[False],
            is_eos=[False],
            diagnostic_suppress_predicted_births=(
                CHANNELS[state.channel][0] == "oracle_visible"
            ),
            diagnostic_cancel_as_continue=(state.route == "shadow"),
        )
        after = tuple(
            (
                record.event_id,
                record.status,
                float(record.start_frame),
                record.owner_embedding.detach().cpu().numpy().tobytes(),
            )
            for record in memory.records(video_name)
        )
        if before != after or int(runtime["padding_prefixes_ignored"][0]) != 1:
            raise RuntimeError(f"{state.channel}/{state.route} padding mutated state")
        for key, value in runtime.items():
            if key == "padding_prefixes_ignored":
                continue
            if bool(value.any().item()):
                raise RuntimeError(
                    f"{state.channel}/{state.route} padding produced {key}"
                )
        state.counters["padding_prefix_count"] += 1
        state.counters["padding_noop_count"] += 1
        return

    state.counters["real_prefix_count"] += 1
    state.counters["observed_eos_count"] += int(is_eos)
    if prepared_owner is None:
        raise RuntimeError("real D1.5 prefix lacks a prepared owner batch")
    target_by_id = {target.event_id: target for target in targets}
    owners = prepared_owner.owners
    owner_padding = prepared_owner.owner_padding
    owner_ids = prepared_owner.owner_ids
    owner_runtime = {}
    trace_rows = []
    if owners.size(1):
        decoded = (
            prepared_owner.state_logits,
            prepared_owner.end_offsets,
            prepared_owner.class_logits,
            prepared_owner.updated_embeddings,
            prepared_owner.attention,
        )
        if any(value is None for value in decoded):
            raise RuntimeError("prepared D1.5 owner batch was not decoded")
        (
            owner_state_logits,
            owner_end_offsets,
            owner_class_logits,
            owner_updated_embeddings,
            owner_attention,
        ) = decoded
        owner_runtime = {
            "owner_state_logits": owner_state_logits,
            "owner_end_offsets": owner_end_offsets,
            "owner_class_logits": owner_class_logits,
            "owner_updated_embeddings": owner_updated_embeddings,
            "owner_valid_mask": ~owner_padding,
            "owner_record_ids": owner_ids,
        }
        records = _record_lookup(memory, video_name)
        for owner_index in range(int(owners.size(1))):
            if bool(owner_padding[0, owner_index].item()):
                continue
            event_id = int(owner_ids[0, owner_index].item())
            record = records[event_id]
            key = (video_name, event_id)
            link = state.links.get(key)
            matched_query = (
                None
                if link is None
                or link.target_event_id not in target_query_map
                or CHANNELS[state.channel][1] != "refreshed"
                else int(target_query_map[link.target_event_id])
            )
            logits = owner_state_logits[0, owner_index]
            margins = _state_margins(logits)
            winner = int(logits.argmax().item())
            first = key not in state.first_decisions
            if first:
                state.first_decisions.add(key)
                state.first_state_counts[str(winner)] += 1
            state.winner_counts[str(winner)] += 1
            state.stats["end_margin"].add(margins[2])
            attention = owner_attention[0, owner_index]
            attention_probs = attention.clamp_min(1e-12)
            entropy = float(
                (-(attention_probs * attention_probs.log()).sum()).item()
            )
            state.stats["owner_attention_entropy"].add(entropy)
            top_query = int(attention.argmax().item())
            oracle_rank = None
            owner_to_oracle = None
            if link is not None and link.target_event_id in target_query_map:
                oracle_query = int(target_query_map[link.target_event_id])
                oracle_weight = float(attention[oracle_query].item())
                oracle_rank = 1 + int((attention > oracle_weight).sum().item())
                owner_to_oracle = float(
                    F.cosine_similarity(
                        record.owner_embedding.to(event_query_features).reshape(1, -1),
                        event_query_features[0, oracle_query].reshape(1, -1),
                    )[0].item()
                )
                state.stats["owner_to_oracle_query_cosine"].add(owner_to_oracle)
            owner_to_birth = None
            if key in state.birth_embeddings:
                owner_to_birth = float(
                    F.cosine_similarity(
                        record.owner_embedding.reshape(1, -1),
                        state.birth_embeddings[key]
                        .to(record.owner_embedding)
                        .reshape(1, -1),
                    )[0].item()
                )
                state.stats["owner_to_birth_cosine"].add(owner_to_birth)
            end_suppressed = (
                winner == 2
                and frame < record.start_frame + memory.min_duration_frames
            )
            state.counters["end_argmax_count"] += int(winner == 2)
            state.counters["end_min_duration_suppression_count"] += int(
                end_suppressed
            )
            target = (
                None
                if link is None
                else target_by_id.get(link.target_event_id)
            )
            trace_rows.append(
                {
                    "protocol": PROTOCOL,
                    "channel": state.channel,
                    "route": state.route,
                    "video_name": video_name,
                    "runtime_event_id": event_id,
                    "diagnostic_target_event_id": (
                        None if link is None else link.target_event_id
                    ),
                    "source": record.source,
                    "creation_frame": float(record.created_frame),
                    "current_frame": float(frame),
                    "first_owner_decision": bool(first),
                    "owner_query_before_refresh": int(record.owner_query_id),
                    "owner_query_after_refresh": (
                        int(record.owner_query_id)
                        if matched_query is None
                        else int(matched_query)
                    ),
                    "formal_owner_query_id_after_intervention": int(
                        record.owner_query_id
                    ),
                    "formal_owner_query_id_unchanged": True,
                    "matched_current_query": matched_query,
                    "owner_to_birth_cosine": owner_to_birth,
                    "owner_to_oracle_query_cosine": owner_to_oracle,
                    "owner_attention_top_query": top_query,
                    "oracle_query_attention_rank": oracle_rank,
                    "owner_attention_entropy": entropy,
                    "cancel_logit": float(logits[0].item()),
                    "continue_logit": float(logits[1].item()),
                    "end_logit": float(logits[2].item()),
                    "cancel_margin": margins[0],
                    "continue_margin": margins[1],
                    "end_margin": margins[2],
                    "state_argmax": winner,
                    "end_suppressed_by_minimum_duration": bool(end_suppressed),
                    "formal_transition": None,
                    "shadow_transition": None,
                    "target_end_observable_now": bool(
                        target is not None and target.is_end
                    ),
                    "frames_from_annotated_end": (
                        None
                        if target is None
                        else float(frame - target.end_frame)
                    ),
                    "formal_lifetime": (
                        float(frame - record.created_frame)
                        if state.route == "formal"
                        else None
                    ),
                    "shadow_lifetime": (
                        float(frame - record.created_frame)
                        if state.route == "shadow"
                        else None
                    ),
                }
            )

    oracle_specs, pending_oracle = _oracle_birth_specs(
        state,
        video_name=video_name,
        birth_targets=birth_targets,
        birth_query_map=birth_query_map,
        candidate_start_frames=candidate_starts[0],
    )
    runtime = memory.step(
        video_names=[video_name],
        current_frames=[frame],
        birth_logits=birth_logits,
        alive_logits=outputs["event_alive_logits"][row],
        end_logits=outputs["event_end_logits"][row],
        end_offsets=outputs["event_end_offsets"][row],
        class_logits=class_logits,
        query_features=event_query_features,
        candidate_start_frames=candidate_starts,
        candidate_state_logits=outputs["event_state_logits"][row],
        is_real_prefix=[True],
        is_eos=[is_eos],
        oracle_births=[oracle_specs],
        diagnostic_suppress_predicted_births=(
            CHANNELS[state.channel][0] == "oracle_visible"
        ),
        diagnostic_cancel_as_continue=(state.route == "shadow"),
        **owner_runtime,
    )
    for output_name, counter_name in (
        ("birth_count", "birth_count"),
        ("cancellation_count", "cancellation_count"),
        ("end_count", "end_count"),
        ("emit_count", "emit_count"),
        ("reacquisition_count", "reacquisition_count"),
        ("runtime_capacity_exhaustions", "capacity_exhaustion_count"),
    ):
        state.counters[counter_name] += int(runtime[output_name][0].item())

    _register_created_records(
        state,
        video_name=video_name,
        frame=frame,
        pending_oracle=pending_oracle,
    )
    if CHANNELS[state.channel][1] == "refreshed":
        # A predicted record is born after the owner decode.  Causal sidecar
        # attachment here makes it refreshable only at its next owner decision.
        _map_unlinked_records(
            state,
            video_name=video_name,
            frame=frame,
            targets=targets,
            target_query_map=target_query_map,
            query_features=event_query_features[0],
        )

    audit_events: dict[int, list[str]] = {}
    for event in memory.last_audit.get("lifecycle_events", ()):
        transition = str(event.get("transition"))
        event_id = int(event["event_id"])
        audit_events.setdefault(event_id, []).append(transition)
        state.transition_counts[transition] += 1
        if transition in {"cancel", "end", "emit"}:
            created_frame = state.created_frames.get((video_name, event_id))
            if created_frame is None:
                raise RuntimeError("transition lacked its recorded creation frame")
            state.stats[f"lifetime_at_{transition}"].add(frame - created_frame)

    for trace_row in trace_rows:
        transition = _trace_transition(
            audit_events=audit_events,
            event_id=int(trace_row["runtime_event_id"]),
            state_winner=int(trace_row["state_argmax"]),
            end_suppressed=bool(
                trace_row["end_suppressed_by_minimum_duration"]
            ),
        )
        field = (
            "formal_transition"
            if state.route == "formal"
            else "shadow_transition"
        )
        trace_row[field] = transition
        if "end" in audit_events.get(int(trace_row["runtime_event_id"]), ()):
            if trace_row["diagnostic_target_event_id"] is not None:
                state.counters["target_backed_end_transition_count"] += 1
                link = state.links[
                    (video_name, int(trace_row["runtime_event_id"]))
                ]
                if abs(float(frame) - link.target_end_frame) <= float(
                    memory.segment_size
                ):
                    state.counters[
                        "target_backed_end_within_one_segment_count"
                    ] += 1
                if bool(trace_row["target_end_observable_now"]):
                    state.counters[
                        "end_at_observed_target_end_count"
                    ] += 1
        trace_handle.write(
            json.dumps(trace_row, sort_keys=True, separators=(",", ":")) + "\n"
        )
        state.counters["decision_row_count"] += 1

    snapshot = _validate_ledger_snapshot(
        video_name,
        memory.ledger(video_name),
        state.ledger_snapshots.get(video_name, ()),
    )
    state.ledger_snapshots[video_name] = snapshot
    _assert_no_gt_runtime_ids(state)
    if is_eos:
        active_after_eos = len(memory.records(video_name))
        state.counters["records_active_after_eos"] += active_after_eos
        state.counters["videos_with_active_records_after_eos"] += int(
            active_after_eos > 0
        )


@dataclass
class PositiveControlState:
    memory: DynamicEventMemory
    links: dict[tuple[str, int], DiagnosticOwnerLink] = field(default_factory=dict)
    admitted_targets: set[tuple[str, int]] = field(default_factory=set)
    ledger_snapshots: dict[str, tuple[dict, ...]] = field(default_factory=dict)
    counters: Counter = field(default_factory=Counter)


def _process_positive_control_prefix(
    control: PositiveControlState,
    *,
    video_name: str,
    frame: float,
    is_real: bool,
    is_eos: bool,
    targets: list[TargetView],
    birth_targets: list[TargetView],
    birth_query_map: dict[int, int],
    outputs: dict,
    row_index: int,
) -> None:
    memory = control.memory
    row = slice(row_index, row_index + 1)
    query_features = outputs["event_query_features"][row]
    class_logits = outputs["pred_cls"][row]
    birth_logits = outputs["event_birth_logits"][row]
    candidate_starts = outputs["event_candidate_start_frames"][row]

    if not is_real:
        before = tuple(
            (record.event_id, record.status, float(record.start_frame))
            for record in memory.records(video_name)
        )
        runtime = memory.step(
            video_names=[video_name],
            current_frames=[frame],
            birth_logits=birth_logits,
            alive_logits=outputs["event_alive_logits"][row],
            end_logits=outputs["event_end_logits"][row],
            end_offsets=outputs["event_end_offsets"][row],
            class_logits=class_logits,
            query_features=query_features,
            candidate_start_frames=candidate_starts,
            candidate_state_logits=outputs["event_state_logits"][row],
            is_real_prefix=[False],
            is_eos=[False],
            diagnostic_suppress_predicted_births=True,
        )
        after = tuple(
            (record.event_id, record.status, float(record.start_frame))
            for record in memory.records(video_name)
        )
        if before != after or int(runtime["padding_prefixes_ignored"][0]) != 1:
            raise RuntimeError("positive control padding mutated state")
        control.counters["padding_prefix_count"] += 1
        control.counters["padding_noop_count"] += 1
        return

    control.counters["real_prefix_count"] += 1
    control.counters["observed_eos_count"] += int(is_eos)
    target_by_id = {target.event_id: target for target in targets}
    owners, owner_padding, owner_ids = memory.owner_batch(
        [video_name],
        device=query_features.device,
        dtype=query_features.dtype,
        embedding_dim=int(query_features.size(-1)),
    )
    owner_runtime = {}
    if owners.size(1):
        state_logits = owners.new_full((1, owners.size(1), 3), -20.0)
        state_logits[..., 1] = 20.0
        for owner_index in range(int(owners.size(1))):
            event_id = int(owner_ids[0, owner_index].item())
            link = control.links.get((video_name, event_id))
            if link is None:
                raise RuntimeError("positive control active record has no GT sidecar")
            target = target_by_id.get(link.target_event_id)
            if target is None:
                raise RuntimeError(
                    "positive control target disappeared before its end crossing"
                )
            if target.is_end:
                state_logits[0, owner_index] = torch.tensor(
                    [-20.0, -20.0, 20.0],
                    device=state_logits.device,
                    dtype=state_logits.dtype,
                )
        owner_runtime = {
            "owner_state_logits": state_logits,
            "owner_end_offsets": owners.new_zeros((1, owners.size(1))),
            "owner_updated_embeddings": owners,
            "owner_valid_mask": ~owner_padding,
            "owner_record_ids": owner_ids,
        }

    oracle_specs = []
    pending = {}
    for target in birth_targets:
        target_key = (video_name, target.event_id)
        if target_key in control.admitted_targets:
            raise RuntimeError("positive control duplicated a target admission")
        query_index = int(birth_query_map[target.event_id])
        oracle_specs.append(
            {
                "query_index": query_index,
                "target_event_id": None,
                "start_frame": float(candidate_starts[0, query_index].item()),
                "source": "diagnostic_positive_control",
                "association_status": "diagnostic_sidecar_only",
                "force_create": True,
                "merge_predicted": False,
            }
        )
        pending[query_index] = target
        control.admitted_targets.add(target_key)

    runtime = memory.step(
        video_names=[video_name],
        current_frames=[frame],
        birth_logits=birth_logits,
        alive_logits=outputs["event_alive_logits"][row],
        end_logits=outputs["event_end_logits"][row],
        end_offsets=outputs["event_end_offsets"][row],
        class_logits=class_logits,
        query_features=query_features,
        candidate_start_frames=candidate_starts,
        candidate_state_logits=outputs["event_state_logits"][row],
        is_real_prefix=[True],
        is_eos=[is_eos],
        oracle_births=[oracle_specs],
        diagnostic_suppress_predicted_births=True,
        **owner_runtime,
    )
    for output_name, counter_name in (
        ("birth_count", "birth_count"),
        ("cancellation_count", "cancellation_count"),
        ("end_count", "end_count"),
        ("emit_count", "emit_count"),
        ("reacquisition_count", "reacquisition_count"),
        ("runtime_capacity_exhaustions", "capacity_exhaustion_count"),
    ):
        control.counters[counter_name] += int(runtime[output_name][0].item())

    lookup = _record_lookup(memory, video_name)
    for event in memory.last_audit.get("lifecycle_events", ()):
        if event.get("transition") != "birth":
            continue
        event_id = int(event["event_id"])
        record = lookup.get(event_id)
        if record is None or record.target_event_id is not None:
            raise RuntimeError("positive control birth runtime/sidecar isolation failed")
        target = pending.get(int(record.owner_query_id))
        if target is None:
            raise RuntimeError("positive control birth/query sidecar did not close")
        control.links[(video_name, event_id)] = DiagnosticOwnerLink(
            target_event_id=target.event_id,
            target_class_id=target.class_id,
            target_start_frame=target.start_frame,
            target_end_frame=target.end_frame,
            association_frame=float(frame),
            birth_embedding=record.owner_embedding.detach().clone(),
            late=False,
        )
    snapshot = _validate_ledger_snapshot(
        video_name,
        memory.ledger(video_name),
        control.ledger_snapshots.get(video_name, ()),
    )
    control.ledger_snapshots[video_name] = snapshot
    _assert_memory_has_no_gt_runtime_ids(
        memory, label="positive lifecycle control"
    )
    if is_eos:
        active_after_eos = len(memory.records(video_name))
        control.counters["records_active_after_eos"] += active_after_eos
        control.counters["videos_with_active_records_after_eos"] += int(
            active_after_eos > 0
        )


def _hash_query_outputs(
    digest,
    outputs: dict,
    *,
    batch_index: int,
) -> str:
    batch_digest = hashlib.sha256()

    def update(payload: bytes) -> None:
        digest.update(payload)
        batch_digest.update(payload)

    update(f"batch:{batch_index}\n".encode("utf-8"))
    for name in QUERY_HASH_FIELDS:
        tensor = outputs.get(name)
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"query stream omitted tensor {name}")
        array = tensor.detach().cpu().contiguous().numpy()
        update(name.encode("utf-8"))
        update(str(tuple(array.shape)).encode("ascii"))
        update(str(array.dtype).encode("ascii"))
        update(array.tobytes(order="C"))
    return batch_digest.hexdigest()


def _dataset_paths(options: dict) -> dict[str, Path]:
    return {
        "annotation": Path(options["video_anno"]).expanduser().resolve(),
        "train_features": Path(
            options["video_feature_all_train"]
        ).expanduser().resolve(),
        "video_len": Path(
            options["video_len_file"].format("train")
        ).expanduser().resolve(),
        "proposal_labels": Path(
            options["ontal_label_file"].format(
                "train",
                options["num_frame"],
                options["num_queries"],
                options["detect_len"],
                options["anti_len"],
                options["max_memory_len"],
                options["p_videos"],
            )
        )
        .expanduser()
        .resolve(),
    }


def _artifact_receipt(paths: dict[str, Path]) -> dict:
    receipt = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"official train artifact is absent: {name}={path}")
        row = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256(path),
        }
        expected = OFFICIAL_TRAIN_ARTIFACTS[name]
        if row["bytes"] != expected["bytes"] or row["sha256"] != expected["sha256"]:
            raise RuntimeError(f"official train artifact identity drifted: {name}")
        receipt[name] = row
    return receipt


def _route_summary(state: RouteState, *, query_hash: str) -> dict:
    ledger_count = sum(len(rows) for rows in state.ledger_snapshots.values())
    if ledger_count != state.counters["emit_count"]:
        raise RuntimeError(
            f"{state.channel}/{state.route} ledger/emission count did not close"
        )
    duplicates = sum(
        max(0, count - 1) for count in state.target_link_history.values()
    )
    linked_targets = len(state.target_link_history)
    required_counts = {
        name: int(state.counters.get(name, 0))
        for name in (
            "real_prefix_count",
            "padding_prefix_count",
            "padding_noop_count",
            "observed_eos_count",
            "birth_count",
            "cancellation_count",
            "end_count",
            "emit_count",
            "reacquisition_count",
            "capacity_exhaustion_count",
            "records_active_after_eos",
            "videos_with_active_records_after_eos",
            "association_count",
            "late_association_count",
            "owner_refresh_count",
            "record_target_exact_tie_count",
            "query_prefix_consumption_count",
            "decision_row_count",
            "end_argmax_count",
            "end_min_duration_suppression_count",
            "target_backed_end_transition_count",
            "target_backed_end_within_one_segment_count",
            "end_at_observed_target_end_count",
        )
    }
    required_counts["end_without_emission_count"] = max(
        0, required_counts["end_count"] - required_counts["emit_count"]
    )
    return {
        "channel": state.channel,
        "route": state.route,
        "admission": CHANNELS[state.channel][0],
        "identity": CHANNELS[state.channel][1],
        "query_stream_sha256": query_hash,
        "query_consumption_sha256": state.query_consumption_digest.hexdigest(),
        "counts": {
            **required_counts,
            **dict(sorted(state.counters.items())),
        },
        "unique_runtime_record_count": len(state.created_records),
        "created_records_by_source": dict(sorted(state.created_by_source.items())),
        "linked_unique_target_count": linked_targets,
        "raw_semantic_duplicate_count": duplicates,
        "batch_local_owner_decision_fragment_count": state.counters[
            "decision_row_count"
        ],
        "first_owner_decision_state_counts": dict(
            sorted(state.first_state_counts.items())
        ),
        "owner_state_winner_counts": dict(sorted(state.winner_counts.items())),
        "transition_counts": dict(sorted(state.transition_counts.items())),
        "statistics": {
            name: stats.summary() for name, stats in sorted(state.stats.items())
        },
        "lifecycle_integrity": {
            "immutable_ledger_verified": True,
            "positive_length_verified": True,
            "nonnegative_start_verified": True,
            "no_duplicate_event_verified": True,
            "contiguous_sequence_id_verified": True,
            "ledger_emit_count_closed": True,
            "ledger_row_count": ledger_count,
            "video_ledger_count": len(state.ledger_snapshots),
            "ground_truth_stored_in_runtime_record": False,
        },
    }


def _positive_control_summary(control: PositiveControlState) -> dict:
    ledger_count = sum(len(rows) for rows in control.ledger_snapshots.values())
    if ledger_count != control.counters["emit_count"]:
        raise RuntimeError("positive control ledger/emission count did not close")
    required_counts = {
        name: int(control.counters.get(name, 0))
        for name in (
            "real_prefix_count",
            "padding_prefix_count",
            "padding_noop_count",
            "observed_eos_count",
            "birth_count",
            "cancellation_count",
            "end_count",
            "emit_count",
            "reacquisition_count",
            "capacity_exhaustion_count",
            "records_active_after_eos",
            "videos_with_active_records_after_eos",
        )
    }
    return {
        "counts": {
            **required_counts,
            **dict(sorted(control.counters.items())),
        },
        "admitted_unique_target_count": len(control.admitted_targets),
        "sidecar_link_count": len(control.links),
        "lifecycle_integrity": {
            "immutable_ledger_verified": True,
            "positive_length_verified": True,
            "nonnegative_start_verified": True,
            "no_duplicate_event_verified": True,
            "contiguous_sequence_id_verified": True,
            "ledger_emit_count_closed": True,
            "ledger_row_count": ledger_count,
            "video_ledger_count": len(control.ledger_snapshots),
            "ground_truth_stored_in_runtime_record": False,
        },
    }


def main() -> None:
    cli = _parse_args()
    if cli.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("formal D1.5 counterfactual requires one CUDA device")
    if torch.cuda.device_count() != 1:
        raise SystemExit("formal D1.5 counterfactual requires one visible GPU")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise SystemExit(f"D1.5 diagnostic source is dirty:\n{status}")
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    if source_commit != cli.expected_diagnostic_source_commit:
        raise RuntimeError("D1.5 diagnostic source commit mismatch")
    if source_tree != cli.expected_diagnostic_source_tree:
        raise RuntimeError("D1.5 diagnostic source tree mismatch")
    manifest_sha256 = _sha256(cli.manifest)
    if manifest_sha256 != cli.expected_manifest_sha256:
        raise RuntimeError("D1.5 manifest SHA-256 mismatch")
    manifest = _load_json(cli.manifest, "D1.5 manifest")
    _validate_registered_source_identities(
        manifest,
        training_source_commit=cli.expected_training_source_commit,
        training_source_tree=cli.expected_training_source_tree,
        d14_source_commit=cli.expected_d14_source_commit,
        d14_source_tree=cli.expected_d14_source_tree,
    )

    options = _load_json(cli.options, "options")
    training_identity = _load_json(
        cli.training_source_identity, "training source identity"
    )
    d14_gate = _load_json(cli.d14_structure_gate, "D1.4 structure gate")
    checkpoint_before = {
        "path": str(cli.checkpoint),
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    options_before = {
        "path": str(cli.options),
        "bytes": cli.options.stat().st_size,
        "mtime_ns": cli.options.stat().st_mtime_ns,
        "sha256": _sha256(cli.options),
    }
    if checkpoint_before["sha256"] != cli.expected_checkpoint_sha256:
        raise RuntimeError("D1.5 checkpoint SHA-256 mismatch")
    if options_before["sha256"] != cli.expected_options_sha256:
        raise RuntimeError("D1.5 options SHA-256 mismatch")
    d14_binding = _validate_d14_gate(
        d14_gate,
        checkpoint=cli.checkpoint,
        options=cli.options,
        checkpoint_sha256=checkpoint_before["sha256"],
        options_sha256=options_before["sha256"],
        d14_source_commit=cli.expected_d14_source_commit,
        d14_source_tree=cli.expected_d14_source_tree,
    )
    checkpoint = torch.load(cli.checkpoint, map_location="cpu")
    _validate_inputs(
        options,
        checkpoint,
        training_identity,
        expected_training_source_commit=cli.expected_training_source_commit,
        expected_training_source_tree=cli.expected_training_source_tree,
        one_epoch_mechanism_gate_status="PASS_TRAIN_MECHANISM_ONLY",
        d13_variant="combined",
        d14_variant="decision_aligned_bag",
    )

    random.seed(52)
    np.random.seed(52)
    torch.manual_seed(52)
    torch.cuda.manual_seed_all(52)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    options["training"] = False
    options["mode"] = "eval"
    options["make_output"] = False
    options["load_model"] = False
    options["num_workers"] = int(options.get("num_workers", 4))
    args = SimpleNamespace(**options)
    device = torch.device("cuda")
    artifact_paths = _dataset_paths(options)
    artifacts_before = _artifact_receipt(artifact_paths)
    dataset = THUMOS14Dataset(args, subset="train")
    if cli.max_batches == 0 and len(dataset.video_list) != 200:
        raise RuntimeError("complete D1.5 scan requires the official 200 videos")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = torch.nn.DataParallel(build_model(args)).to(device)
    validate_d1_checkpoint_compatibility(checkpoint, model, args)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval()
    memory_initialize(model, args)
    model.module.set_event_diagnostic_query_only(True)

    routes = {
        (channel, route): RouteState(
            channel=channel,
            route=route,
            memory=_new_memory(model.module.event_memory),
        )
        for channel in CHANNELS
        for route in ROUTES
    }
    positive_control = PositiveControlState(
        memory=_new_memory(model.module.event_memory)
    )
    global_counts = Counter()
    query_digest = hashlib.sha256()
    processed_batches = 0
    last_physical_frame: dict[str, float] = {}
    last_real_frame: dict[str, float] = {}
    closed_streams: set[str] = set()
    started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with gzip.open(
        cli.trace_output,
        "wt",
        encoding="utf-8",
        compresslevel=1,
        newline="\n",
    ) as trace_handle:
        with torch.no_grad():
            for batch_index, (features, targets, infos) in enumerate(loader):
                if cli.max_batches and batch_index >= cli.max_batches:
                    break
                processed_batches += 1
                features, targets, infos = parrallel_collate_fn(
                    features, targets, infos, args.p_videos
                )
                batch_size = int(features.size(0))
                if batch_size != int(args.batch):
                    raise RuntimeError(
                        f"frozen MATR physical batch drifted: {batch_size} != {args.batch}"
                    )
                names = [
                    str(value)
                    for value in _metadata_values(
                        infos["video_name"],
                        batch_size=batch_size,
                        label="video_name",
                    )
                ]
                frames = [
                    float(value)
                    for value in _metadata_values(
                        infos["current_frame"],
                        batch_size=batch_size,
                        label="current_frame",
                    )
                ]
                real_flags = [
                    bool(value)
                    for value in _metadata_values(
                        infos["is_real_prefix"],
                        batch_size=batch_size,
                        label="is_real_prefix",
                    )
                ]
                eos_flags = [
                    bool(value)
                    for value in _metadata_values(
                        infos["is_eos"],
                        batch_size=batch_size,
                        label="is_eos",
                    )
                ]
                validate_parallel_window_causality(
                    features,
                    video_names=names,
                    current_frames=frames,
                    segment_size=int(args.num_frame),
                )
                global_counts["parallel_window_causality_batch_count"] += 1
                for video_name, frame, is_real, is_eos in zip(
                    names, frames, real_flags, eos_flags
                ):
                    previous_physical = last_physical_frame.get(video_name)
                    if previous_physical is not None and frame <= previous_physical:
                        raise RuntimeError("D1.5 physical prefix order is non-monotonic")
                    last_physical_frame[video_name] = frame
                    if is_eos and not is_real:
                        raise RuntimeError("padding cannot expose EOS")
                    if is_real:
                        if video_name in closed_streams:
                            raise RuntimeError("real prefix appeared after observed EOS")
                        previous_real = last_real_frame.get(video_name)
                        if previous_real is not None and frame <= previous_real:
                            raise RuntimeError("D1.5 real prefix order is non-monotonic")
                        last_real_frame[video_name] = frame
                        if is_eos:
                            closed_streams.add(video_name)
                    elif video_name not in closed_streams:
                        raise RuntimeError("padding appeared before observed EOS")

                if "segment_flag" not in infos:
                    raise RuntimeError("D1.5 loader omitted inherited segment_flag")
                stripped = D1_RUNTIME_FORBIDDEN_MODEL_INFO.intersection(infos)
                if stripped != D1_RUNTIME_FORBIDDEN_MODEL_INFO:
                    raise RuntimeError("D1.5 cannot audit all forbidden metadata fields")
                features = features.to(device, non_blocking=True)
                payload = make_model_inputs(args, features, infos)
                if D1_RUNTIME_FORBIDDEN_MODEL_INFO.intersection(payload["infos"]):
                    raise RuntimeError("future/full-video metadata reached query backbone")
                if "event_targets" in payload or "event_valid_mask" in payload:
                    raise RuntimeError("ground truth reached D1.5 query backbone")
                outputs = model(payload, device)
                query_batch_sha256 = _hash_query_outputs(
                    query_digest,
                    outputs,
                    batch_index=batch_index,
                )
                for runtime_name in (
                    "event_birth_count",
                    "event_cancellation_count",
                    "event_end_count",
                    "event_emit_count",
                    "event_reacquisition_count",
                    "event_runtime_capacity_exhaustions",
                ):
                    if bool(outputs[runtime_name].any().item()):
                        raise RuntimeError(
                            f"query-only backbone mutated internal runtime: {runtime_name}"
                        )

                # Ground truth becomes visible only here, after the GT-free
                # query/transition forward has completed.
                target_rows = targets["event_targets"].to(device)
                target_valid = targets["event_valid_mask"].to(device).bool()
                for row_index, (video_name, frame, is_real, is_eos) in enumerate(
                    zip(names, frames, real_flags, eos_flags)
                ):
                    visible_targets = (
                        _parse_targets(target_rows[row_index], target_valid[row_index])
                        if is_real
                        else []
                    )
                    birth_targets = [
                        target for target in visible_targets if target.is_birth
                    ]
                    target_query_map, target_ties = (
                        deterministic_target_query_assignment(
                            visible_targets,
                            class_logits=outputs["pred_cls"][row_index],
                            birth_logits=outputs["event_birth_logits"][row_index],
                            candidate_start_frames=outputs[
                                "event_candidate_start_frames"
                            ][row_index],
                            segment_size=int(args.num_frame),
                            include_birth_evidence=False,
                        )
                        if visible_targets
                        else ({}, 0)
                    )
                    birth_query_map, birth_ties = (
                        deterministic_target_query_assignment(
                            birth_targets,
                            class_logits=outputs["pred_cls"][row_index],
                            birth_logits=outputs["event_birth_logits"][row_index],
                            candidate_start_frames=outputs[
                                "event_candidate_start_frames"
                            ][row_index],
                            segment_size=int(args.num_frame),
                            include_birth_evidence=True,
                        )
                        if birth_targets
                        else ({}, 0)
                    )
                    global_counts["target_query_exact_tie_count"] += target_ties
                    global_counts["birth_query_exact_tie_count"] += birth_ties
                    global_counts["real_prefix_count"] += int(is_real)
                    global_counts["padding_prefix_count"] += int(not is_real)
                    global_counts["visible_target_count"] += len(visible_targets)
                    global_counts["visible_birth_target_count"] += len(birth_targets)
                    global_counts["observed_eos_count"] += int(is_eos)
                    prepared_owners = {}
                    if is_real:
                        prepared_owners = {
                            key: _prepare_owner_batch(
                                state,
                                video_name=video_name,
                                frame=frame,
                                targets=visible_targets,
                                target_query_map=target_query_map,
                                event_query_features=outputs[
                                    "event_query_features"
                                ][row_index],
                                class_logits=outputs["pred_cls"][row_index],
                            )
                            for key, state in routes.items()
                        }
                        _decode_prepared_owner_batches(
                            prepared_owners,
                            owner_decoder=model.module.event_owner_decoder,
                            event_query_features=outputs[
                                "event_query_features"
                            ][row_index],
                        )
                    for key, state in routes.items():
                        _process_route_prefix(
                            state,
                            video_name=video_name,
                            frame=frame,
                            is_real=is_real,
                            is_eos=is_eos,
                            targets=visible_targets,
                            target_query_map=target_query_map,
                            birth_targets=birth_targets,
                            birth_query_map=birth_query_map,
                            outputs=outputs,
                            row_index=row_index,
                            query_batch_sha256=query_batch_sha256,
                            prepared_owner=prepared_owners.get(key),
                            trace_handle=trace_handle,
                        )
                    _process_positive_control_prefix(
                        positive_control,
                        video_name=video_name,
                        frame=frame,
                        is_real=is_real,
                        is_eos=is_eos,
                        targets=visible_targets,
                        birth_targets=birth_targets,
                        birth_query_map=birth_query_map,
                        outputs=outputs,
                        row_index=row_index,
                    )

    elapsed = time.perf_counter() - started_at
    query_hash = query_digest.hexdigest()
    artifacts_after = _artifact_receipt(artifact_paths)
    if artifacts_after != artifacts_before:
        raise RuntimeError("official train artifacts changed during D1.5")
    checkpoint_after = {
        "path": str(cli.checkpoint),
        "bytes": cli.checkpoint.stat().st_size,
        "mtime_ns": cli.checkpoint.stat().st_mtime_ns,
        "sha256": _sha256(cli.checkpoint),
    }
    options_after = {
        "path": str(cli.options),
        "bytes": cli.options.stat().st_size,
        "mtime_ns": cli.options.stat().st_mtime_ns,
        "sha256": _sha256(cli.options),
    }
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during D1.5")
    if options_after != options_before:
        raise RuntimeError("options changed during D1.5")
    if _sha256(cli.manifest) != manifest_sha256:
        raise RuntimeError("manifest changed during D1.5")
    final_status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if final_status:
        raise RuntimeError(f"D1.5 diagnostic source changed:\n{final_status}")

    complete_scan = cli.max_batches == 0
    if complete_scan:
        for key, expected in EXPECTED_COMPLETE_CENSUS.items():
            if global_counts[key] != expected:
                raise RuntimeError(
                    f"D1.5 complete census drifted: {key}={global_counts[key]} "
                    f"!= {expected}"
                )
        if closed_streams != set(dataset.video_list):
            raise RuntimeError("D1.5 complete scan did not observe every EOS")
        pf = routes[("PF", "formal")].counters
        for key, expected in EXPECTED_D14_COUNTS.items():
            if pf[key] != expected:
                raise RuntimeError(
                    f"PF failed exact D1.4 reproduction: {key}={pf[key]} "
                    f"!= {expected}"
                )
        for channel in ("OF", "OR"):
            for route in ROUTES:
                admitted = len(routes[(channel, route)].oracle_admitted_targets)
                if admitted != EXPECTED_COMPLETE_CENSUS[
                    "visible_birth_target_count"
                ]:
                    raise RuntimeError(
                        f"{channel}/{route} did not admit exactly one record per event"
                    )
        control_counts = positive_control.counters
        expected_events = EXPECTED_COMPLETE_CENSUS["visible_birth_target_count"]
        for key in ("birth_count", "end_count", "emit_count"):
            if control_counts[key] != expected_events:
                raise RuntimeError(
                    f"positive control {key} did not close: "
                    f"{control_counts[key]} != {expected_events}"
                )
        for key in (
            "cancellation_count",
            "reacquisition_count",
            "capacity_exhaustion_count",
            "records_active_after_eos",
        ):
            if control_counts[key] != 0:
                raise RuntimeError(f"positive control produced nonzero {key}")

    channel_summaries = {
        channel: {
            route: _route_summary(routes[(channel, route)], query_hash=query_hash)
            for route in ROUTES
        }
        for channel in CHANNELS
    }
    query_hashes = {
        summary["query_stream_sha256"]
        for channel in channel_summaries.values()
        for summary in channel.values()
    }
    if query_hashes != {query_hash}:
        raise RuntimeError("D1.5 channels did not share one exact query stream")
    route_consumption_hashes = {
        summary["query_consumption_sha256"]
        for channel in channel_summaries.values()
        for summary in channel.values()
    }
    if len(route_consumption_hashes) != 1:
        raise RuntimeError("D1.5 channels did not consume one exact prefix stream")
    route_consumption_hash = next(iter(route_consumption_hashes))
    trace_receipt = {
        "path": str(cli.trace_output),
        "bytes": cli.trace_output.stat().st_size,
        "sha256": _sha256(cli.trace_output),
        "compression": "gzip_level_1",
        "format": "chronological_jsonl",
    }
    if trace_receipt["bytes"] <= 0:
        raise RuntimeError("D1.5 trace artifact is empty")

    result = {
        "status": "DIAGNOSTIC_COMPLETE",
        "execution_status": "PASS",
        "status_semantics": (
            "counterfactual_completed_not_model_or_performance_gate_pass"
        ),
        "protocol": PROTOCOL,
        "complete_scan": complete_scan,
        "processed_batches": processed_batches,
        "seed": 52,
        "train_only": True,
        "counterfactual": True,
        "paper_performance_valid": False,
        "strict_causal_paper_result_valid": False,
        "ground_truth_visible_to_query_backbone": False,
        "ground_truth_visible_to_owner_intervention": True,
        "ground_truth_stored_in_runtime_record": False,
        "parallel_window_causality_verified": True,
        "optimizer_constructed": False,
        "optimizer_step_count": 0,
        "checkpoint_updated": False,
        "threshold_search": False,
        "test_access": False,
        "locked_test_release": False,
        "official_comparison_release": False,
        "single_forward_query_stream_shared_across_channels": True,
        "query_stream_sha256": query_hash,
        "route_consumption_stream_sha256": route_consumption_hash,
        "query_compatibility_rule": {
            "birth": (
                "target_class_log_probability_plus_birth_logsigmoid_minus_"
                "absolute_candidate_start_error_over_segment_size"
            ),
            "visible_refresh": (
                "target_class_log_probability_minus_absolute_candidate_start_"
                "error_over_segment_size"
            ),
            "record_target_link": (
                "owner_current_query_cosine_plus_record_target_class_log_"
                "probability_minus_start_error_over_segment_size"
            ),
            "assignment": "deterministic_greedy_one_to_one_no_effect_threshold",
            "target_query_tie_break": "target_event_id_then_query_index",
            "record_target_tie_break": "runtime_event_id_then_target_event_id",
        },
        "global_census": dict(sorted(global_counts.items())),
        "channels": channel_summaries,
        "positive_lifecycle_control": _positive_control_summary(positive_control),
        "trace_artifact": trace_receipt,
        "source_identity": {
            "commit": source_commit,
            "tree": source_tree,
            "clean_start_and_final": True,
        },
        "training_source_identity": training_identity,
        "manifest": {
            "path": str(cli.manifest),
            "bytes": cli.manifest.stat().st_size,
            "sha256": manifest_sha256,
        },
        "checkpoint": checkpoint_before,
        "options": options_before,
        "d14_structure_gate": {
            "path": str(cli.d14_structure_gate),
            "bytes": cli.d14_structure_gate.stat().st_size,
            "sha256": _sha256(cli.d14_structure_gate),
            "binding": d14_binding,
        },
        "official_train_artifacts": artifacts_before,
        "systems": {
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": elapsed,
            "real_prefixes_per_second": (
                global_counts["real_prefix_count"] / elapsed
                if elapsed > 0
                else None
            ),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    cli.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(cli.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "PF_formal": result["channels"]["PF"]["formal"]["counts"],
                "positive_control": result["positive_lifecycle_control"]["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
