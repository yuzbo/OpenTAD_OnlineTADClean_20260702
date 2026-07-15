import hashlib

import pytest

from opentad.utils.immutable_event_ledger import (
    ImmutableEventLedger,
    LedgerVerificationError,
    persist_verified_ledger,
)
from opentad.utils.online_protocol import (
    verify_emission_result_dict_rows,
    verified_emission_result_dict,
)


def _event(event_id, *, video_id="video-a", emit_frame=10, slot_id=0):
    provenance = hashlib.sha256(f"{video_id}:{emit_frame}".encode("utf-8")).hexdigest()
    return {
        "event_id": event_id,
        "stream_id": f"stream:{video_id}",
        "stream_key": f"stream:{video_id}",
        "video_id": video_id,
        "immutable": True,
        "slot_id": slot_id,
        "label": "action",
        "score": 0.75,
        "start_frame": emit_frame - 2,
        "end_frame": emit_frame,
        "emit_frame": emit_frame,
        "source_frame": emit_frame,
        "provenance_digest": provenance,
        "segment": [(emit_frame - 2) / 30.0, emit_frame / 30.0],
    }


def test_verified_results_preserve_hash_chain_order_for_same_frame_emissions(tmp_path):
    ledger = ImmutableEventLedger()
    first = ledger.append(_event("first", emit_frame=10, slot_id=1))
    second = ledger.append(_event("second", emit_frame=10, slot_id=0))

    ledger_path = tmp_path / "emissions.jsonl"
    commitment_path = tmp_path / "emissions.commitment.json"
    persist_verified_ledger(ledger_path, commitment_path, [first, second])
    verified = verified_emission_result_dict(ledger_path, commitment_path)

    assert [row["sequence"] for row in verified["video-a"]] == [0, 1]
    with pytest.raises(LedgerVerificationError, match="replayed|sequence|reordered"):
        verify_emission_result_dict_rows({"video-a": [second, first]})


def test_verified_results_reject_video_bucket_rebinding():
    ledger = ImmutableEventLedger()
    row = ledger.append(_event("first", video_id="video-a"))

    with pytest.raises(LedgerVerificationError, match="video bucket"):
        verify_emission_result_dict_rows({"video-b": [row]})


def test_verified_results_reject_unhashed_legacy_rows():
    with pytest.raises(LedgerVerificationError, match="envelope"):
        verify_emission_result_dict_rows(
            {
                "video-a": [
                    {
                        "video_id": "video-a",
                        "emit_frame": 10,
                        "source_frame": 10,
                        "end_frame": 10,
                        "immutable": True,
                    }
                ]
            }
        )
