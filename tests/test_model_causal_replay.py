import json
from pathlib import Path
import subprocess
import sys

import pytest


_TORCH_PROBE = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    check=False,
    timeout=20,
)
if _TORCH_PROBE.returncode != 0:
    pytest.skip("torch is unavailable in this environment", allow_module_level=True)

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


class CausalAccumulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.running = 0.0
        self.last_step_audit = None

    def reset_online_states(self):
        self.running = 0.0
        self.last_step_audit = None

    def forward(self, inputs, metas, return_loss=False, **kwargs):
        del return_loss, kwargs
        self.running += float((inputs * self.weight).sum().detach().item())
        current = int(metas[0]["packet_end_frame"]) - 1
        self.last_step_audit = {
            "time": current,
            "logits": {"score": self.running},
            "max_raw_frame_read": current,
            "max_cache_source_frame": current,
            "emissions": [],
        }
        return {metas[0]["video_id"]: []}


def _batches(count=4):
    return [
        {
            "inputs": torch.tensor([[float(index + 1)]]),
            "metas": [
                {
                    "video_id": "v1",
                    "packet_start_frame": index * 8,
                    "packet_end_frame": (index + 1) * 8,
                    "is_video_start": index == 0,
                    "is_video_end": index == count - 1,
                }
            ],
        }
        for index in range(count)
    ]


def test_future_perturbation_replay_preserves_every_prefix_field():
    from opentad.utils.model_causal_replay import audit_model_future_perturbation

    report = audit_model_future_perturbation(
        CausalAccumulator(),
        _batches(),
        cut_packet_index=1,
        perturbation_scale=100.0,
    )

    assert report["passed"] is True
    assert report["cut_packet_index"] == 1
    assert report["through_time"] == 15
    assert report["reference_trace"][:2] == report["perturbed_trace"][:2]
    assert report["reference_trace"][2:] != report["perturbed_trace"][2:]
    json.dumps(report)


def test_future_perturbation_requires_a_nonempty_future_suffix():
    from opentad.utils.model_causal_replay import audit_model_future_perturbation

    with pytest.raises(ValueError, match="future packet"):
        audit_model_future_perturbation(
            CausalAccumulator(),
            _batches(2),
            cut_packet_index=1,
        )


def test_future_perturbation_rejects_an_input_insensitive_model():
    from opentad.utils.model_causal_replay import audit_model_future_perturbation

    class InputInsensitive(CausalAccumulator):
        def forward(self, inputs, metas, return_loss=False, **kwargs):
            del inputs, return_loss, kwargs
            current = int(metas[0]["packet_end_frame"]) - 1
            self.last_step_audit = {
                "time": current,
                "logits": {"score": 0.0},
                "max_raw_frame_read": current,
                "max_cache_source_frame": current,
                "emissions": [],
            }
            return {metas[0]["video_id"]: []}

    with pytest.raises(RuntimeError, match="did not change any post-cut"):
        audit_model_future_perturbation(
            InputInsensitive(),
            _batches(),
            cut_packet_index=1,
        )


def test_model_audit_cli_loads_checkpoint_and_writes_replay_report():
    source = (ROOT / "tools" / "audit_pceh_model.py").read_text(encoding="utf-8")

    for token in (
        "Config.fromfile",
        "build_dataset",
        "build_detector",
        "torch.load",
        "audit_model_future_perturbation",
        "--cut-packet-index",
        "--output",
    ):
        assert token in source
