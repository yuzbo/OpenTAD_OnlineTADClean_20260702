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


class TinyStreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        self.reset_count = 0
        self.last_read_trace = None

    def reset_online_states(self):
        self.reset_count += 1

    def forward(self, inputs, metas, return_loss=True, **kwargs):
        del kwargs
        hidden = torch.tanh(self.backbone(inputs))
        value = self.head(hidden)
        meta = metas[0]
        self.last_read_trace = {
            "packet_start_frame": meta["packet_start_frame"],
            "packet_end_frame": meta["packet_end_frame"],
            "max_raw_frame_read": meta["packet_end_frame"] - 1,
            "max_cache_source_frame": meta["packet_end_frame"] - 1,
        }
        if return_loss:
            cost = value.square().mean()
            return {"cost": cost, "aux": cost * 0.5}
        return {meta["video_id"]: [{"score": float(value.detach().item())}]}


def _packets(count=2):
    return [
        {
            "inputs": torch.tensor([[0.25 + index, -0.5]]),
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


def test_train_smoke_runs_chronological_packets_and_audits_updates():
    from opentad.utils.stream_smoke import run_stream_smoke

    model = TinyStreamModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report = run_stream_smoke(
        model,
        _packets(),
        optimizer=optimizer,
        max_packets=2,
        required_module_prefixes=("backbone", "head"),
    )

    assert report["mode"] == "train_step"
    assert report["packets_processed"] == 2
    assert model.reset_count == 1
    assert [row["packet_end_frame"] for row in report["read_traces"]] == [8, 16]
    assert all(row["update_audit"]["passed"] for row in report["steps"])
    json.dumps(report)


def test_inference_smoke_collects_results_without_optimizer():
    from opentad.utils.stream_smoke import run_stream_smoke

    model = TinyStreamModel()
    report = run_stream_smoke(model, _packets(1), optimizer=None, max_packets=1)

    assert report["mode"] == "inference"
    assert report["steps"][0]["results"]["v1"][0]["score"] == pytest.approx(
        report["steps"][0]["results"]["v1"][0]["score"]
    )
    assert "update_audit" not in report["steps"][0]


def test_train_smoke_rejects_nonfinite_cost():
    from opentad.utils.stream_smoke import StreamSmokeError, run_stream_smoke

    class NonFiniteModel(TinyStreamModel):
        def forward(self, inputs, metas, return_loss=True, **kwargs):
            losses = super().forward(inputs, metas, return_loss=return_loss, **kwargs)
            if return_loss:
                losses["cost"] = losses["cost"] * float("nan")
            return losses

    model = NonFiniteModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(StreamSmokeError, match="non-finite cost"):
        run_stream_smoke(model, _packets(1), optimizer=optimizer, max_packets=1)


def test_smoke_cli_builds_config_dataset_model_and_optimizer():
    source = (ROOT / "tools" / "smoke_pceh_stream.py").read_text(encoding="utf-8")

    for token in (
        "Config.fromfile",
        "build_dataset",
        "build_dataloader",
        "build_detector",
        "build_optimizer",
        "run_stream_smoke",
        "--output",
    ):
        assert token in source
