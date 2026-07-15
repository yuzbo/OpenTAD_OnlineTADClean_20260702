from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_locked_q2_config_builds_and_runs_in_fresh_production_import():
    program = r'''
import torch
from mmengine.config import Config
from opentad.models import build_detector

cfg = Config.fromfile("configs/causaltad/thumos_pes_q2_persist_fixed.py")
detector = build_detector(cfg.model).eval()
inputs = torch.zeros(1, cfg.feature_dim, 2)
masks = torch.ones(1, 2, dtype=torch.bool)
meta = {
    "video_name": "production-smoke",
    "video_id": "production-smoke",
    "stream_id": "full-petal-production-import",
    "input_format": "cached_features",
    "input_provenance_digest": "a" * 64,
    "feature_stride": cfg.feature_stride,
    "fps": cfg.fps,
    "source_frames": (7, 15),
    "current_frame": 15,
}
with torch.no_grad():
    output = detector.infer_step(inputs, masks, meta)
assert type(detector).__name__ == "PersistentTrajectoryOnlineDetector"
assert type(detector.head).__name__ == "PersistentEventSetHead"
assert detector.trajectory_binding_mode == "fixed_birth_slot"
assert detector.head.endpoint_offset_head is None
assert len(output.logits) == 2
assert output.runtime_state.source_frames == (7, 15)
print("FULL_PETAL_PRODUCTION_IMPORT_PASS")
'''
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "FULL_PETAL_PRODUCTION_IMPORT_PASS"
