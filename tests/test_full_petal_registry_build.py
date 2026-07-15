from pathlib import Path

from mmengine.config import Config

from opentad.models.builder import DETECTORS
from opentad.models.dense_heads.persistent_event_set_head import PersistentEventSetHead
from opentad.models.detectors.persistent_trajectory_ontad import (
    PersistentTrajectoryOnlineDetector,
)


ROOT = Path(__file__).resolve().parents[1]


def test_locked_q2_config_builds_through_real_mmengine_registry():
    cfg = Config.fromfile(ROOT / "configs/causaltad/thumos_pes_q2_persist_fixed.py")

    detector = DETECTORS.build(cfg.model)

    assert isinstance(detector, PersistentTrajectoryOnlineDetector)
    assert isinstance(detector.head, PersistentEventSetHead)
    assert detector.trajectory_binding_mode == "fixed_birth_slot"
    assert detector.head.endpoint_offset_head is None
    assert DETECTORS.__class__.__module__.startswith("mmengine.registry")
