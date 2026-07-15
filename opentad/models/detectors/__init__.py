"""Route-aware detector registration for production imports."""

from importlib import import_module

from .persistent_event_set_ontad import PersistentEventSetOnlineDetector
from .persistent_trajectory_ontad import PersistentTrajectoryOnlineDetector


_LEGACY_DETECTOR_MODULES = (
    "base",
    "single_stage",
    "two_stage",
    "afsd",
    "bmn",
    "gtad",
    "tsi",
    "etad",
    "actionformer",
    "tridet",
    "temporalmaxer",
    "detr",
    "deformable_detr",
    "tadtr",
    "vsgn",
    "mamba",
    "dyfadet",
    "irregular_actionformer",
    "sparse_completion_actionformer",
    "pceh_ontad",
    "query_sparse_detector",
)


def register_all_detectors():
    for module_name in _LEGACY_DETECTOR_MODULES:
        import_module(f"{__name__}.{module_name}")

    from ..builder import DETECTORS
    from .pceh_ontad import PCEHOnlineDetector

    if DETECTORS.get("PCEHOnlineDetector") is None:
        DETECTORS.register_module()(PCEHOnlineDetector)


__all__ = [
    "PersistentEventSetOnlineDetector",
    "PersistentTrajectoryOnlineDetector",
    "register_all_detectors",
]
