"""Model builders with a dependency-light Full PETAL production route."""

from importlib import import_module

from .builder import build_detector
from .dense_heads import PersistentEventSetHead, register_all_dense_heads
from .detectors import (
    PersistentEventSetOnlineDetector,
    PersistentTrajectoryOnlineDetector,
    register_all_detectors,
)


_LEGACY_REGISTERED = False


def register_legacy_models():
    """Register the complete OpenTAD model catalogue on demand."""

    global _LEGACY_REGISTERED
    if _LEGACY_REGISTERED:
        return
    register_all_dense_heads()
    for package in (
        "backbones",
        "projections",
        "necks",
        "roi_heads",
        "losses",
        "transformer",
        "selectors",
    ):
        import_module(f"{__name__}.{package}")
    register_all_detectors()
    _LEGACY_REGISTERED = True


__all__ = [
    "PersistentEventSetHead",
    "PersistentEventSetOnlineDetector",
    "PersistentTrajectoryOnlineDetector",
    "build_detector",
    "register_legacy_models",
]
