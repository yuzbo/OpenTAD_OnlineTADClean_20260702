"""Model builders with a dependency-light Full PETAL production route."""

from importlib import import_module

_LEGACY_REGISTERED = False
_EXPORTS = {
    "build_detector": (".builder", "build_detector"),
    "PersistentEventSetHead": (
        ".dense_heads",
        "PersistentEventSetHead",
    ),
    "PersistentEventSetOnlineDetector": (
        ".detectors",
        "PersistentEventSetOnlineDetector",
    ),
    "PersistentTrajectoryOnlineDetector": (
        ".detectors",
        "PersistentTrajectoryOnlineDetector",
    ),
}


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def register_legacy_models():
    """Register the complete OpenTAD model catalogue on demand."""

    global _LEGACY_REGISTERED
    if _LEGACY_REGISTERED:
        return
    register_all_dense_heads = getattr(
        import_module(f"{__name__}.dense_heads"),
        "register_all_dense_heads",
    )
    register_all_detectors = getattr(
        import_module(f"{__name__}.detectors"),
        "register_all_detectors",
    )
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
