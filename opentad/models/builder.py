from mmengine.registry import Registry
from mmengine.registry import MODELS as MM_BACKBONES

MODELS = Registry("models")

PROJECTIONS = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
PRIOR_GENERATORS = MODELS
PROPOSAL_GENERATORS = MODELS
HEADS = MODELS
TRANSFORMERS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
MATCHERS = MODELS


def build_detector(cfg):
    """Build detector."""
    detector_type = cfg.get("type") if hasattr(cfg, "get") else None
    if DETECTORS.get(detector_type) is None:
        from . import register_legacy_models

        register_legacy_models()
    return DETECTORS.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    if cfg.get("type", None) == "OnlineVideoMAEAdapter":
        return MM_BACKBONES.build(cfg)
    from .backbones import BackboneWrapper

    return BackboneWrapper(cfg)


def build_projection(cfg):
    """Build projection."""
    return PROJECTIONS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_prior_generator(cfg):
    """Build prior generator."""
    return PRIOR_GENERATORS.build(cfg)


def build_proposal_generator(cfg):
    """Build proposal generator."""
    return PROPOSAL_GENERATORS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return ROI_EXTRACTORS.build(cfg)


def build_transformer(cfg):
    """Build transformer in DETR-like model."""
    return TRANSFORMERS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_matcher(cfg):
    """Build external classifier."""
    return MATCHERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)
