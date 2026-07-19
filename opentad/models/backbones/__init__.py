"""Lazy backbone exports keep formal feature-cache imports dependency-scoped."""

from importlib import import_module


_EXPORTS = {
    "BackboneWrapper": (".backbone_wrapper", "BackboneWrapper"),
    "ResNet2Plus1d_TSP": (".r2plus1d_tsp", "ResNet2Plus1d_TSP"),
    "SwinTransformer3D_inv": (".re2tal_swin", "SwinTransformer3D_inv"),
    "ResNet3dSlowFast_inv": (".re2tal_slowfast", "ResNet3dSlowFast_inv"),
    "VisionTransformerCP": (".vit", "VisionTransformerCP"),
    "VisionTransformerAdapter": (".vit_adapter", "VisionTransformerAdapter"),
    "VisionTransformerLadder": (".vit_ladder", "VisionTransformerLadder"),
    "OnlineVideoMAEAdapter": (
        ".online_videomae_adapter",
        "OnlineVideoMAEAdapter",
    ),
    "CausalMotionBranch": (".online_siglip_adapter", "CausalMotionBranch"),
    "OnlineSigLIPFrameEncoder": (
        ".online_siglip_adapter",
        "OnlineSigLIPFrameEncoder",
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

__all__ = [
    "BackboneWrapper",
    "ResNet2Plus1d_TSP",
    "SwinTransformer3D_inv",
    "ResNet3dSlowFast_inv",
    "VisionTransformerCP",
    "VisionTransformerAdapter",
    "VisionTransformerLadder",
    "OnlineVideoMAEAdapter",
    "CausalMotionBranch",
    "OnlineSigLIPFrameEncoder",
]
