"""Route-aware dense-head registration.

The Full PETAL head is dependency-light and registered on normal package import.
Legacy OpenTAD heads remain available through explicit lazy registration.
"""

from importlib import import_module

from .persistent_event_set_head import PersistentEventSetHead


_LEGACY_HEAD_MODULES = (
    "prior_generator",
    "anchor_head",
    "anchor_free_head",
    "rpn_head",
    "afsd_coarse_head",
    "actionformer_head",
    "tridet_head",
    "temporalmaxer_head",
    "tem_head",
    "vsgn_rpn_head",
    "dyn_head",
    "irregular_actionformer_head",
    "irregular_actionformer_bridge_head",
    "irregular_actionformer_head_v2",
    "irregular_actionformer_head_v3",
    "irregular_actionformer_head_v3_oabs",
    "geometry_residual",
    "native_physical_point_head",
    "native_physical_multiscale_head",
    "query_decoder_head",
    "physical_segment_head",
    "matr_head",
    "prefix_event_emission_head",
)


def register_all_dense_heads():
    for module_name in _LEGACY_HEAD_MODULES:
        import_module(f"{__name__}.{module_name}")

    from ..builder import HEADS
    from .prefix_event_emission_head import PrefixEventEmissionHead

    if HEADS.get("PrefixEventEmissionHead") is None:
        HEADS.register_module()(PrefixEventEmissionHead)


__all__ = ["PersistentEventSetHead", "register_all_dense_heads"]
