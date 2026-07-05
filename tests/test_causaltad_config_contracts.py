from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_causaltad_n16r4_configs_load_and_keep_protocol():
    Config = pytest.importorskip("mmengine.config").Config
    configs = [
        "configs/causaltad/thumos_i3d_n16r4.py",
        "configs/causaltad/thumos_internvideo2_n16r4.py",
    ]

    for config in configs:
        cfg = Config.fromfile(str(ROOT / config))
        assert cfg.model.type == "VideoMambaSuite"
        assert cfg.model.projection.type == "CausalProj"
        assert cfg.model.rpn_head.type == "ActionFormerHead"
        assert not bool(cfg.inference.load_from_raw_predictions)
        assert not bool(cfg.inference.save_raw_prediction)
        assert cfg.dataset.train.ann_file.endswith("thumos_14_anno.json")
        assert cfg.dataset.val.ann_file.endswith("thumos_14_anno.json")
        assert cfg.dataset.test.ann_file.endswith("thumos_14_anno.json")
        assert "/data/run01/sczc063/yuzibo/thumos14" in cfg.dataset.train.data_path


def test_causal_projection_is_online_causal_route():
    source = read("opentad/models/projections/causal_proj.py")
    assert "class CausalProj" in source
    assert "MAMBA_AVAILABLE" in source
    assert "FLASHATTN_AVAILABLE" in source
    assert "causal=True" in source
    assert "mamba_inner_fn_no_out_proj" in source


def test_vit_adapter_imported_rasterizer_is_present():
    adapter_source = read("opentad/models/backbones/vit_adapter.py")
    rasterizer_source = read("opentad/models/backbones/time_aligned_rasterizer.py")

    assert "from .time_aligned_rasterizer import TimeAlignedRasterizer" in adapter_source
    assert "class TimeAlignedRasterizer" in rasterizer_source
    assert "branch_scale" in rasterizer_source
