from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_siglip_route_configs_do_not_overclaim_formal_or_paper_ready_status():
    p1 = _read("configs/causaltad/thumos_siglip2_matr_ontad_p1.py")
    p2 = _read("configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py")

    assert "formal raw-frame online TAD training route" not in p1
    assert "paper-contribution route" not in p2
    assert "candidate" in p1.lower()
    assert "candidate" in p2.lower()
    assert "under validation" in p1.lower()
    assert "under validation" in p2.lower()


def test_all_raw_frame_routes_keep_formal_training_ready_false_until_online_eval_exists():
    for path in [
        "configs/causaltad/thumos_siglip2_matr_ontad_p0.py",
        "configs/causaltad/thumos_siglip2_matr_ontad_p0_smoke.py",
        "configs/causaltad/thumos_siglip2_matr_ontad_p1.py",
        "configs/causaltad/thumos_siglip2_matr_ontad_p1_pilot.py",
        "configs/causaltad/thumos_siglip2_motion_matr_ontad_p2.py",
        "configs/causaltad/thumos_videomae_adapter_matr_ontad.py",
    ]:
        cfg = Config.fromfile(str(ROOT / path))
        assert cfg.formal_training_ready is False


def test_readme_labels_current_raw_frame_routes_as_validation_candidates():
    readme = _read("README.md")
    causaltad_readme = _read("configs/causaltad/README.md")

    assert "validation candidate" in readme.lower()
    assert "validation candidate" in causaltad_readme.lower()
    assert "single-rank" in readme.lower()
    assert "single-rank" in causaltad_readme.lower()
    assert "fully end-to-end trainable video model" not in readme.lower()
