#!/usr/bin/env python3
"""Run all B0 contracts with target Torch and a real MMEngine registry."""

import argparse
import importlib
import importlib.machinery
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--dependency-site",
        action="append",
        default=[],
        type=Path,
        help="Append a site-packages directory after importing this env's Torch",
    )
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    if not args.pytest_args:
        parser.error("at least one pytest target is required after --")
    return args


def _namespace(name, path):
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = module.__path__
    module.__spec__ = spec
    sys.modules[name] = module
    parent_name, _, child_name = name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _execute_package_init(name, path):
    module = sys.modules[name]
    init_path = Path(path) / "__init__.py"
    module.__file__ = str(init_path)
    code = compile(init_path.read_text(encoding="utf-8"), str(init_path), "exec")
    exec(code, module.__dict__)
    return module


def _install_cpu_nms_test_double(torch):
    module = types.ModuleType("nms_1d_cpu")
    module.__spec__ = importlib.machinery.ModuleSpec("nms_1d_cpu", loader=None)

    def nms(segs, scores, iou_threshold=0.5):
        del segs, iou_threshold
        return torch.argsort(scores, descending=True).cpu()

    def softnms(
        segs,
        scores,
        dets,
        iou_threshold=0.5,
        sigma=0.5,
        min_score=0.0,
        method=2,
        t1=0.0,
        t2=0.0,
    ):
        del iou_threshold, sigma, method, t1, t2
        indices = torch.argsort(scores, descending=True).cpu()
        kept = indices[scores[indices].cpu() >= float(min_score)]
        if len(kept):
            dets[: len(kept), :2].copy_(segs[kept].cpu())
            dets[: len(kept), 2].copy_(scores[kept].cpu())
        return kept

    module.nms = nms
    module.softnms = softnms
    sys.modules[module.__name__] = module


def _ensure_cpu_nms_module(torch):
    try:
        importlib.import_module("nms_1d_cpu")
    except ModuleNotFoundError as exc:
        if exc.name != "nms_1d_cpu":
            raise
        _install_cpu_nms_test_double(torch)


def _install_b0_opentad(repo_root):
    import torch
    from mmengine.registry import MODELS as MM_MODELS
    from mmengine.registry import Registry

    packages = (
        ("opentad", "opentad"),
        ("opentad.cores", "opentad/cores"),
        ("opentad.models", "opentad/models"),
        ("opentad.models.backbones", "opentad/models/backbones"),
        ("opentad.models.bricks", "opentad/models/bricks"),
        ("opentad.models.dense_heads", "opentad/models/dense_heads"),
        ("opentad.models.detectors", "opentad/models/detectors"),
        ("opentad.models.losses", "opentad/models/losses"),
        ("opentad.models.necks", "opentad/models/necks"),
        ("opentad.models.projections", "opentad/models/projections"),
        ("opentad.models.selectors", "opentad/models/selectors"),
    )
    for name, relative_path in packages:
        _namespace(name, repo_root / relative_path)

    registry = Registry("full_petal_b0_models")
    builder = types.ModuleType("opentad.models.builder")
    builder.__spec__ = importlib.machinery.ModuleSpec(
        "opentad.models.builder", loader=None
    )
    for name in (
        "MODELS",
        "PROJECTIONS",
        "NECKS",
        "ROI_EXTRACTORS",
        "PRIOR_GENERATORS",
        "PROPOSAL_GENERATORS",
        "HEADS",
        "TRANSFORMERS",
        "LOSSES",
        "DETECTORS",
        "MATCHERS",
    ):
        setattr(builder, name, registry)
    for name in (
        "build_detector",
        "build_backbone",
        "build_projection",
        "build_neck",
        "build_prior_generator",
        "build_proposal_generator",
        "build_roi_extractor",
        "build_transformer",
        "build_head",
        "build_matcher",
        "build_loss",
    ):
        setattr(builder, name, registry.build)

    def build_backbone(cfg):
        if cfg.get("type") in {
            "OnlineSigLIPFrameEncoder",
            "OnlineVideoMAEAdapter",
            "CausalFrameSelector",
        }:
            return MM_MODELS.build(cfg)
        return registry.build(cfg)

    builder.build_backbone = build_backbone
    sys.modules[builder.__name__] = builder
    setattr(sys.modules["opentad.models"], "builder", builder)

    _ensure_cpu_nms_module(torch)
    _execute_package_init("opentad.models.bricks", repo_root / "opentad/models/bricks")
    _execute_package_init("opentad.models.losses", repo_root / "opentad/models/losses")

    selected_modules = (
        "opentad.models.backbones.online_siglip_adapter",
        "opentad.models.backbones.online_videomae_adapter",
        "opentad.models.selectors.causal_frame_selector",
        "opentad.models.dense_heads.prior_generator",
        "opentad.models.dense_heads.anchor_free_head",
        "opentad.models.dense_heads.matr_head",
        "opentad.models.projections.temporalmaxer_proj",
        "opentad.models.projections.causal_temporalmaxer_proj",
        "opentad.models.necks.fpn",
        "opentad.models.detectors.mamba",
    )
    for name in selected_modules:
        importlib.import_module(name)
    sys.modules["opentad.models"].build_detector = builder.build_detector


def main(argv=None):
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not (repo_root / "opentad").is_dir():
        raise SystemExit(f"invalid repository root: {repo_root}")

    import torch

    for dependency_site in args.dependency_site:
        resolved = dependency_site.resolve()
        if not resolved.is_dir():
            raise SystemExit(f"dependency site does not exist: {resolved}")
        sys.path.append(str(resolved))

    try:
        import pytest
    except ImportError as exc:
        raise SystemExit(
            "pytest is unavailable; install it in the Torch environment or pass "
            "--dependency-site"
        ) from exc

    sys.path.insert(0, str(repo_root))
    _namespace("tests", repo_root / "tests")
    _install_b0_opentad(repo_root)
    print(
        "FULL_PETAL_ISOLATED_TORCH="
        f"python={sys.executable} torch={torch.__version__} "
        f"registry=mmengine repo={repo_root}"
    )
    return int(pytest.main(args.pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
