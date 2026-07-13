#!/usr/bin/env python3
"""Run focused Torch contracts without importing unrelated optional OpenTAD ops."""

import argparse
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


class _FocusedRegistry:
    def __init__(self):
        self.modules = {}

    def register_module(self, module=None, name=None, force=False, **kwargs):
        del force, kwargs

        def register(target):
            self.modules[name or target.__name__] = target
            return target

        return register(module) if module is not None else register

    def build(self, cfg):
        if not isinstance(cfg, dict):
            return cfg
        options = dict(cfg)
        target = options.pop("type")
        if isinstance(target, str):
            if target not in self.modules:
                raise KeyError(f"focused registry has not imported {target!r}")
            target = self.modules[target]
        return target(**options)


def _install_focused_opentad(repo_root):
    packages = (
        ("opentad", "opentad"),
        ("opentad.cores", "opentad/cores"),
        ("opentad.models", "opentad/models"),
        ("opentad.models.bricks", "opentad/models/bricks"),
        ("opentad.models.dense_heads", "opentad/models/dense_heads"),
        ("opentad.models.detectors", "opentad/models/detectors"),
        ("opentad.models.losses", "opentad/models/losses"),
        ("opentad.models.projections", "opentad/models/projections"),
        ("opentad.utils", "opentad/utils"),
        ("opentad.evaluations", "opentad/evaluations"),
    )
    for name, relative_path in packages:
        _namespace(name, repo_root / relative_path)

    registry = _FocusedRegistry()
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
    sys.modules[builder.__name__] = builder
    setattr(sys.modules["opentad.models"], "builder", builder)


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
    _install_focused_opentad(repo_root)
    print(
        "FULL_PETAL_ISOLATED_TORCH="
        f"python={sys.executable} torch={torch.__version__} repo={repo_root}"
    )
    return int(pytest.main(args.pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
