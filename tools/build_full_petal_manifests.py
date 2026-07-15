#!/usr/bin/env python3
"""Build deterministic Full PETAL data and qualification manifests."""

import argparse
from pathlib import Path
import sys

from mmengine import Config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_data_contract import (  # noqa: E402
    ContractValidationError,
    THUMOS_DEVELOPMENT_SPLIT_SEED,
    build_fineaction_qualification_report,
    build_hardware_runtime_manifest,
    build_id_file_provenance,
    build_reporting_universe_manifest,
    build_thumos_manifest_from_annotation_subsets,
    build_thumos_manifest_from_split_files,
    compare_reporting_universe,
    load_id_file,
    load_json,
    save_json,
    write_thumos_development_split,
)


HARDWARE_SPEC_FIELDS = {
    "gpu_name",
    "gpu_count",
    "software_versions",
    "precision",
    "dimensions",
    "peak_memory_bytes",
    "elapsed_seconds",
    "steps",
    "throughput",
    "throughput_unit",
}


def _add_common_output_arguments(parser, include_strict=False):
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--timestamp",
        "--created-at",
        dest="created_at",
        required=True,
        help="Explicit timezone-aware ISO-8601 creation timestamp",
    )
    parser.add_argument("--output", type=Path, required=True)
    if include_strict:
        strict = parser.add_mutually_exclusive_group()
        strict.add_argument("--strict", dest="strict", action="store_true")
        strict.add_argument("--no-strict", dest="strict", action="store_false")
        parser.set_defaults(strict=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    thumos = commands.add_parser(
        "thumos",
        help="Lock the canonical THUMOS 160/40 train/validation protocol",
    )
    thumos.add_argument("--annotation", type=Path, required=True)
    thumos.add_argument("--train-split", type=Path)
    thumos.add_argument("--validation-split", type=Path)
    thumos.add_argument("--train-subset", default="train")
    thumos.add_argument("--validation-subset", default="validation")
    thumos.add_argument(
        "--feature-identity",
        "--feature-identity-json",
        dest="feature_identity",
        type=Path,
        required=True,
    )
    thumos.add_argument(
        "--extraction-identity",
        "--extraction-identity-json",
        dest="extraction_identity",
        type=Path,
        required=True,
    )
    _add_common_output_arguments(thumos, include_strict=True)

    development_split = commands.add_parser(
        "thumos-development-split",
        help="Generate the deterministic THUMOS fit-core/calibration split",
    )
    development_split.add_argument("--annotation", type=Path, required=True)
    development_split.add_argument("--train-subset", required=True)
    development_split.add_argument(
        "--chunk-duration-seconds",
        "--chunk-duration",
        dest="chunk_duration_seconds",
        type=float,
        required=True,
    )
    development_split.add_argument(
        "--bounded-memory-seconds",
        "--bounded-memory",
        dest="bounded_memory_seconds",
        type=float,
        required=True,
    )
    development_split.add_argument(
        "--fit-output",
        "--fit-ids-output",
        dest="fit_output",
        type=Path,
        required=True,
    )
    development_split.add_argument(
        "--calibration-output",
        "--calibration-ids-output",
        dest="calibration_output",
        type=Path,
        required=True,
    )
    development_split.add_argument(
        "--seed",
        type=int,
        default=THUMOS_DEVELOPMENT_SPLIT_SEED,
    )
    development_split.add_argument("--output", type=Path, required=True)
    development_strict = development_split.add_mutually_exclusive_group()
    development_strict.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
    )
    development_strict.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
    )
    development_split.set_defaults(strict=True)

    reporting_lock = commands.add_parser(
        "reporting-lock",
        help="Lock the historical 211-ID reporting universe",
    )
    reporting_lock.add_argument("--ids", type=Path, required=True)
    _add_common_output_arguments(reporting_lock, include_strict=True)

    reporting_compare = commands.add_parser(
        "reporting-compare",
        help="Compare an observed 213-ID universe against a historical lock",
    )
    reporting_compare.add_argument("--locked-manifest", type=Path, required=True)
    reporting_compare.add_argument("--observed-ids", type=Path, required=True)
    reporting_compare.add_argument("--difference-reasons", type=Path, required=True)
    _add_common_output_arguments(reporting_compare, include_strict=True)

    hardware = commands.add_parser(
        "hardware",
        help="Validate and lock a fixed-step hardware/runtime JSON spec",
    )
    hardware.add_argument("--input", type=Path, required=True)
    _add_common_output_arguments(hardware)

    fineaction = commands.add_parser(
        "fineaction",
        help="Build a FineAction qualification report from gate evidence JSON",
    )
    fineaction.add_argument("--input", type=Path, required=True)
    fineaction.add_argument(
        "--trust-config",
        type=Path,
        default=ROOT / "configs" / "causaltad" / "thumos_pes_q2_base.py",
    )
    _add_common_output_arguments(fineaction)
    return parser, parser.parse_args(argv)


def _build_thumos(args):
    feature_identity = load_json(args.feature_identity)
    extraction_identity = load_json(args.extraction_identity)
    has_train_file = args.train_split is not None
    has_validation_file = args.validation_split is not None
    if has_train_file != has_validation_file:
        raise ContractValidationError(
            "provide both --train-split and --validation-split, or neither"
        )
    common = {
        "feature_identity": feature_identity,
        "extraction_identity": extraction_identity,
        "seed": args.seed,
        "created_at": args.created_at,
        "strict": args.strict,
    }
    if has_train_file:
        return build_thumos_manifest_from_split_files(
            args.train_split,
            args.validation_split,
            annotation_path=args.annotation,
            **common,
        )
    return build_thumos_manifest_from_annotation_subsets(
        args.annotation,
        train_subset=args.train_subset,
        validation_subset=args.validation_subset,
        **common,
    )


def _build_thumos_development_split(args):
    return write_thumos_development_split(
        args.annotation,
        fit_ids_path=args.fit_output,
        calibration_ids_path=args.calibration_output,
        manifest_path=args.output,
        train_subset=args.train_subset,
        chunk_duration_seconds=args.chunk_duration_seconds,
        bounded_memory_seconds=args.bounded_memory_seconds,
        seed=args.seed,
        strict=args.strict,
    )


def _build_reporting_lock(args):
    ids = load_id_file(args.ids)
    return build_reporting_universe_manifest(
        ids,
        provenance=build_id_file_provenance(args.ids, ids),
        seed=args.seed,
        created_at=args.created_at,
        strict=args.strict,
    )


def _build_reporting_comparison(args):
    locked_manifest = load_json(args.locked_manifest)
    observed_ids = load_id_file(args.observed_ids)
    difference_reasons = load_json(args.difference_reasons)
    return compare_reporting_universe(
        locked_manifest,
        observed_ids,
        observed_provenance=build_id_file_provenance(
            args.observed_ids, observed_ids
        ),
        difference_reasons=difference_reasons,
        seed=args.seed,
        created_at=args.created_at,
        strict=args.strict,
    )


def _build_hardware(args):
    spec = load_json(args.input)
    if not isinstance(spec, dict):
        raise ContractValidationError("hardware input must be a JSON object")
    missing = sorted(HARDWARE_SPEC_FIELDS.difference(spec))
    extra = sorted(set(spec).difference(HARDWARE_SPEC_FIELDS))
    if missing or extra:
        raise ContractValidationError(
            f"hardware input fields do not match schema; missing={missing}, extra={extra}"
        )
    return build_hardware_runtime_manifest(
        **spec,
        seed=args.seed,
        created_at=args.created_at,
    )


def _build_fineaction(args):
    spec = load_json(args.input)
    if not isinstance(spec, dict):
        raise ContractValidationError("FineAction input must be a JSON object")
    if set(spec) != {"sources"} or not isinstance(spec["sources"], dict):
        raise ContractValidationError(
            "FineAction input requires exactly one sources object"
        )
    sources = {}
    for name, raw_path in spec["sources"].items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ContractValidationError(
                f"FineAction source {name} must be a non-empty path"
            )
        path = Path(raw_path).expanduser()
        sources[name] = path if path.is_absolute() else args.input.parent / path
    try:
        cfg = Config.fromfile(str(args.trust_config))
        trust_roots = dict(cfg.reporting_contract.fineaction_trust_roots)
    except Exception as exc:
        raise ContractValidationError(
            f"cannot load pinned FineAction trust roots: {exc}"
        ) from exc
    return build_fineaction_qualification_report(
        sources,
        seed=args.seed,
        created_at=args.created_at,
        trust_roots=trust_roots,
    )


BUILDERS = {
    "thumos": _build_thumos,
    "thumos-development-split": _build_thumos_development_split,
    "reporting-lock": _build_reporting_lock,
    "reporting-compare": _build_reporting_comparison,
    "hardware": _build_hardware,
    "fineaction": _build_fineaction,
}


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        manifest = BUILDERS[args.command](args)
        if args.command == "thumos-development-split":
            output = args.output
        else:
            output = save_json(args.output, manifest)
    except ContractValidationError as exc:
        parser.error(str(exc))
    print(f"FULL_PETAL_MANIFEST={output}")
    if args.command == "thumos-development-split":
        print(f"FULL_PETAL_FIT_IDS={args.fit_output}")
        print(f"FULL_PETAL_CALIBRATION_IDS={args.calibration_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
