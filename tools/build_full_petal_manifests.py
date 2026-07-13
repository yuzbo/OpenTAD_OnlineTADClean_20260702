#!/usr/bin/env python3
"""Build deterministic Full PETAL data and qualification manifests."""

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.utils.full_petal_data_contract import (  # noqa: E402
    ContractValidationError,
    build_fineaction_qualification_report,
    build_hardware_runtime_manifest,
    build_reporting_universe_manifest,
    build_thumos_manifest_from_annotation_subsets,
    build_thumos_manifest_from_split_files,
    canonical_json_sha256,
    compare_reporting_universe,
    load_id_file,
    load_json,
    save_json,
    sha256_file,
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
    _add_common_output_arguments(fineaction)
    return parser, parser.parse_args(argv)


def _id_file_provenance(path, ids):
    path = Path(path)
    return {
        "kind": "explicit_id_file",
        "source": {
            "name": path.name,
            "sha256": sha256_file(path),
            "content_sha256": canonical_json_sha256(sorted(ids)),
        },
    }


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


def _build_reporting_lock(args):
    ids = load_id_file(args.ids)
    return build_reporting_universe_manifest(
        ids,
        provenance=_id_file_provenance(args.ids, ids),
        seed=args.seed,
        created_at=args.created_at,
        strict=args.strict,
    )


def _build_reporting_comparison(args):
    locked_manifest = load_json(args.locked_manifest)
    observed_ids = load_id_file(args.observed_ids)
    return compare_reporting_universe(
        locked_manifest,
        observed_ids,
        observed_provenance=_id_file_provenance(args.observed_ids, observed_ids),
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
    if "gates" in spec:
        if set(spec) != {"gates"}:
            raise ContractValidationError(
                "FineAction input with a gates field cannot contain other fields"
            )
        gates = spec["gates"]
    else:
        gates = spec
    return build_fineaction_qualification_report(
        gates,
        seed=args.seed,
        created_at=args.created_at,
    )


BUILDERS = {
    "thumos": _build_thumos,
    "reporting-lock": _build_reporting_lock,
    "reporting-compare": _build_reporting_comparison,
    "hardware": _build_hardware,
    "fineaction": _build_fineaction,
}


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        manifest = BUILDERS[args.command](args)
        output = save_json(args.output, manifest)
    except ContractValidationError as exc:
        parser.error(str(exc))
    print(f"FULL_PETAL_MANIFEST={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
