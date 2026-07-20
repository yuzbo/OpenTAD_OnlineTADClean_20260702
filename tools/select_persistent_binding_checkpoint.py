"""Select one checkpoint from calibration-only candidate summaries."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = "persistent_binding_calibration_candidate.v1"
RECEIPT_SCHEMA = "persistent_binding_calibration_receipt.v1"
COMMON_FIELDS = (
    "arm",
    "seed",
    "code_commit",
    "config_sha256",
    "calibration_manifest_sha256",
    "census_sha256",
    "calibration_protocol_sha256",
    "metric_name",
    "metric_unit",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path):
    with Path(path).open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError(f"unexpected calibration candidate schema: {path}")
    if value.get("reporting_accessed") is not False:
        raise ValueError("calibration candidate must not access reporting data")
    checkpoint = Path(value["checkpoint_path"]).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"calibration checkpoint does not exist: {checkpoint}")
    if _sha256(checkpoint) != value.get("checkpoint_sha256"):
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    metric = float(value["metric_value"])
    if not math.isfinite(metric):
        raise ValueError("calibration metric must be finite")
    result = dict(value)
    result["checkpoint_path"] = str(checkpoint)
    result["metric_value"] = metric
    result["epoch"] = int(result["epoch"])
    return result


def select_checkpoint(candidate_paths, output):
    candidates = [_load(path) for path in candidate_paths]
    if not candidates:
        raise ValueError("at least one calibration candidate is required")
    reference = candidates[0]
    for candidate in candidates[1:]:
        for field in COMMON_FIELDS:
            if candidate.get(field) != reference.get(field):
                raise ValueError(
                    f"calibration candidates disagree on {field}"
                )
    selected = min(
        candidates,
        key=lambda row: (
            -row["metric_value"],
            row["epoch"],
            row["checkpoint_sha256"],
        ),
    )
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "arm": selected["arm"],
        "seed": selected["seed"],
        "selection_metric": selected["metric_name"],
        "selection_direction": "maximize",
        "selection_tie_breaking": [
            "lower_epoch",
            "checkpoint_sha256",
        ],
        "metric_unit": selected["metric_unit"],
        "selected_metric_value": selected["metric_value"],
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_path": selected["checkpoint_path"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_emission_ledger_sha256": selected[
            "emission_ledger_sha256"
        ],
        "candidate_count": len(candidates),
        "candidate_summary_sha256": [
            _sha256(path) for path in candidate_paths
        ],
        "code_commit": selected["code_commit"],
        "config_sha256": selected["config_sha256"],
        "calibration_manifest_sha256": selected[
            "calibration_manifest_sha256"
        ],
        "census_sha256": selected["census_sha256"],
        "calibration_protocol_sha256": selected[
            "calibration_protocol_sha256"
        ],
        "reporting_accessed": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = select_checkpoint(args.candidate, args.output)
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
