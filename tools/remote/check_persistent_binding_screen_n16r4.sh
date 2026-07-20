#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_persistent_binding_screen_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" -o '%i|%T|%M|%R' || true
    sacct -j "$JOB_ID" \
        --format=JobID,State,ExitCode,Elapsed,AllocTRES%60 \
        -n -P || true
fi

python3 - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for relative in (
    "split_census.json",
    "fixed/screen_result.json",
    "rematch/screen_result.json",
    "pair_resource_report.json",
    "screen_gate.json",
):
    path = root / relative
    if not path.is_file():
        print(f"{relative}: pending")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    if relative == "screen_gate.json":
        print(
            relative,
            {
                "screen_pass": payload.get("screen_pass"),
                "technical_failures": payload.get("technical_failures"),
                "actual_pair_gpu_hours": payload.get(
                    "actual_pair_gpu_hours"
                ),
                "effectiveness_claim_authorized": payload.get(
                    "effectiveness_claim_authorized"
                ),
            },
        )
    elif relative.endswith("screen_result.json"):
        row = payload.get("gate_row", {})
        print(
            relative,
            {
                key: row.get(key)
                for key in (
                    "average_map_pct",
                    "duplicate_rate",
                    "fragmentation_rate",
                    "prediction_gt_ratio",
                    "recall_tiou_0p3",
                    "protocol_violations",
                    "expected_updates",
                    "successful_updates",
                )
            },
        )
    else:
        print(relative, "present")
PY

for path in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$path" ]]; then
        echo "===== $path ====="
        tail -n 100 "$path"
    fi
done
