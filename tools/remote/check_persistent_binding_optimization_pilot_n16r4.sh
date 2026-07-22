#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_persistent_binding_optimization_pilot_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" -o "%.18i %.9P %.24j %.8T %.10M %.6D %R" || true
    sacct -j "$JOB_ID" \
        --format=JobID,JobName%28,State,ExitCode,Elapsed,AllocTRES%48 \
        -n -P || true
fi

python - "$RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
paths = (
    "pilot_contract.json",
    "split_census.json",
    "fixed/screen_result.json",
    "rematch/screen_result.json",
    "optimization_activation.json",
    "fixed_score_diagnosis.json",
    "rematch_score_diagnosis.json",
    "pair_resource_report.json",
    "screen_gate.json",
    "artifact_sha256.txt",
)
for relative in paths:
    path = root / relative
    print(f"{relative}: {'present' if path.exists() else 'missing'}")
    if path.suffix == ".json" and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = {
            key: payload.get(key)
            for key in (
                "schema_version",
                "variant",
                "passed",
                "screen_pass",
                "technical_pass",
                "learning_readiness_pass",
                "operational_pass",
                "epoch1_fixed_threshold_role",
                "formal_fixed_threshold_gate_epoch",
                "reporting_accessed",
                "raw_rgb_authorized",
            )
            if key in payload
        }
        print(json.dumps(summary, sort_keys=True))
PY

for log in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    [[ -f "$log" ]] || continue
    echo "===== $log ====="
    tail -n 120 "$log"
done
