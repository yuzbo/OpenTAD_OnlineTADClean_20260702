#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_persistent_binding_profile_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}
PYTHON_BIN=${PYTHON_BIN:-python3}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" || true
    sacct -j "$JOB_ID" \
        --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocTRES || true
fi

for path in \
    "$RUN_DIR"/fixed_train_profile.json \
    "$RUN_DIR"/rematch_train_profile.json \
    "$RUN_DIR"/fixed_calibration_inference_profile.json \
    "$RUN_DIR"/rematch_calibration_inference_profile.json \
    "$RUN_DIR"/profile_gate.json; do
    if [[ -f "$path" ]]; then
        echo "===== $(basename "$path") ====="
        "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "passed": payload.get("passed"),
    "mode": payload.get("mode"),
    "binding_mode": payload.get("binding_mode"),
    "timing_seconds": payload.get("timing_seconds"),
    "max_gpu_memory_mib": payload.get("max_gpu_memory_mib"),
    "dropped_gt_birth_targets": payload.get("dropped_gt_birth_targets"),
    "runtime_capacity_exhaustions": payload.get("runtime_capacity_exhaustions"),
    "estimated_pair_gpu_hours": payload.get("estimated_pair_gpu_hours"),
    "budget_passed": payload.get("budget_passed"),
    "error": payload.get("error"),
}, indent=2, sort_keys=True))
PY
    fi
done

for path in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$path" ]]; then
        echo "===== tail $(basename "$path") ====="
        tail -n 120 "$path"
    fi
done
