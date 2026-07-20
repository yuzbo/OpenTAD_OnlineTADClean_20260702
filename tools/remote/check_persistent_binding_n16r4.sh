#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_persistent_binding_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}
PYTHON_BIN=${PYTHON_BIN:-python3}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Missing Python 3 interpreter: $PYTHON_BIN" >&2
    exit 2
}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" || true
    sacct -j "$JOB_ID" \
        --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocTRES || true
fi

for path in \
    "$RUN_DIR"/manifests/selection.json \
    "$RUN_DIR"/direct_fixed_train.json \
    "$RUN_DIR"/direct_rematch_train.json \
    "$RUN_DIR"/gate_summary.json; do
    if [[ -f "$path" ]]; then
        echo "===== $(basename "$path") ====="
        "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "passed": payload.get("passed"),
    "binding_mode": payload.get("binding_mode"),
    "chunks_processed": payload.get("chunks_processed"),
    "dropped_gt_birth_targets": payload.get("dropped_gt_birth_targets"),
    "runtime_capacity_exhaustions": payload.get("runtime_capacity_exhaustions"),
    "checkpoint_reload_emissions_identical": payload.get(
        "checkpoint_reload_emissions_identical"
    ),
    "emission_summary": payload.get("emission_summary"),
    "selections": payload.get("selections"),
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
