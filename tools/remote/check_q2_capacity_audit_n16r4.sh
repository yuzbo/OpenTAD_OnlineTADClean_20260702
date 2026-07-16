#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_q2_capacity_audit_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" || true
    sacct -j "$JOB_ID" \
        --format=JobID,JobName,State,Elapsed,TotalCPU,ExitCode,MaxRSS,AllocTRES \
        || true
fi

if [[ -f "$RUN_DIR/evidence/capacity_summary.json" ]]; then
    python3 - "$RUN_DIR/evidence/capacity_summary.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "status": report.get("status"),
    "annotation_minimum_oracle_k": report.get("annotation", {}).get("minimum_oracle_free_k"),
    "complete_seeds": report.get("chronological_replay", {}).get("complete_seeds"),
    "actual_exhaustions": report.get("gate", {}).get("actual_exhaustions"),
    "legal_zero_candidates": report.get("gate", {}).get("legal_zero_exhaustion_candidates"),
    "selected_contract": report.get("gate", {}).get("selected_contract"),
    "r1_allowed": report.get("gate", {}).get("r1_implementation_allowed"),
}, indent=2, sort_keys=True))
PY
fi

for path in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$path" ]]; then
        echo "===== tail $(basename "$path") ====="
        tail -n 100 "$path"
    fi
done
