#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_pes_stage1_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" || true
    sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocTRES || true
fi

for path in "$RUN_DIR"/gate_summary.json "$RUN_DIR"/train_smoke_*.json "$RUN_DIR"/inference_smoke_*.json; do
    if [[ -f "$path" ]]; then
        echo "===== $(basename "$path") ====="
        python - "$path" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(json.dumps({
    "passed": payload.get("passed"),
    "mode": payload.get("mode"),
    "route_variant": payload.get("route_variant"),
    "chunks_processed": payload.get("chunks_processed"),
    "train_smoke": payload.get("train_smoke"),
}, indent=2, sort_keys=True))
PY
    fi
done

for path in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$path" ]]; then
        echo "===== tail $(basename "$path") ====="
        tail -n 100 "$path"
    fi
done
