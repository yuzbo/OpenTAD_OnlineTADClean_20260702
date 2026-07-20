#!/usr/bin/env bash

set -euo pipefail

RUN_DIR=${1:?usage: check_persistent_binding_score_diagnosis_n16r4.sh RUN_DIR [JOB_ID]}
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
for arm in ("fixed", "rematch"):
    path = root / f"{arm}_score_diagnosis.json"
    if not path.is_file():
        print(f"{arm}: pending")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = {
        "passed": payload.get("passed"),
        "tokens": payload.get("dataset_tokens"),
        "lifecycle": payload.get("lifecycle_counts"),
        "channels": {
            channel: {
                key: payload["channel_score_distributions"][channel][
                    "all_slots"
                ].get(key)
                for key in (
                    "mean",
                    "p50",
                    "p95",
                    "p99",
                    "max",
                    "threshold_crossings",
                    "max_minus_threshold",
                )
            }
            for channel in ("birth", "alive", "end")
        },
        "biases": {
            channel: payload["prior_and_checkpoint_bias_audit"][channel]
            for channel in ("birth", "alive", "end")
        },
    }
    print(arm, json.dumps(row, sort_keys=True))
PY

for path in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$path" ]]; then
        echo "===== $path ====="
        tail -n 60 "$path"
    fi
done
