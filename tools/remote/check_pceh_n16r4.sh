#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=${BASE_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}
source "$BASE_DIR/tools/env/activate_n16r4_causaltad.sh"

RUN_DIR=${1:?usage: check_pceh_n16r4.sh RUN_DIR [JOB_ID]}
JOB_ID=${2:-}

if [[ -n "$JOB_ID" ]]; then
    squeue -j "$JOB_ID" || true
    sacct -j "$JOB_ID" --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS,AllocTRES || true
fi

for file in train_step_report.json inference_report.json causal_replay_report.json gate_summary.json; do
    path="$RUN_DIR/$file"
    if [[ -f "$path" ]]; then
        echo "===== $file ====="
        python - "$path" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="utf-8"))
packet_audit = data.get("packet_audit")
if isinstance(packet_audit, dict):
    packet_audit = packet_audit.get("passed")
summary = {
    "passed": data.get("passed") if data.get("passed") is not None else packet_audit,
    "packets_processed": data.get("packets_processed"),
    "mode": data.get("mode"),
    "cut_packet_index": data.get("cut_packet_index"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
    else
        echo "missing: $path"
    fi
done

for file in "$RUN_DIR"/slurm.*.out "$RUN_DIR"/slurm.*.err; do
    if [[ -f "$file" ]]; then
        echo "===== tail $file ====="
        tail -n 80 "$file"
    fi
done
