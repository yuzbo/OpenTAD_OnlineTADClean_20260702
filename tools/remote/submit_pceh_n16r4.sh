#!/usr/bin/env bash

# Submit strict single-rank PCEH smoke or gated pilot work from the N16R4
# login node. GPU execution always happens inside Slurm.

set -euo pipefail

MODE=${1:-smoke}
CONFIG=${2:-configs/causaltad/thumos_pceh_ontad.py}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_OnlineTADClean_20260702}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/pceh}
RUN_ID=${RUN_ID:-0}
JOB_NAME=${JOB_NAME:-pceh_${MODE}}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SMOKE_TIME=${SMOKE_TIME:-04:00:00}
PILOT_TIME=${PILOT_TIME:-24:00:00}
TIME=${TIME:-}
MEM=${MEM:-}
ALLOW_PILOT=${ALLOW_PILOT:-0}
SMOKE_RUN_DIR=${SMOKE_RUN_DIR:-}

if [[ "$MODE" != "smoke" && "$MODE" != "pilot" ]]; then
    echo "MODE must be smoke or pilot" >&2
    exit 2
fi
if [[ -z "$TIME" ]]; then
    if [[ "$MODE" == "smoke" ]]; then
        TIME=$SMOKE_TIME
    else
        TIME=$PILOT_TIME
    fi
fi
if [[ "$GPUS_PER_NODE" != "1" ]]; then
    # GPUS_PER_NODE != 1 is forbidden until stream-state sharding exists.
    echo "PCEH streaming runs require GPUS_PER_NODE=1" >&2
    exit 2
fi
if [[ ! -d "$BASE_DIR" ]]; then
    echo "Missing remote code directory: $BASE_DIR" >&2
    exit 2
fi
cd "$BASE_DIR"
if [[ ! -f "$CONFIG" ]]; then
    echo "Missing config: $CONFIG" >&2
    exit 2
fi

source tools/env/activate_n16r4_causaltad.sh

python - "$CONFIG" <<'PY'
import importlib.util
import os
import sys
from mmengine.config import Config

config_path = sys.argv[1]
cfg = Config.fromfile(config_path)
required_modules = ("torch", "mmengine", "transformers", "cv2", "nms_1d_cpu")
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing dependencies: " + ", ".join(missing))
for path in (
    cfg.dataset.train.ann_file,
    cfg.dataset.train.class_map,
    cfg.dataset.train.data_path,
    cfg.model.backbone.backbone.model_name,
):
    if not os.path.exists(path):
        raise SystemExit(f"missing configured path: {path}")
if cfg.inference.load_from_raw_predictions:
    raise SystemExit("raw-prediction shortcut is forbidden")
print("pceh-login-preflight: ok")
PY

if [[ "$MODE" == "pilot" ]]; then
    if [[ "$ALLOW_PILOT" != "1" ]]; then
        echo "pilot submission requires ALLOW_PILOT=1" >&2
        exit 2
    fi
    if [[ -z "$SMOKE_RUN_DIR" || ! -d "$SMOKE_RUN_DIR" ]]; then
        echo "pilot submission requires an existing SMOKE_RUN_DIR" >&2
        exit 2
    fi
    python - "$SMOKE_RUN_DIR" <<'PY'
import json
import os
import sys

root = sys.argv[1]
train = json.load(open(os.path.join(root, "train_step_report.json"), encoding="utf-8"))
causal = json.load(open(os.path.join(root, "causal_replay_report.json"), encoding="utf-8"))
if not train["packet_audit"]["passed"]:
    raise SystemExit("packet_audit did not pass")
if not all(step["update_audit"]["passed"] for step in train["steps"]):
    raise SystemExit("update_audit did not pass")
if not causal["passed"]:
    raise SystemExit("causal_replay did not pass")
print("pceh-smoke-gate: passed")
PY
fi

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/${JOB_NAME}_${STAMP}"
mkdir -p "$RUN_DIR"
SCRIPT_PATH="$RUN_DIR/job.sbatch"
MEM_DIRECTIVE=""
if [[ -n "$MEM" ]]; then
    MEM_DIRECTIVE="#SBATCH --mem=${MEM}"
fi

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J ${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
${MEM_DIRECTIVE}
#SBATCH --time=${TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

set -euo pipefail
cd "${BASE_DIR}"
source tools/env/activate_n16r4_causaltad.sh
export OMP_NUM_THREADS=${CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false

if [[ "${MODE}" == "smoke" ]]; then
    python tools/smoke_pceh_stream.py "${CONFIG}" \
        --split train --max-packets 8 --device cuda:0 --train-step \
        --output "${RUN_DIR}/train_step_report.json"
    python tools/smoke_pceh_stream.py "${CONFIG}" \
        --split test --max-packets 8 --device cuda:0 \
        --output "${RUN_DIR}/inference_report.json"
    python tools/audit_pceh_model.py "${CONFIG}" \
        --split test --max-packets 8 --cut-packet-index 3 --device cuda:0 \
        --output "${RUN_DIR}/causal_replay_report.json"
    python - "${RUN_DIR}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
train = json.load(open(os.path.join(root, "train_step_report.json"), encoding="utf-8"))
causal = json.load(open(os.path.join(root, "causal_replay_report.json"), encoding="utf-8"))
checks = {
    "packet_audit": bool(train["packet_audit"]["passed"]),
    "update_audit": all(step["update_audit"]["passed"] for step in train["steps"]),
    "causal_replay": bool(causal["passed"]),
}
checks["passed"] = all(checks.values())
with open(os.path.join(root, "gate_summary.json"), "w", encoding="utf-8") as file:
    json.dump(checks, file, indent=2, sort_keys=True)
if not checks["passed"]:
    raise SystemExit(f"PCEH smoke gate failed: {checks}")
print(f"PCEH smoke gate passed: {checks}")
PY
else
    export MASTER_PORT=\${MASTER_PORT:-\$((20000 + \${SLURM_JOB_ID:-0} % 40000))}
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:\${MASTER_PORT}" \
        tools/train.py "${CONFIG}" --id "${RUN_ID}"
fi
SBATCH

sbatch_args=()
if [[ -n "${SBATCH_PARTITION:-}" ]]; then
    sbatch_args+=(--partition "$SBATCH_PARTITION")
fi
if [[ -n "${SBATCH_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account "$SBATCH_ACCOUNT")
fi
if [[ -n "${SBATCH_QOS:-}" ]]; then
    sbatch_args+=(--qos "$SBATCH_QOS")
fi

echo "PCEH_RUN_DIR=$RUN_DIR"
sbatch "${sbatch_args[@]}" "$SCRIPT_PATH"
