#!/usr/bin/env bash

# Submit SigLIP2 online TAD experiments from the N16R4 login node.
# The login node only prepares and submits a Slurm script; training runs inside
# the allocated job.

set -euo pipefail

BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_OnlineTADClean_20260702}
CONFIG=${1:-configs/causaltad/thumos_siglip2_matr_ontad_p0.py}
RUN_ID=${RUN_ID:-0}
JOB_NAME=${JOB_NAME:-ontad_siglip2}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-}
TIME=${TIME:-24:00:00}
EXTRA_ARGS=${EXTRA_ARGS:-}
PREFLIGHT=${PREFLIGHT:-1}

cd "$BASE_DIR"

if [ ! -f "$CONFIG" ]; then
    echo "Missing config: $CONFIG" >&2
    exit 1
fi

if [ ! -f tools/env/activate_n16r4_causaltad.sh ]; then
    echo "Missing tools/env/activate_n16r4_causaltad.sh" >&2
    exit 1
fi

if [ "$GPUS_PER_NODE" != "1" ]; then
    echo "streaming-safe SigLIP/SigLIP2 online TAD routes require GPUS_PER_NODE=1 for now" >&2
    exit 1
fi

if [ "$PREFLIGHT" != "0" ]; then
    source tools/env/activate_n16r4_causaltad.sh
    python - <<'PY'
import importlib.util
import os

checks = {
    "mmengine": "mmengine",
    "torch": "torch",
    "transformers": "transformers",
    "opencv": "cv2",
    "mamba_ssm": "mamba_ssm",
    "causal_conv1d": "causal_conv1d",
    "flash_attn": "flash_attn",
}
missing = [name for name, module in checks.items() if importlib.util.find_spec(module) is None]
if missing:
    raise SystemExit(f"Missing Python dependencies for CausalTAD SigLIP2 route: {', '.join(missing)}")

data_path = "/data/run01/sczc063/yuzibo/thumos14/raw_data/video"
if not os.path.isdir(data_path):
    raise SystemExit(f"Missing raw video data path: {data_path}")
print("preflight-ok: dependencies and raw video path are present")
PY
fi

mkdir -p "$BASE_DIR/slurm"
STAMP=$(date +"%Y%m%d_%H%M%S")
SCRIPT_PATH="$BASE_DIR/slurm/${JOB_NAME}_${STAMP}.sbatch"
MEM_DIRECTIVE=""
if [ -n "$MEM" ]; then
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
#SBATCH -o ${BASE_DIR}/slurm/${JOB_NAME}_${STAMP}.%j.out
#SBATCH -e ${BASE_DIR}/slurm/${JOB_NAME}_${STAMP}.%j.err

set -euo pipefail

cd "${BASE_DIR}"
source tools/env/activate_n16r4_causaltad.sh

export OMP_NUM_THREADS=${CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=\${NCCL_DEBUG:-WARN}

torchrun --standalone --nnodes=1 --nproc_per_node="${GPUS_PER_NODE}" tools/train.py "${CONFIG}" --id "${RUN_ID}" ${EXTRA_ARGS}
SBATCH

sbatch_args=()
if [ -n "${SBATCH_PARTITION:-}" ]; then
    sbatch_args+=(--partition "$SBATCH_PARTITION")
fi
if [ -n "${SBATCH_ACCOUNT:-}" ]; then
    sbatch_args+=(--account "$SBATCH_ACCOUNT")
fi
if [ -n "${SBATCH_QOS:-}" ]; then
    sbatch_args+=(--qos "$SBATCH_QOS")
fi

echo "Submitting $SCRIPT_PATH"
sbatch "${sbatch_args[@]}" "$SCRIPT_PATH"
