#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-}
CONFIG=${2:-}
TICKET=${3:-}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_FullPETAL}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/full_petal_q2}
PROFILE_TIME=${PROFILE_TIME:-01:00:00}
FORMAL_TIME=${FORMAL_TIME:-04:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=${SEED:-705}
RUN_ID=${RUN_ID:-0}
ALLOW_FORMAL=${ALLOW_FORMAL:-0}

case "$MODE" in
    profile)
        TIME=$PROFILE_TIME
        ;;
    formal)
        [[ "$ALLOW_FORMAL" == "1" ]] || {
            echo "formal launch requires ALLOW_FORMAL=1" >&2
            exit 2
        }
        TIME=$FORMAL_TIME
        ;;
    *)
        echo "MODE must be profile or formal" >&2
        exit 2
        ;;
esac

case "$CONFIG" in
    configs/causaltad/thumos_pes_q2_persist_fixed.py|configs/causaltad/thumos_pes_q2_persist_rematch.py)
        ;;
    *)
        echo "CONFIG must be one of the two locked Full PETAL Q2 variants" >&2
        exit 2
        ;;
esac

[[ -n "$TICKET" && -f "$TICKET" ]] || {
    echo "A readable launch ticket is required" >&2
    exit 2
}
git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Missing deployment checkout: $BASE_DIR" >&2
    exit 2
}
[[ -z "$(git -C "$BASE_DIR" status --porcelain)" ]] || {
    echo "Full PETAL launch requires a clean checkout" >&2
    exit 2
}

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/${MODE}_$(basename "$CONFIG" .py)_${STAMP}"
mkdir -p "$RUN_DIR"
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_MODE '%q' "$MODE"
printf -v Q_CONFIG '%q' "$CONFIG"
printf -v Q_TICKET '%q' "$(realpath "$TICKET")"
printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_SEED '%q' "$SEED"
printf -v Q_RUN_ID '%q' "$RUN_ID"

cat >"$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J full_petal_${MODE}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

MODE=$Q_MODE
CONFIG=$Q_CONFIG
TICKET=$Q_TICKET
BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
CPUS_PER_TASK=$Q_CPUS_PER_TASK
SEED=$Q_SEED
RUN_ID=$Q_RUN_ID
SBATCH

cat >>"$SCRIPT_PATH" <<'SBATCH'
set -euo pipefail

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
if [[ "$MODE" == "profile" ]]; then
    [[ -n "${FULL_PETAL_PROFILE_ATTESTATION_KEY:-}" && -f "$FULL_PETAL_PROFILE_ATTESTATION_KEY" ]] || {
        echo "Profile job lacks the external attestation private key" >&2
        exit 2
    }
fi
[[ -z "$(git status --porcelain)" ]] || {
    echo "Compute-node checkout became dirty before launch" >&2
    exit 2
}
export OMP_NUM_THREADS=$CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}

torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    tools/train.py "$CONFIG" \
    --seed "$SEED" \
    --id "$RUN_ID" \
    --launch-mode "${MODE}" \
    --launch-ticket "${TICKET}" \
    --cfg-options work_dir="${RUN_DIR}/work"

if [[ "$MODE" == "profile" ]]; then
    PROFILE_COUNT=$(find "$RUN_DIR/work" -type f -name fixed_step_profile.json | wc -l)
    [[ "$PROFILE_COUNT" -eq 1 ]] || {
        echo "Profile launch did not produce exactly one fixed-step artifact" >&2
        exit 2
    }
    if find "$RUN_DIR/work" -type f -name '*.pth' -print -quit | grep -q .; then
        echo "Profile launch unexpectedly produced a checkpoint" >&2
        exit 2
    fi
fi
SBATCH

echo "FULL_PETAL_RUN_DIR=$RUN_DIR"
sbatch "$SCRIPT_PATH"
