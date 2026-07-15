#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-}
CONFIG=${2:-}
TICKET=${3:-}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_FullPETAL}
PROFILE_TIME=${PROFILE_TIME:-01:00:00}
FORMAL_TIME=${FORMAL_TIME:-04:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
ALLOW_FORMAL=${ALLOW_FORMAL:-0}
PYTHON_BIN=${PYTHON_BIN:-python3}
SBATCH_BIN=${SBATCH_BIN:-/usr/bin/sbatch}

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
    configs/causaltad/thumos_pes_q2_persist_fixed.py|configs/causaltad/thumos_pes_q2_persist_rematch.py|configs/causaltad/thumos_pes_q2_crs_eps_fixed.py|configs/causaltad/thumos_pes_q2_crs_eps_rematch.py)
        ;;
    *)
        echo "CONFIG must be one of the four locked Full PETAL Q2 variants" >&2
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

declare -A TICKET_FIELDS=()
while IFS= read -r -d '' KEY && IFS= read -r -d '' VALUE; do
    TICKET_FIELDS["$KEY"]=$VALUE
done < <("$PYTHON_BIN" "$BASE_DIR/tools/read_full_petal_launch_ticket.py" "$TICKET")

[[ "${TICKET_FIELDS[mode]:-}" == "$MODE" ]] || {
    echo "MODE does not match the launch ticket" >&2
    exit 2
}
[[ "${TICKET_FIELDS[entrypoint]:-}" == "train" ]] || {
    echo "The Slurm launcher requires a train-entrypoint ticket" >&2
    exit 2
}
[[ "${TICKET_FIELDS[deterministic]:-}" == "true" && \
   "${TICKET_FIELDS[not_eval]:-}" == "false" && \
   -z "${TICKET_FIELDS[resume_checkpoint]:-}" ]] || {
    echo "The ticket violates the locked deterministic fresh-run contract" >&2
    exit 2
}

TICKET=$(realpath "$TICKET")
WORK_DIR=${TICKET_FIELDS[work_dir]:-}
RUN_DIR=$(dirname "$WORK_DIR")
[[ "$WORK_DIR" == "$RUN_DIR/work" && "$RUN_DIR" == "$(dirname "$TICKET")" ]] || {
    echo "Ticket artifact root and work_dir are inconsistent" >&2
    exit 2
}
SEED=${TICKET_FIELDS[seed]:-}
RUN_ID=${TICKET_FIELDS[run_id]:-}
[[ "$SEED" =~ ^[0-9]+$ && "$RUN_ID" =~ ^[0-9]+$ ]] || {
    echo "Ticket seed and run id must be non-negative integers" >&2
    exit 2
}
SCRIPT_PATH="$RUN_DIR/job.sbatch"
[[ ! -e "$SCRIPT_PATH" ]] || {
    echo "Refusing to overwrite the ticket-bound Slurm script" >&2
    exit 2
}

printf -v Q_MODE '%q' "$MODE"
printf -v Q_CONFIG '%q' "$CONFIG"
printf -v Q_TICKET '%q' "$(realpath "$TICKET")"
printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_WORK_DIR '%q' "$WORK_DIR"
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
WORK_DIR=$Q_WORK_DIR
CPUS_PER_TASK=$Q_CPUS_PER_TASK
SEED=$Q_SEED
RUN_ID=$Q_RUN_ID
SBATCH

cat >>"$SCRIPT_PATH" <<'SBATCH'
set -euo pipefail

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
[[ -n "${FULL_PETAL_EXECUTION_ATTESTATION_KEY:-}" && -f "$FULL_PETAL_EXECUTION_ATTESTATION_KEY" ]] || {
    echo "Full PETAL job lacks the external execution attestation private key" >&2
    exit 2
}
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
    --cfg-options work_dir="${WORK_DIR}"

if [[ "$MODE" == "profile" ]]; then
    PROFILE_BUNDLE=$(dirname "$TICKET")
    [[ -f "$PROFILE_BUNDLE/fixed_step_profile.json" && \
       -f "$PROFILE_BUNDLE/fixed_step_optimizer_trace.jsonl" && \
       -f "$PROFILE_BUNDLE/fixed_step_optimizer_trace.commitment.json" ]] || {
        echo "Profile launch did not publish the complete ticket-bound evidence bundle" >&2
        exit 2
    }
    if find "$WORK_DIR" -type f -name '*.pth' -print -quit | grep -q .; then
        echo "Profile launch unexpectedly produced a checkpoint" >&2
        exit 2
    fi
fi
SBATCH

echo "FULL_PETAL_RUN_DIR=$RUN_DIR"
"$SBATCH_BIN" "$SCRIPT_PATH"
