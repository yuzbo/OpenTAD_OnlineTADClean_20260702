#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-configs/causaltad/thumos_pes_q2_persist_fixed.py}
RUN_DIR=${2:?usage: submit_q2_capacity_audit_n16r4.sh [CONFIG] RUN_DIR}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_FullPETAL}
CPU_THREADS=${CPU_THREADS:-8}
WALL_TIME=${WALL_TIME:-06:00:00}
MEMORY=${MEMORY:-50G}
PYTHON_BIN=/usr/bin/python3

[[ "$CONFIG" == "configs/causaltad/thumos_pes_q2_persist_fixed.py" ]] || {
    echo "Capacity audit requires the frozen fixed Q2 config" >&2
    exit 2
}
[[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ && "$CPU_THREADS" -le 16 ]] || {
    echo "CPU_THREADS must be an integer in [1,16]" >&2
    exit 2
}
[[ "$WALL_TIME" == "06:00:00" ]] || {
    echo "Capacity audit wall time is frozen at 06:00:00" >&2
    exit 2
}
RUN_DIR=$(realpath -m "$RUN_DIR")
case "$RUN_DIR" in
    /data/run01/sczc063/yuzibo/*|/home/*/run/yuzibo/*)
        ;;
    *)
        echo "RUN_DIR escapes the allowed N16R4 roots" >&2
        exit 2
        ;;
esac
[[ ! -e "$RUN_DIR" ]] || {
    echo "Refusing to overwrite capacity audit run: $RUN_DIR" >&2
    exit 2
}
git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
    echo "Missing deployment checkout: $BASE_DIR" >&2
    exit 2
}
[[ -z "$(git -C "$BASE_DIR" status --porcelain)" ]] || {
    echo "Capacity audit requires a clean deployment checkout" >&2
    exit 2
}

mkdir -p "$RUN_DIR"
SCRIPT_PATH="$RUN_DIR/job.sbatch"
CHECKPOINT_DIR="$RUN_DIR/checkpoints"
EVIDENCE_DIR="$RUN_DIR/evidence"

printf -v Q_CONFIG '%q' "$CONFIG"
printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_CHECKPOINT_DIR '%q' "$CHECKPOINT_DIR"
printf -v Q_EVIDENCE_DIR '%q' "$EVIDENCE_DIR"
printf -v Q_CPU_THREADS '%q' "$CPU_THREADS"

{
cat <<SBATCH
#!/usr/bin/env bash
#SBATCH -J q2_capacity_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPU_THREADS}
#SBATCH --mem=${MEMORY}
#SBATCH --time=${WALL_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

CONFIG=$Q_CONFIG
BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
CHECKPOINT_DIR=$Q_CHECKPOINT_DIR
EVIDENCE_DIR=$Q_EVIDENCE_DIR
CPU_THREADS=$Q_CPU_THREADS
SBATCH

cat <<'SBATCH'
set -euo pipefail

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
[[ -z "$(git status --porcelain)" ]] || {
    echo "Compute-node checkout became dirty before capacity audit" >&2
    exit 2
}
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=$CPU_THREADS
export MKL_NUM_THREADS=$CPU_THREADS
export OPENBLAS_NUM_THREADS=$CPU_THREADS
export NUMEXPR_NUM_THREADS=$CPU_THREADS
export TOKENIZERS_PARALLELISM=false

python tools/build_q2_capacity_checkpoints.py "$CONFIG" \
    --output "$CHECKPOINT_DIR" \
    --seeds 705 706 707

python tools/audit_q2_capacity_lifecycle.py "$CONFIG" \
    --checkpoint-bundle "$CHECKPOINT_DIR" \
    --output "$EVIDENCE_DIR" \
    --cpu-threads "$CPU_THREADS" \
    --wall-time-budget-seconds 21000 \
    --cpu-hour-cap 48

test -f "$EVIDENCE_DIR/capacity_summary.json"
test -f "$EVIDENCE_DIR/capacity_trace.jsonl.gz"
test -f "$EVIDENCE_DIR/commitment.json"
SBATCH
} | "$PYTHON_BIN" "$BASE_DIR/tools/submit_full_petal_slurm_script.py" \
    --output "$SCRIPT_PATH" \
    --submission-cwd "$BASE_DIR"

echo "Q2_CAPACITY_RUN_DIR=$RUN_DIR"
