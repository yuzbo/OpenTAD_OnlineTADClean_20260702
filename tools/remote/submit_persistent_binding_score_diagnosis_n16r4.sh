#!/usr/bin/env bash

set -euo pipefail

SCREEN_RUN_DIR=${SCREEN_RUN_DIR:?set SCREEN_RUN_DIR to the completed seed-705 screen}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Science_27a59de_20260720}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
DIAGNOSIS_TIME=${DIAGNOSIS_TIME:-00:30:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=705
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}

if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Persistent-binding score diagnosis requires a clean checkout" >&2
    exit 2
fi
CURRENT_BRANCH=$(git -C "$BASE_DIR" branch --show-current)
CURRENT_COMMIT=$(git -C "$BASE_DIR" rev-parse HEAD)
if [[ -n "$EXPECTED_COMMIT" ]]; then
    if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
        echo "EXPECTED_COMMIT must be a full lowercase Git SHA" >&2
        exit 2
    fi
    if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
        echo "Deployment checkout does not match EXPECTED_COMMIT" >&2
        exit 2
    fi
elif [[ "$CURRENT_BRANCH" != "codex/ontad-science-fixed-rematch" ]]; then
    echo "Detached deployment requires EXPECTED_COMMIT" >&2
    exit 2
fi
for arm in fixed rematch; do
    for path in \
        "$SCREEN_RUN_DIR/$arm/screen_result.json" \
        "$SCREEN_RUN_DIR/$arm/gpu1_id0/checkpoint/epoch_0.pth"; do
        [[ -f "$path" ]] || { echo "Missing diagnosis input: $path" >&2; exit 2; }
    done
done

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/score_diagnosis_${STAMP}"
mkdir -p "$RUN_DIR"
COMMIT_SHA=$CURRENT_COMMIT
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_SCREEN_RUN_DIR '%q' "$SCREEN_RUN_DIR"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_COMMIT_SHA '%q' "$COMMIT_SHA"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J pb_score_diag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${DIAGNOSIS_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
SCREEN_RUN_DIR=$Q_SCREEN_RUN_DIR
CPUS_PER_TASK=$Q_CPUS_PER_TASK
COMMIT_SHA=$Q_COMMIT_SHA
SEED=$SEED
SBATCH

cat >> "$SCRIPT_PATH" <<'SBATCH'
set -euo pipefail
cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
export OMP_NUM_THREADS=$CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

test "$(git rev-parse HEAD)" = "$COMMIT_SHA"
test -z "$(git status --porcelain)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

python -m pytest \
    tests/test_persistent_binding_score_diagnosis.py \
    tests/test_persistent_binding_configs.py \
    tests/test_persistent_trajectory_detector.py \
    -q -p no:cacheprovider

python tools/diagnose_persistent_binding_scores.py \
    configs/causaltad/thumos_persistent_binding_fixed_screen.py \
    --checkpoint "$SCREEN_RUN_DIR/fixed/gpu1_id0/checkpoint/epoch_0.pth" \
    --screen-result "$SCREEN_RUN_DIR/fixed/screen_result.json" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/fixed_score_diagnosis.json"

python tools/diagnose_persistent_binding_scores.py \
    configs/causaltad/thumos_persistent_binding_rematch_screen.py \
    --checkpoint "$SCREEN_RUN_DIR/rematch/gpu1_id0/checkpoint/epoch_0.pth" \
    --screen-result "$SCREEN_RUN_DIR/rematch/screen_result.json" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/rematch_score_diagnosis.json"

sha256sum \
    "$RUN_DIR/fixed_score_diagnosis.json" \
    "$RUN_DIR/rematch_score_diagnosis.json" \
    > "$RUN_DIR/artifact_sha256.txt"
SBATCH

echo "PERSISTENT_BINDING_SCORE_DIAGNOSIS_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_SCORE_DIAGNOSIS_COMMIT=$COMMIT_SHA"
SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
printf '%s\n' "$SUBMIT_OUTPUT"
