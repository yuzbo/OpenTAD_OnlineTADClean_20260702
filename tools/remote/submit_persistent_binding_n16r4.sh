#!/usr/bin/env bash

set -euo pipefail

CONFIG=${1:-configs/causaltad/thumos_persistent_binding_smoke.py}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Science_27a59de_20260720}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
THUMOS_ROOT=${THUMOS_ROOT:-/data/run01/sczc063/yuzibo/thumos14}
FEATURE_CACHE=${FEATURE_CACHE:-$THUMOS_ROOT/features/pes_siglip2_stride8}
ANNOTATION=${ANNOTATION:-$THUMOS_ROOT/annotations/thumos_14_anno.json}
CLASS_MAP=${CLASS_MAP:-$THUMOS_ROOT/annotations/category_idx.txt}
MANIFEST_ROOT=${MANIFEST_ROOT:-$THUMOS_ROOT/manifests/persistent_binding}
FIT_LIST=${FIT_LIST:-$MANIFEST_ROOT/thumos_fit_core_160.txt}
CALIBRATION_LIST=${CALIBRATION_LIST:-$MANIFEST_ROOT/thumos_calibration_40.txt}
REPORTING_LIST=${REPORTING_LIST:-$MANIFEST_ROOT/thumos_reporting_locked_211.txt}
SMOKE_TIME=${SMOKE_TIME:-00:30:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=${SEED:-705}
RUN_ID=${RUN_ID:-0}

if [[ "$CONFIG" != "configs/causaltad/thumos_persistent_binding_smoke.py" ]]; then
    echo "Only the registered persistent-binding smoke config may be submitted" >&2
    exit 2
fi
if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Persistent-binding smoke requires a clean checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ "$(git -C "$BASE_DIR" branch --show-current)" != "codex/ontad-science-fixed-rematch" ]]; then
    echo "Deployment checkout is not on codex/ontad-science-fixed-rematch" >&2
    exit 2
fi

for path in \
    "$BASE_DIR/$CONFIG" \
    "$FEATURE_CACHE/manifest.json" \
    "$ANNOTATION" \
    "$CLASS_MAP" \
    "$FIT_LIST" \
    "$CALIBRATION_LIST" \
    "$REPORTING_LIST"; do
    [[ -f "$path" ]] || { echo "Missing smoke input: $path" >&2; exit 2; }
done

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/smoke_${STAMP}"
MANIFEST_DIR="$RUN_DIR/manifests"
mkdir -p "$MANIFEST_DIR"

python tools/prepare_persistent_binding_smoke_manifests.py \
    --annotation "$ANNOTATION" \
    --feature-manifest "$FEATURE_CACHE/manifest.json" \
    --fit-list "$FIT_LIST" \
    --calibration-list "$CALIBRATION_LIST" \
    --reporting-list "$REPORTING_LIST" \
    --output-dir "$MANIFEST_DIR" \
    > "$RUN_DIR/selection.stdout.json"

TRAIN_LIST="$MANIFEST_DIR/train.txt"
VAL_LIST="$MANIFEST_DIR/val.txt"
TEST_LIST="$MANIFEST_DIR/test.txt"
COMMIT_SHA=$(git rev-parse HEAD)
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_CONFIG '%q' "$CONFIG"
printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_TRAIN_LIST '%q' "$TRAIN_LIST"
printf -v Q_VAL_LIST '%q' "$VAL_LIST"
printf -v Q_TEST_LIST '%q' "$TEST_LIST"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_SEED '%q' "$SEED"
printf -v Q_RUN_ID '%q' "$RUN_ID"
printf -v Q_COMMIT_SHA '%q' "$COMMIT_SHA"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J pb_smoke
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${SMOKE_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

CONFIG=$Q_CONFIG
BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
TRAIN_LIST=$Q_TRAIN_LIST
VAL_LIST=$Q_VAL_LIST
TEST_LIST=$Q_TEST_LIST
CPUS_PER_TASK=$Q_CPUS_PER_TASK
SEED=$Q_SEED
RUN_ID=$Q_RUN_ID
COMMIT_SHA=$Q_COMMIT_SHA
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

python tools/census_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_fixed.py \
    --output "$RUN_DIR/split_census.json"

python -m pytest \
    tests/test_prefix_trajectory_supervision.py \
    tests/test_persistent_event_set_head.py \
    tests/test_persistent_trajectory_detector.py \
    tests/test_online_instance_metrics.py \
    tests/test_persistent_binding_gate.py \
    tests/test_persistent_binding_configs.py \
    tests/test_streaming_feature_dataset.py \
    tests/test_persistent_binding_smoke_tools.py \
    tests/test_core_single_process_contracts.py \
    -q -p no:cacheprovider

python tools/smoke_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_fixed.py \
    --split train \
    --max-chunks 4 \
    --device cuda:0 \
    --seed "$SEED" \
    --train-step \
    --output "$RUN_DIR/direct_fixed_train.json" \
    --cfg-options dataset.train.allow_list="$TRAIN_LIST"

python tools/smoke_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_rematch.py \
    --split train \
    --max-chunks 4 \
    --device cuda:0 \
    --seed "$SEED" \
    --train-step \
    --output "$RUN_DIR/direct_rematch_train.json" \
    --cfg-options dataset.train.allow_list="$TRAIN_LIST"

export MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 40000))}
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    tools/train.py "$CONFIG" \
    --allow-unready-smoke \
    --seed "$SEED" \
    --id "$RUN_ID" \
    --cfg-options \
        work_dir="$RUN_DIR/train_work" \
        dataset.train.allow_list="$TRAIN_LIST" \
        dataset.val.allow_list="$VAL_LIST" \
        dataset.test.allow_list="$TEST_LIST" \
        evaluation.allowed_videos="$TEST_LIST"

TRAIN_WORK="$RUN_DIR/train_work/gpu1_id${RUN_ID}"
CHECKPOINT="$TRAIN_WORK/checkpoint/epoch_0.pth"
TRAIN_LEDGER="$TRAIN_WORK/persistent_binding_emissions.json"
TRAIN_AUDIT="$TRAIN_WORK/training_audit.json"
test -f "$CHECKPOINT"
test -f "$TRAIN_LEDGER"
test -f "$TRAIN_AUDIT"

MASTER_PORT=$((MASTER_PORT + 1))
export MASTER_PORT
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
    --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
    tools/test.py "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --seed "$SEED" \
    --id "$RUN_ID" \
    --cfg-options \
        work_dir="$RUN_DIR/reload_work" \
        dataset.test.allow_list="$TEST_LIST" \
        evaluation.allowed_videos="$TEST_LIST"

RELOAD_WORK="$RUN_DIR/reload_work/gpu1_id${RUN_ID}"
RELOAD_LEDGER="$RELOAD_WORK/persistent_binding_emissions.json"
test -f "$RELOAD_LEDGER"

python tools/verify_persistent_binding_smoke.py "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --train-ledger "$TRAIN_LEDGER" \
    --reload-ledger "$RELOAD_LEDGER" \
    --allowed-videos "$TEST_LIST" \
    --direct-report "$RUN_DIR/direct_fixed_train.json" \
    --direct-report "$RUN_DIR/direct_rematch_train.json" \
    --seed "$SEED" \
    --output "$RUN_DIR/gate_summary.json"
SBATCH

echo "PERSISTENT_BINDING_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_COMMIT=$COMMIT_SHA"
SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
printf '%s\n' "$SUBMIT_OUTPUT"
