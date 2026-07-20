#!/usr/bin/env bash

set -euo pipefail

SMOKE_RUN_DIR=${SMOKE_RUN_DIR:?set SMOKE_RUN_DIR to the repaired passed smoke}
PROFILE_RUN_DIR=${PROFILE_RUN_DIR:?set PROFILE_RUN_DIR to the repaired strict profile}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Science_27a59de_20260720}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
SCREEN_TIME=${SCREEN_TIME:-02:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=705
PAIRED_GPU_HOUR_CAP=2
FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_fixed_screen.py
REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_rematch_screen.py
PROFILE_GATE="$PROFILE_RUN_DIR/seed705_screen_profile_gate.json"
SMOKE_GATE="$SMOKE_RUN_DIR/gate_summary.json"
SMOKE_CENSUS="$SMOKE_RUN_DIR/split_census.json"

if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Persistent-binding screen requires a clean checkout" >&2
    exit 2
fi
if [[ "$(git -C "$BASE_DIR" branch --show-current)" != "codex/ontad-science-fixed-rematch" ]]; then
    echo "Deployment checkout is on the wrong branch" >&2
    exit 2
fi
for path in \
    "$PROFILE_GATE" \
    "$PROFILE_RUN_DIR/job.sbatch" \
    "$SMOKE_GATE" \
    "$SMOKE_CENSUS" \
    "$SMOKE_RUN_DIR/job.sbatch"; do
    [[ -f "$path" ]] || { echo "Missing screen prerequisite: $path" >&2; exit 2; }
done

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
COMMIT_SHA=$(git rev-parse HEAD)
PROFILE_COMMIT=$(sed -n 's/^COMMIT_SHA=//p' "$PROFILE_RUN_DIR/job.sbatch" | head -n 1)
SMOKE_COMMIT=$(sed -n 's/^COMMIT_SHA=//p' "$SMOKE_RUN_DIR/job.sbatch" | head -n 1)
if [[ "$PROFILE_COMMIT" != "$COMMIT_SHA" || "$SMOKE_COMMIT" != "$COMMIT_SHA" ]]; then
    echo "Smoke/profile evidence does not belong to current commit" >&2
    exit 2
fi
python - "$PROFILE_GATE" "$SMOKE_GATE" "$SMOKE_CENSUS" <<'PY'
import json
import sys

profile, smoke, census = (
    json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]
)
if profile.get("passed") is not True or int(profile.get("epochs", -1)) != 1:
    raise SystemExit("one-epoch profile gate did not pass")
estimate = float(profile["estimated_pair_gpu_hours"]["with_safety_factor"])
cap = float(profile["paired_gpu_hour_cap"])
if estimate > cap or cap != 2.0:
    raise SystemExit("screen profile changed or exceeded the frozen cap")
if smoke.get("passed") is not True:
    raise SystemExit("repaired smoke gate did not pass")
if census.get("passed") is not True:
    raise SystemExit("repaired split census did not pass")
PY

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/screen_seed705_${STAMP}"
mkdir -p "$RUN_DIR"
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_PROFILE_GATE '%q' "$PROFILE_GATE"
printf -v Q_SMOKE_GATE '%q' "$SMOKE_GATE"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_COMMIT_SHA '%q' "$COMMIT_SHA"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J pb_screen705
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${SCREEN_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
PROFILE_GATE=$Q_PROFILE_GATE
SMOKE_GATE=$Q_SMOKE_GATE
CPUS_PER_TASK=$Q_CPUS_PER_TASK
COMMIT_SHA=$Q_COMMIT_SHA
SEED=$SEED
PAIRED_GPU_HOUR_CAP=$PAIRED_GPU_HOUR_CAP
FIXED_CONFIG=$FIXED_CONFIG
REMATCH_CONFIG=$REMATCH_CONFIG
SBATCH

cat >> "$SCRIPT_PATH" <<'SBATCH'
set -euo pipefail
cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
export OMP_NUM_THREADS=$CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

test "$(git rev-parse HEAD)" = "$COMMIT_SHA"
test -z "$(git status --porcelain)"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
test "$GPU_NAME" = "NVIDIA GeForce RTX 4090"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
PAIR_STARTED=$(date +%s)

python -m pytest \
    tests/test_prefix_trajectory_supervision.py \
    tests/test_persistent_event_set_head.py \
    tests/test_persistent_trajectory_detector.py \
    tests/test_online_instance_metrics.py \
    tests/test_persistent_binding_gate.py \
    tests/test_persistent_binding_configs.py \
    tests/test_streaming_feature_dataset.py \
    tests/test_persistent_binding_scientific_contract_tools.py \
    tests/test_persistent_binding_screen_tools.py \
    tests/test_core_single_process_contracts.py \
    -q -p no:cacheprovider

python tools/census_persistent_binding.py \
    "$FIXED_CONFIG" \
    --output "$RUN_DIR/split_census.json"

MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 40000))}

run_arm() {
    local arm=$1
    local config=$2
    local root="$RUN_DIR/$arm"
    local started
    local ended
    started=$(date +%s)
    mkdir -p "$root"

    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
        tools/train.py "$config" \
        --allow-unready-screen \
        --seed "$SEED" \
        --id 0 \
        --cfg-options work_dir="$root"
    MASTER_PORT=$((MASTER_PORT + 1))

    local work="$root/gpu1_id0"
    local checkpoint="$work/checkpoint/epoch_0.pth"
    local audit="$work/training_audit.json"
    test -f "$checkpoint"
    test -f "$audit"

    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
        tools/test.py "$config" \
        --evaluation-role calibration \
        --checkpoint "$checkpoint" \
        --seed "$SEED" \
        --id 0 \
        --cfg-options work_dir="$root"
    MASTER_PORT=$((MASTER_PORT + 1))

    local ledger="$work/calibration/persistent_binding_emissions.json"
    local candidate="$root/calibration_candidate.json"
    local receipt="$root/calibration_receipt.json"
    test -f "$ledger"
    python tools/build_persistent_binding_calibration_candidate.py \
        --repo "$BASE_DIR" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --ledger "$ledger" \
        --census "$RUN_DIR/split_census.json" \
        --seed "$SEED" \
        --arm "$arm" \
        --epoch 0 \
        --output "$candidate"
    python tools/select_persistent_binding_checkpoint.py \
        --candidate "$candidate" \
        --output "$receipt"

    ended=$(date +%s)
    python tools/build_persistent_binding_resource_report.py \
        --repo "$BASE_DIR" \
        --scope "$arm" \
        --started-unix "$started" \
        --ended-unix "$ended" \
        --gpu-count 1 \
        --gpu-name "$GPU_NAME" \
        --slurm-job-id "${SLURM_JOB_ID:-unknown}" \
        --output "$root/resource_report.json"

    python tools/build_persistent_binding_screen_result.py \
        --repo "$BASE_DIR" \
        --config "$config" \
        --checkpoint "$checkpoint" \
        --ledger "$ledger" \
        --training-audit "$audit" \
        --census "$RUN_DIR/split_census.json" \
        --calibration-receipt "$receipt" \
        --resource-report "$root/resource_report.json" \
        --profile-gate "$PROFILE_GATE" \
        --smoke-gate "$SMOKE_GATE" \
        --seed "$SEED" \
        --arm "$arm" \
        --output "$root/screen_result.json"
}

run_arm fixed "$FIXED_CONFIG"
run_arm rematch "$REMATCH_CONFIG"

PAIR_ENDED=$(date +%s)
python tools/build_persistent_binding_resource_report.py \
    --repo "$BASE_DIR" \
    --scope pair \
    --started-unix "$PAIR_STARTED" \
    --ended-unix "$PAIR_ENDED" \
    --gpu-count 1 \
    --gpu-name "$GPU_NAME" \
    --slurm-job-id "${SLURM_JOB_ID:-unknown}" \
    --output "$RUN_DIR/pair_resource_report.json"

set +e
python tools/evaluate_persistent_binding_screen.py \
    --fixed "$RUN_DIR/fixed/screen_result.json" \
    --rematch "$RUN_DIR/rematch/screen_result.json" \
    --resource-report "$RUN_DIR/pair_resource_report.json" \
    --output "$RUN_DIR/screen_gate.json"
GATE_STATUS=$?
set -e

sha256sum \
    "$RUN_DIR/split_census.json" \
    "$RUN_DIR/fixed/screen_result.json" \
    "$RUN_DIR/rematch/screen_result.json" \
    "$RUN_DIR/pair_resource_report.json" \
    "$RUN_DIR/screen_gate.json" \
    > "$RUN_DIR/artifact_sha256.txt"
exit "$GATE_STATUS"
SBATCH

echo "PERSISTENT_BINDING_SCREEN_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_SCREEN_COMMIT=$COMMIT_SHA"
echo "PERSISTENT_BINDING_SCREEN_SEED=$SEED"
SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
printf '%s\n' "$SUBMIT_OUTPUT"
