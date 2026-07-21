#!/usr/bin/env bash

set -euo pipefail

VARIANT=${VARIANT:?set VARIANT to sw, margin, transport, lifecycle, or reserve}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Science_27a59de_20260720}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
SMOKE_RUN_DIR=${SMOKE_RUN_DIR:-/data/run01/sczc063/yuzibo/runs/persistent_binding/smoke_20260721_010459}
PILOT_TIME=${PILOT_TIME:-02:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=705
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
PAIRED_GPU_HOUR_CAP=2
SMOKE_GATE="$SMOKE_RUN_DIR/gate_summary.json"

case "$VARIANT" in
    sw)
        FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_opt_sw_fixed.py
        REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_opt_sw_rematch.py
        ;;
    margin)
        FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_opt_margin_fixed.py
        REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_opt_margin_rematch.py
        ;;
    transport)
        FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_opt_transport_fixed.py
        REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_opt_transport_rematch.py
        ;;
    lifecycle)
        FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_opt_lifecycle_fixed.py
        REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_opt_lifecycle_rematch.py
        ;;
    reserve)
        FIXED_CONFIG=configs/causaltad/thumos_persistent_binding_opt_reserve_fixed.py
        REMATCH_CONFIG=configs/causaltad/thumos_persistent_binding_opt_reserve_rematch.py
        ;;
    *)
        echo "VARIANT must be sw, margin, transport, lifecycle, or reserve" >&2
        exit 2
        ;;
esac

if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Model-optimization pilot requires a clean checkout" >&2
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
for path in "$FIXED_CONFIG" "$REMATCH_CONFIG" "$SMOKE_GATE"; do
    if [[ "$path" = configs/* ]]; then
        path="$BASE_DIR/$path"
    fi
    [[ -f "$path" ]] || { echo "Missing pilot prerequisite: $path" >&2; exit 2; }
done

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/model_opt_${VARIANT}_seed705_${STAMP}"
mkdir -p "$RUN_DIR"
COMMIT_SHA=$CURRENT_COMMIT
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_SMOKE_GATE '%q' "$SMOKE_GATE"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_COMMIT_SHA '%q' "$COMMIT_SHA"
printf -v Q_VARIANT '%q' "$VARIANT"
printf -v Q_FIXED_CONFIG '%q' "$FIXED_CONFIG"
printf -v Q_REMATCH_CONFIG '%q' "$REMATCH_CONFIG"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J pb_opt_${VARIANT}
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${PILOT_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
SMOKE_GATE=$Q_SMOKE_GATE
CPUS_PER_TASK=$Q_CPUS_PER_TASK
COMMIT_SHA=$Q_COMMIT_SHA
VARIANT=$Q_VARIANT
FIXED_CONFIG=$Q_FIXED_CONFIG
REMATCH_CONFIG=$Q_REMATCH_CONFIG
SEED=$SEED
PAIRED_GPU_HOUR_CAP=$PAIRED_GPU_HOUR_CAP
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
PROFILE_DIR="$RUN_DIR/profile"
PROFILE_GATE="$PROFILE_DIR/seed705_screen_profile_gate.json"
mkdir -p "$PROFILE_DIR"

python -m pytest \
    tests/test_prefix_trajectory_supervision.py \
    tests/test_persistent_event_set_head.py \
    tests/test_persistent_trajectory_detector.py \
    tests/test_persistent_binding_configs.py \
    tests/test_persistent_binding_scheduler.py \
    tests/test_online_instance_metrics.py \
    tests/test_persistent_binding_gate.py \
    tests/test_streaming_feature_dataset.py \
    tests/test_persistent_binding_scientific_contract_tools.py \
    tests/test_persistent_binding_screen_tools.py \
    tests/test_persistent_binding_score_diagnosis.py \
    tests/test_core_single_process_contracts.py \
    tests/test_persistent_binding_optimization_comparison.py \
    tests/test_persistent_binding_optimization_activation.py \
    -q -p no:cacheprovider

python tools/census_persistent_binding.py \
    "$FIXED_CONFIG" \
    --output "$RUN_DIR/split_census.json"

REPORTING_CHUNKS=$(python - \
    /data/run01/sczc063/yuzibo/thumos14/features/pes_siglip2_stride8/manifest.json \
    /data/run01/sczc063/yuzibo/thumos14/manifests/persistent_binding/thumos_reporting_locked_211.txt <<'PY'
import json
import math
import sys

videos = json.load(open(sys.argv[1], encoding="utf-8"))["videos"]
names = [
    line.strip()
    for line in open(sys.argv[2], encoding="utf-8")
    if line.strip()
]
print(
    sum(
        math.ceil(int(videos[name]["num_tokens"]) / 64)
        for name in names
    )
)
PY
)
[[ "$REPORTING_CHUNKS" =~ ^[0-9]+$ && "$REPORTING_CHUNKS" -gt 0 ]]

python tools/profile_persistent_binding.py \
    "$FIXED_CONFIG" \
    --mode train \
    --split train \
    --warmup-steps 50 \
    --measured-steps 200 \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$PROFILE_DIR/fixed_train_profile.json"

python tools/profile_persistent_binding.py \
    "$REMATCH_CONFIG" \
    --mode train \
    --split train \
    --warmup-steps 50 \
    --measured-steps 200 \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$PROFILE_DIR/rematch_train_profile.json"

python tools/profile_persistent_binding.py \
    "$FIXED_CONFIG" \
    --mode inference \
    --split val \
    --warmup-steps 50 \
    --measured-steps 200 \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$PROFILE_DIR/fixed_calibration_inference_profile.json" \
    --cfg-options dataset.val.test_mode=True

python tools/profile_persistent_binding.py \
    "$REMATCH_CONFIG" \
    --mode inference \
    --split val \
    --warmup-steps 50 \
    --measured-steps 200 \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$PROFILE_DIR/rematch_calibration_inference_profile.json" \
    --cfg-options dataset.val.test_mode=True

python tools/evaluate_persistent_binding_profile.py \
    --fixed-train "$PROFILE_DIR/fixed_train_profile.json" \
    --rematch-train "$PROFILE_DIR/rematch_train_profile.json" \
    --fixed-calibration-inference "$PROFILE_DIR/fixed_calibration_inference_profile.json" \
    --rematch-calibration-inference "$PROFILE_DIR/rematch_calibration_inference_profile.json" \
    --reporting-chunks "$REPORTING_CHUNKS" \
    --epochs 1 \
    --safety-factor 1.25 \
    --paired-gpu-hour-cap "$PAIRED_GPU_HOUR_CAP" \
    --output "$PROFILE_GATE"

python - \
    "$RUN_DIR/pilot_contract.json" \
    "$VARIANT" \
    "$COMMIT_SHA" \
    "$FIXED_CONFIG" \
    "$REMATCH_CONFIG" <<'PY'
import json
from pathlib import Path
import sys

output, variant, commit, fixed, rematch = sys.argv[1:]
payload = {
    "schema_version": "persistent_binding_model_optimization_deployment.v1",
    "variant": variant,
    "code_commit": commit,
    "seed": 705,
    "epochs": 1,
    "fixed_config": fixed,
    "rematch_config": rematch,
    "input": "fixed_cached_causal_features",
    "fit_only": True,
    "calibration_diagnosis_only": True,
    "reporting_accessed": False,
    "threshold_search": False,
    "raw_rgb_authorized": False,
    "same_commit_profile_required": True,
    "transport_numerical_revision": "batched_soft_sinkhorn_0p25_iter56_v3",
    "rendezvous_port_block_size": 4,
}
Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

JOB_PORT_SLOT=$((${SLURM_JOB_ID:-0} % 10000))
MASTER_PORT=${MASTER_PORT:-$((20000 + JOB_PORT_SLOT * 4))}

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

set +e
python tools/evaluate_persistent_binding_optimization_activation.py \
    --variant "$VARIANT" \
    --fixed-audit "$RUN_DIR/fixed/gpu1_id0/training_audit.json" \
    --rematch-audit "$RUN_DIR/rematch/gpu1_id0/training_audit.json" \
    --output "$RUN_DIR/optimization_activation.json"
ACTIVATION_STATUS=$?
set -e

python tools/diagnose_persistent_binding_scores.py \
    "$FIXED_CONFIG" \
    --checkpoint "$RUN_DIR/fixed/gpu1_id0/checkpoint/epoch_0.pth" \
    --screen-result "$RUN_DIR/fixed/screen_result.json" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/fixed_score_diagnosis.json"

python tools/diagnose_persistent_binding_scores.py \
    "$REMATCH_CONFIG" \
    --checkpoint "$RUN_DIR/rematch/gpu1_id0/checkpoint/epoch_0.pth" \
    --screen-result "$RUN_DIR/rematch/screen_result.json" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/rematch_score_diagnosis.json"

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
    "$RUN_DIR/pilot_contract.json" \
    "$RUN_DIR/split_census.json" \
    "$PROFILE_GATE" \
    "$RUN_DIR/fixed/screen_result.json" \
    "$RUN_DIR/rematch/screen_result.json" \
    "$RUN_DIR/fixed_score_diagnosis.json" \
    "$RUN_DIR/rematch_score_diagnosis.json" \
    "$RUN_DIR/optimization_activation.json" \
    "$RUN_DIR/pair_resource_report.json" \
    "$RUN_DIR/screen_gate.json" \
    > "$RUN_DIR/artifact_sha256.txt"
if (( ACTIVATION_STATUS != 0 || GATE_STATUS != 0 )); then
    exit 1
fi
exit 0
SBATCH

echo "PERSISTENT_BINDING_OPTIMIZATION_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_OPTIMIZATION_VARIANT=$VARIANT"
echo "PERSISTENT_BINDING_OPTIMIZATION_COMMIT=$COMMIT_SHA"
SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
printf '%s\n' "$SUBMIT_OUTPUT"
