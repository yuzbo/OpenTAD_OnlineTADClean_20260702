#!/usr/bin/env bash

set -euo pipefail

SMOKE_RUN_DIR=${SMOKE_RUN_DIR:?set SMOKE_RUN_DIR to a passed persistent-binding smoke}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_OnlineTAD_Science_27a59de_20260720}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/persistent_binding}
THUMOS_ROOT=${THUMOS_ROOT:-/data/run01/sczc063/yuzibo/thumos14}
FEATURE_CACHE=${FEATURE_CACHE:-$THUMOS_ROOT/features/pes_siglip2_stride8}
FIT_LIST=${FIT_LIST:-$THUMOS_ROOT/manifests/persistent_binding/thumos_fit_core_160.txt}
CALIBRATION_LIST=${CALIBRATION_LIST:-$THUMOS_ROOT/manifests/persistent_binding/thumos_calibration_40.txt}
REPORTING_LIST=${REPORTING_LIST:-$THUMOS_ROOT/manifests/persistent_binding/thumos_reporting_locked_211.txt}
PROFILE_TIME=${PROFILE_TIME:-00:30:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=705
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
WARMUP_STEPS=50
MEASURED_STEPS=200
PAIRED_GPU_HOUR_CAP=2
SAFETY_FACTOR=1.25
EPOCHS=12

if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Persistent-binding profile requires a clean checkout" >&2
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
if [[ ! -f "$SMOKE_RUN_DIR/gate_summary.json" ]]; then
    echo "Missing smoke gate: $SMOKE_RUN_DIR/gate_summary.json" >&2
    exit 2
fi
if [[ ! -f "$SMOKE_RUN_DIR/split_census.json" ]]; then
    echo "Missing split census from the smoke run" >&2
    exit 2
fi

cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
python - "$SMOKE_RUN_DIR/gate_summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("persistent-binding smoke gate did not pass")
PY
python - "$SMOKE_RUN_DIR/split_census.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit("persistent-binding split census did not pass")
PY

for path in \
    "$FEATURE_CACHE/manifest.json" \
    "$FIT_LIST" \
    "$CALIBRATION_LIST" \
    "$REPORTING_LIST"; do
    [[ -f "$path" ]] || { echo "Missing profile input: $path" >&2; exit 2; }
done

REPORTING_CHUNKS=$(python - "$FEATURE_CACHE/manifest.json" "$REPORTING_LIST" <<'PY'
import json
import math
import sys

videos = json.load(open(sys.argv[1], encoding="utf-8"))["videos"]
names = [line.strip() for line in open(sys.argv[2], encoding="utf-8") if line.strip()]
print(sum(math.ceil(int(videos[name]["num_tokens"]) / 64) for name in names))
PY
)
[[ "$REPORTING_CHUNKS" =~ ^[0-9]+$ && "$REPORTING_CHUNKS" -gt 0 ]] || {
    echo "Invalid reporting chunk count: $REPORTING_CHUNKS" >&2
    exit 2
}

STAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="$RUNS_ROOT/profile_${STAMP}"
mkdir -p "$RUN_DIR"
COMMIT_SHA=$CURRENT_COMMIT
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_FIT_LIST '%q' "$FIT_LIST"
printf -v Q_CALIBRATION_LIST '%q' "$CALIBRATION_LIST"
printf -v Q_REPORTING_CHUNKS '%q' "$REPORTING_CHUNKS"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_SEED '%q' "$SEED"
printf -v Q_COMMIT_SHA '%q' "$COMMIT_SHA"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J pb_profile
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${PROFILE_TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
FIT_LIST=$Q_FIT_LIST
CALIBRATION_LIST=$Q_CALIBRATION_LIST
REPORTING_CHUNKS=$Q_REPORTING_CHUNKS
CPUS_PER_TASK=$Q_CPUS_PER_TASK
SEED=$Q_SEED
COMMIT_SHA=$Q_COMMIT_SHA
WARMUP_STEPS=$WARMUP_STEPS
MEASURED_STEPS=$MEASURED_STEPS
PAIRED_GPU_HOUR_CAP=$PAIRED_GPU_HOUR_CAP
SAFETY_FACTOR=$SAFETY_FACTOR
EPOCHS=$EPOCHS
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
    tests/test_persistent_binding_profile_tools.py \
    tests/test_persistent_binding_configs.py \
    tests/test_persistent_trajectory_detector.py \
    -q -p no:cacheprovider

python tools/profile_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_fixed.py \
    --mode train \
    --split train \
    --warmup-steps "$WARMUP_STEPS" \
    --measured-steps "$MEASURED_STEPS" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/fixed_train_profile.json" \
    --cfg-options dataset.train.allow_list="$FIT_LIST"

python tools/profile_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_rematch.py \
    --mode train \
    --split train \
    --warmup-steps "$WARMUP_STEPS" \
    --measured-steps "$MEASURED_STEPS" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/rematch_train_profile.json" \
    --cfg-options dataset.train.allow_list="$FIT_LIST"

python tools/profile_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_fixed.py \
    --mode inference \
    --split val \
    --warmup-steps "$WARMUP_STEPS" \
    --measured-steps "$MEASURED_STEPS" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/fixed_calibration_inference_profile.json" \
    --cfg-options \
        dataset.val.allow_list="$CALIBRATION_LIST" \
        dataset.val.test_mode=True

python tools/profile_persistent_binding.py \
    configs/causaltad/thumos_persistent_binding_rematch.py \
    --mode inference \
    --split val \
    --warmup-steps "$WARMUP_STEPS" \
    --measured-steps "$MEASURED_STEPS" \
    --device cuda:0 \
    --seed "$SEED" \
    --output "$RUN_DIR/rematch_calibration_inference_profile.json" \
    --cfg-options \
        dataset.val.allow_list="$CALIBRATION_LIST" \
        dataset.val.test_mode=True

python tools/evaluate_persistent_binding_profile.py \
    --fixed-train "$RUN_DIR/fixed_train_profile.json" \
    --rematch-train "$RUN_DIR/rematch_train_profile.json" \
    --fixed-calibration-inference "$RUN_DIR/fixed_calibration_inference_profile.json" \
    --rematch-calibration-inference "$RUN_DIR/rematch_calibration_inference_profile.json" \
    --reporting-chunks "$REPORTING_CHUNKS" \
    --epochs "$EPOCHS" \
    --safety-factor "$SAFETY_FACTOR" \
    --paired-gpu-hour-cap "$PAIRED_GPU_HOUR_CAP" \
    --output "$RUN_DIR/profile_gate.json"
SBATCH

echo "PERSISTENT_BINDING_PROFILE_RUN_DIR=$RUN_DIR"
echo "PERSISTENT_BINDING_PROFILE_COMMIT=$COMMIT_SHA"
echo "PERSISTENT_BINDING_REPORTING_CHUNKS=$REPORTING_CHUNKS"
SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
printf '%s\n' "$SUBMIT_OUTPUT"
