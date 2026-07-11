#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-smoke}
CONFIG=${2:-configs/causaltad/thumos_pes_stage1_persistent.py}
BASE_DIR=${BASE_DIR:-/data/run01/sczc063/yuzibo/projects/OpenTAD_PES_Stage1_20260712}
RUNS_ROOT=${RUNS_ROOT:-/data/run01/sczc063/yuzibo/runs/pes_stage1}
FEATURE_CACHE=${FEATURE_CACHE:-/data/run01/sczc063/yuzibo/thumos14/features/pes_siglip2_stride8}
ANNOTATION=${ANNOTATION:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}
RAW_VIDEO_DIR=${RAW_VIDEO_DIR:-/data/run01/sczc063/yuzibo/thumos14/raw_data/video}
MODEL_PATH=${MODEL_PATH:-/data/run01/sczc063/yuzibo/hf_models/google-siglip2-base-patch16-224}
HARD_BUDGET_GPU_HOURS=10
PILOT_MAX_SECONDS=4000
CACHE_TIME=${CACHE_TIME:-02:00:00}
SMOKE_TIME=${SMOKE_TIME:-00:30:00}
PILOT_TIME=${PILOT_TIME:-01:00:00}
TIME=${TIME:-}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}
SEED=${SEED:-705}
RUN_ID=${RUN_ID:-0}
ALLOW_PILOT=${ALLOW_PILOT:-0}
SMOKE_RUN_DIR=${SMOKE_RUN_DIR:-}

slurm_time_to_seconds() {
    local raw=$1
    local days=0
    local clock=$raw
    if [[ ! "$raw" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
        echo "Unsupported Slurm time format: $raw" >&2
        return 2
    fi
    if [[ "$raw" == *-* ]]; then
        days=${raw%%-*}
        clock=${raw#*-}
    fi
    local hours minutes seconds
    IFS=: read -r hours minutes seconds <<<"$clock"
    if (( 10#$minutes > 59 || 10#$seconds > 59 )); then
        echo "Invalid Slurm time: $raw" >&2
        return 2
    fi
    echo $((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

if [[ "$MODE" != "cache" && "$MODE" != "smoke" && "$MODE" != "pilot" ]]; then
    echo "MODE must be cache, smoke, or pilot" >&2
    exit 2
fi
if ! git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Missing clean deployment checkout: $BASE_DIR" >&2
    exit 2
fi
if [[ -n "$(git -C "$BASE_DIR" status --porcelain)" ]]; then
    echo "Stage-1 deployment requires a clean checkout: $BASE_DIR" >&2
    exit 2
fi
cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh

if [[ "$MODE" == "cache" ]]; then
    [[ -f "$ANNOTATION" ]] || { echo "Missing annotation: $ANNOTATION" >&2; exit 2; }
    [[ -d "$RAW_VIDEO_DIR" ]] || { echo "Missing raw videos: $RAW_VIDEO_DIR" >&2; exit 2; }
    [[ -d "$MODEL_PATH" ]] || { echo "Missing visual model: $MODEL_PATH" >&2; exit 2; }
    TIME=${TIME:-$CACHE_TIME}
else
    case "$CONFIG" in
        configs/causaltad/thumos_pes_stage1_fresh.py|configs/causaltad/thumos_pes_stage1_trackformer.py|configs/causaltad/thumos_pes_stage1_persistent.py)
            ;;
        *thumos_pceh_ontad_finetune.py*)
            echo "Raw-video finetuning is forbidden before the Stage-1 mechanism gate" >&2
            exit 2
            ;;
        *)
            echo "Unregistered Stage-1 config: $CONFIG" >&2
            exit 2
            ;;
    esac
    [[ -f "$FEATURE_CACHE/manifest.json" ]] || { echo "Missing frozen cache manifest" >&2; exit 2; }
    if [[ "$MODE" == "smoke" ]]; then
        TIME=${TIME:-$SMOKE_TIME}
    else
        TIME=${TIME:-$PILOT_TIME}
    fi
fi

if [[ "$MODE" == "pilot" ]]; then
    [[ "$ALLOW_PILOT" == "1" ]] || { echo "pilot requires ALLOW_PILOT=1" >&2; exit 2; }
    [[ -n "$SMOKE_RUN_DIR" && -f "$SMOKE_RUN_DIR/gate_summary.json" ]] || {
        echo "pilot requires SMOKE_RUN_DIR/gate_summary.json" >&2
        exit 2
    }
    python - "$SMOKE_RUN_DIR/gate_summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("passed"):
    raise SystemExit("Stage-1 smoke gate did not pass")
PY
    case "$SEED" in
        705|706|707) ;;
        *) echo "pilot seed must be one of 705, 706, 707" >&2; exit 2 ;;
    esac
    PILOT_SECONDS=$(slurm_time_to_seconds "$TIME")
    if (( PILOT_SECONDS > PILOT_MAX_SECONDS )); then
        echo "pilot time $TIME exceeds the per-run share of the 10 GPU-hour matrix budget" >&2
        exit 2
    fi
fi

STAMP=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="pes_${MODE}"
if [[ "$MODE" == "pilot" ]]; then
    JOB_NAME="pes_$(basename "$CONFIG" .py)_s${SEED}"
fi
RUN_DIR="$RUNS_ROOT/${JOB_NAME}_${STAMP}"
mkdir -p "$RUN_DIR"
SCRIPT_PATH="$RUN_DIR/job.sbatch"

printf -v Q_MODE '%q' "$MODE"
printf -v Q_CONFIG '%q' "$CONFIG"
printf -v Q_BASE_DIR '%q' "$BASE_DIR"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"
printf -v Q_FEATURE_CACHE '%q' "$FEATURE_CACHE"
printf -v Q_ANNOTATION '%q' "$ANNOTATION"
printf -v Q_RAW_VIDEO_DIR '%q' "$RAW_VIDEO_DIR"
printf -v Q_MODEL_PATH '%q' "$MODEL_PATH"
printf -v Q_CPUS_PER_TASK '%q' "$CPUS_PER_TASK"
printf -v Q_SEED '%q' "$SEED"
printf -v Q_RUN_ID '%q' "$RUN_ID"
printf -v Q_SMOKE_RUN_DIR '%q' "$SMOKE_RUN_DIR"

cat > "$SCRIPT_PATH" <<SBATCH
#!/usr/bin/env bash
#SBATCH -J ${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --time=${TIME}
#SBATCH -o ${RUN_DIR}/slurm.%j.out
#SBATCH -e ${RUN_DIR}/slurm.%j.err

MODE=$Q_MODE
CONFIG=$Q_CONFIG
BASE_DIR=$Q_BASE_DIR
RUN_DIR=$Q_RUN_DIR
FEATURE_CACHE=$Q_FEATURE_CACHE
ANNOTATION=$Q_ANNOTATION
RAW_VIDEO_DIR=$Q_RAW_VIDEO_DIR
MODEL_PATH=$Q_MODEL_PATH
CPUS_PER_TASK=$Q_CPUS_PER_TASK
SEED=$Q_SEED
RUN_ID=$Q_RUN_ID
SMOKE_RUN_DIR=$Q_SMOKE_RUN_DIR
SBATCH

cat >> "$SCRIPT_PATH" <<'SBATCH'
set -euo pipefail
cd "$BASE_DIR"
source tools/env/activate_n16r4_causaltad.sh
export OMP_NUM_THREADS=$CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

if [[ "${MODE}" == "cache" ]]; then
    python tools/analyze_ontad_instances.py "${ANNOTATION}" \
        --subset training \
        --output "${RUN_DIR}/thumos_training_instance_audit.json" >/dev/null
    python tools/analyze_ontad_instances.py "${ANNOTATION}" \
        --subset validation \
        --output "${RUN_DIR}/thumos_validation_instance_audit.json" >/dev/null
    python tools/cache_ontad_features.py \
        --ann-file "${ANNOTATION}" \
        --video-dir "${RAW_VIDEO_DIR}" \
        --output-dir "${FEATURE_CACHE}" \
        --model-name "${MODEL_PATH}" \
        --feature-stride 8 \
        --batch-size 64 \
        --image-size 224 \
        --device cuda:0 \
        --dtype float16 \
        --resume
elif [[ "${MODE}" == "smoke" ]]; then
    python -m pytest \
        tests/test_prefix_instance_schedule.py \
        tests/test_streaming_feature_dataset.py \
        tests/test_persistent_event_set_head.py \
        tests/test_persistent_event_set_detector.py \
        tests/test_ontad_feature_cache.py \
        tests/test_analyze_ontad_instances.py \
        tests/test_pes_stage1_config_contracts.py \
        tests/test_pes_stage1_result_gate.py \
        tests/test_pes_stage1_run_summary.py \
        tests/test_pes_stage1_launch_contracts.py \
        -q -p no:cacheprovider
    for config in \
        configs/causaltad/thumos_pes_stage1_fresh.py \
        configs/causaltad/thumos_pes_stage1_trackformer.py \
        configs/causaltad/thumos_pes_stage1_persistent.py; do
        variant=$(basename "$config" .py)
        python tools/smoke_pes_stage1.py "$config" \
            --split train --max-chunks 4 --device cuda:0 --train-step \
            --output "${RUN_DIR}/train_smoke_${variant}.json"
    done
    python tools/smoke_pes_stage1.py configs/causaltad/thumos_pes_stage1_persistent.py \
        --split test --max-chunks 4 --device cuda:0 \
        --output "${RUN_DIR}/inference_smoke_persistent.json"
    python - "${RUN_DIR}" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
reports = sorted(glob.glob(os.path.join(root, "train_smoke_*.json")))
reports.append(os.path.join(root, "inference_smoke_persistent.json"))
payloads = [json.load(open(path, encoding="utf-8")) for path in reports]
summary = {
    "passed": len(payloads) == 4 and all(item.get("passed") for item in payloads),
    "reports": reports,
    "train_smoke": [item.get("route_variant") for item in payloads if item.get("mode") == "train_step"],
}
with open(os.path.join(root, "gate_summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
if not summary["passed"]:
    raise SystemExit(f"Stage-1 smoke gate failed: {summary}")
PY
else
    python - "${SMOKE_RUN_DIR}/gate_summary.json" <<'PY'
import json
import sys

if not json.load(open(sys.argv[1], encoding="utf-8")).get("passed"):
    raise SystemExit("Stage-1 smoke gate did not pass")
PY
    export MASTER_PORT=${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 40000))}
    TRAIN_START=$(date +%s)
    torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d \
        --rdzv_endpoint "127.0.0.1:${MASTER_PORT}" \
        tools/train.py "${CONFIG}" --seed "${SEED}" --id "${RUN_ID}" \
        --cfg-options work_dir="${RUN_DIR}/work"
    TRAIN_END=$(date +%s)
    GPU_HOURS=$(python - "$TRAIN_START" "$TRAIN_END" <<'PY'
import sys

print((float(sys.argv[2]) - float(sys.argv[1])) / 3600.0)
PY
)
    WORK_DIR="${RUN_DIR}/work/gpu1_id${RUN_ID}"
    LEDGER="${WORK_DIR}/pes_stage1_emission_ledger.json"
    CHECKPOINT=$(find "$WORK_DIR" -maxdepth 1 -type f -name 'epoch_*.pth' -print | sort -V | tail -n 1)
    [[ -f "$LEDGER" ]] || { echo "Missing final immutable ledger: $LEDGER" >&2; exit 2; }
    [[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]] || { echo "Missing final checkpoint" >&2; exit 2; }
    case "$(basename "$CONFIG" .py)" in
        thumos_pes_stage1_fresh) VARIANT=fresh ;;
        thumos_pes_stage1_trackformer) VARIANT=trackformer ;;
        thumos_pes_stage1_persistent) VARIANT=persistent ;;
        *) echo "Cannot map config to Stage-1 variant" >&2; exit 2 ;;
    esac
    python tools/summarize_pes_stage1_run.py \
        --annotation "$ANNOTATION" \
        --ledger "$LEDGER" \
        --subset validation \
        --variant "$VARIANT" \
        --seed "$SEED" \
        --gpu-hours "$GPU_HOURS" \
        --checkpoint "$CHECKPOINT" \
        --output "${RUN_DIR}/run_summary.json"
fi
SBATCH

echo "PES_RUN_DIR=$RUN_DIR"
echo "PES_HARD_BUDGET_GPU_HOURS=$HARD_BUDGET_GPU_HOURS"
if [[ "$MODE" == "pilot" ]]; then
    PILOT_REGISTRY="$RUNS_ROOT/.pilot_registry"
    PILOT_KEY="$(basename "$CONFIG" .py)_s${SEED}"
    PILOT_MARKER="$PILOT_REGISTRY/$PILOT_KEY"
    mkdir -p "$PILOT_REGISTRY"
    if ! mkdir "$PILOT_MARKER" 2>/dev/null; then
        echo "pilot variant/seed already submitted: $PILOT_KEY" >&2
        exit 2
    fi
    if ! SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH"); then
        rmdir "$PILOT_MARKER"
        exit 1
    fi
    printf '%s\n' "$SUBMIT_OUTPUT" > "$PILOT_MARKER/submission.txt"
else
    SUBMIT_OUTPUT=$(sbatch "$SCRIPT_PATH")
fi
printf '%s\n' "$SUBMIT_OUTPUT"
