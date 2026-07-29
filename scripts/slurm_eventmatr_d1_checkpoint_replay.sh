#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d1-replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_PILOT_ROOT:?MATR_PILOT_ROOT is required}"
: "${MATR_REPLAY_ROOT:?MATR_REPLAY_ROOT is required}"
: "${MATR_REPLAY_COMMIT:?MATR_REPLAY_COMMIT is required}"
: "${MATR_REPLAY_TREE:?MATR_REPLAY_TREE is required}"
: "${MATR_REPLAY_MANIFEST_SHA256:?MATR_REPLAY_MANIFEST_SHA256 is required}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

LANES=(R T H TH)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 0 || TASK_ID >= 4 )); then
  echo "replay array index must be in [0, 3], got ${TASK_ID}" >&2
  exit 2
fi
LANE="${LANES[${TASK_ID}]}"
LANE_ROOT="${MATR_PILOT_ROOT}/e5/${LANE}"
OUTPUT_ROOT="${MATR_REPLAY_ROOT}/${LANE}"
mkdir -p "${OUTPUT_ROOT}"

mapfile -t CHECKPOINTS < <(
  find "${LANE_ROOT}/result" -type f -name terminal_epoch5.pth -print
)
mapfile -t OPTIONS < <(
  find "${LANE_ROOT}/result" -type f -name opts.json -print
)
if (( ${#CHECKPOINTS[@]} != 1 || ${#OPTIONS[@]} != 1 )); then
  echo "expected one checkpoint/options pair for ${LANE}" >&2
  printf 'checkpoints: %s\n' "${CHECKPOINTS[*]-}" >&2
  printf 'options: %s\n' "${OPTIONS[*]-}" >&2
  exit 2
fi
PILOT_RECEIPT="${LANE_ROOT}/pilot_receipt.json"
if [[ ! -f "${PILOT_RECEIPT}" ]]; then
  echo "pilot receipt is missing: ${PILOT_RECEIPT}" >&2
  exit 2
fi

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${OUTPUT_ROOT}/pycache"
MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_checkpoint_replay.json"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_ROOT}/source_identity_start.json" \
  --expected-commit "${MATR_REPLAY_COMMIT}" \
  --expected-tree "${MATR_REPLAY_TREE}" \
  --expected-manifest-sha256 "${MATR_REPLAY_MANIFEST_SHA256}"

python3 scripts/run_eventmatr_d1_checkpoint_replay.py \
  --lane "${LANE}" \
  --checkpoint "${CHECKPOINTS[0]}" \
  --options "${OPTIONS[0]}" \
  --pilot-receipt "${PILOT_RECEIPT}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_ROOT}" \
  --trace

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_ROOT}/source_identity_final.json" \
  --expected-commit "${MATR_REPLAY_COMMIT}" \
  --expected-tree "${MATR_REPLAY_TREE}" \
  --expected-manifest-sha256 "${MATR_REPLAY_MANIFEST_SHA256}"
