#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d1-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_PILOT_ROOT:?MATR_PILOT_ROOT is required}"
: "${MATR_PILOT_TAG:?MATR_PILOT_TAG is required}"
: "${MATR_PROTOCOL_ANNO:?MATR_PROTOCOL_ANNO is required}"
: "${MATR_TRAIN_FEATURE:?MATR_TRAIN_FEATURE is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"
: "${MATR_D1_SMOKE_RECEIPT:?MATR_D1_SMOKE_RECEIPT is required}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}"

LANES=(N R T H TH)
HORIZONS=(5 10 20)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
if (( TASK_ID < 0 || TASK_ID >= 15 )); then
  echo "array index must be in [0, 14], got ${TASK_ID}" >&2
  exit 2
fi
LANE="${LANES[$((TASK_ID % 5))]}"
EPOCHS="${HORIZONS[$((TASK_ID / 5))]}"
RUN_ROOT="${MATR_PILOT_ROOT}/e${EPOCHS}/${LANE}"
mkdir -p "${RUN_ROOT}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache"
export MATR_DEVICE=0
export MATR_LANE="${LANE}"
export MATR_PILOT_EPOCHS="${EPOCHS}"
export MATR_RUN_TAG="${MATR_PILOT_TAG}_e${EPOCHS}"
export MATR_OUTPUT_ROOT="${RUN_ROOT}"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/source_identity_start.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_D1_SMOKE_RECEIPT}"

bash scripts/train_eventmatr_d1_pilot.sh

python3 scripts/finalize_eventmatr_d1_pilot.py \
  --output-root "${RUN_ROOT}" \
  --lane "${LANE}" \
  --epochs "${EPOCHS}" \
  --source-commit "${MATR_SOURCE_COMMIT}" \
  --source-tree "${MATR_SOURCE_TREE}" \
  --manifest-sha256 "${MATR_MANIFEST_SHA256}"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/source_identity_final.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_D1_SMOKE_RECEIPT}"
