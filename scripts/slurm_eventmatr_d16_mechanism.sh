#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d16-pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-1
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_MECHANISM_ROOT:?MATR_MECHANISM_ROOT is required}"
: "${MATR_MECHANISM_TAG:?MATR_MECHANISM_TAG is required}"
: "${MATR_PROTOCOL_ANNO:?MATR_PROTOCOL_ANNO is required}"
: "${MATR_TRAIN_FEATURE:?MATR_TRAIN_FEATURE is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"
: "${MATR_D16_SMOKE_RECEIPT:?MATR_D16_SMOKE_RECEIPT is required}"

D16_ARMS=(control risk)
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
if (( TASK_ID < 0 || TASK_ID >= ${#D16_ARMS[@]} )); then
  echo "invalid D1.6 array task ${TASK_ID}" >&2
  exit 2
fi
MATR_D16_ARM="${D16_ARMS[${TASK_ID}]}"
RUN_ROOT="${MATR_MECHANISM_ROOT}/${MATR_D16_ARM}/TH"
mkdir -p "${RUN_ROOT}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache"
export MATR_DEVICE=0
export MATR_LANE=TH
export MATR_RUN_TAG="${MATR_MECHANISM_TAG}_${MATR_D16_ARM}"
export MATR_OUTPUT_ROOT="${RUN_ROOT}"
export MATR_D16_ARM
export MATR_PROTOCOL_ANNO MATR_TRAIN_FEATURE MATR_VIDEO_LEN_PATTERN MATR_LABEL_PATTERN

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/source_identity_start.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_D16_SMOKE_RECEIPT}"

bash scripts/train_eventmatr_d16_mechanism.sh

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${RUN_ROOT}/source_identity_final.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_D16_SMOKE_RECEIPT}"

python3 scripts/finalize_eventmatr_d16_mechanism.py \
  --output-root "${RUN_ROOT}" \
  --arm "${MATR_D16_ARM}" \
  --source-commit "${MATR_SOURCE_COMMIT}" \
  --source-tree "${MATR_SOURCE_TREE}" \
  --manifest-sha256 "${MATR_MANIFEST_SHA256}"
