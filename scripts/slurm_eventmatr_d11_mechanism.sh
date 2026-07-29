#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d11-mechanism
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
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
: "${MATR_D1_SMOKE_RECEIPT:?MATR_D1_SMOKE_RECEIPT is required}"

RUN_ROOT="${MATR_MECHANISM_ROOT}/TH"
mkdir -p "${RUN_ROOT}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${RUN_ROOT}/pycache"
export MATR_DEVICE=0
export MATR_LANE=TH
export MATR_RUN_TAG="${MATR_MECHANISM_TAG}"
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

bash scripts/train_eventmatr_d11_mechanism.sh

python3 scripts/finalize_eventmatr_d11_mechanism.py \
  --output-root "${RUN_ROOT}" \
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
