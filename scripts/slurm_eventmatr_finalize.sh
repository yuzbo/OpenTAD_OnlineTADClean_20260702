#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"
: "${MATR_SMOKE_RECEIPT:?MATR_SMOKE_RECEIPT is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_OUTPUT_ROOT}/pycache/finalizer"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_bxo_official_thumos14.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_OUTPUT_ROOT}/source_identity_finalizer.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_SMOKE_RECEIPT}"

python3 scripts/verify_eventmatr_completion.py \
  --output-root "${MATR_OUTPUT_ROOT}" \
  --run-tag "${MATR_RUN_TAG}" \
  --source-identity "${MATR_OUTPUT_ROOT}/source_identity.json" \
  --smoke-receipt "${MATR_SMOKE_RECEIPT}" \
  --lane-identity-dir "${MATR_OUTPUT_ROOT}/source_identity_lanes"
