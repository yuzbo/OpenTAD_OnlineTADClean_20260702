#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d16-end-margin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_PAIR_RECEIPT:?MATR_PAIR_RECEIPT is required}"
: "${MATR_ENDPOINT_OUTPUT:?MATR_ENDPOINT_OUTPUT is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_ENDPOINT_OUTPUT}.pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
IDENTITY_ROOT="$(dirname "${MATR_ENDPOINT_OUTPUT}")"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${IDENTITY_ROOT}/source_identity_endpoint_start.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}"

python3 scripts/analyze_eventmatr_d16_endpoint_margin.py \
  --paired-receipt "${MATR_PAIR_RECEIPT}" \
  --output-root "${MATR_ENDPOINT_OUTPUT}" \
  --diagnostic-source-commit "${MATR_SOURCE_COMMIT}" \
  --diagnostic-source-tree "${MATR_SOURCE_TREE}" \
  --manifest-sha256 "${MATR_MANIFEST_SHA256}"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${IDENTITY_ROOT}/source_identity_endpoint_final.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}"
