#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d11-assoc-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_SCAN_ROOT:?MATR_SCAN_ROOT is required}"
: "${MATR_CHECKPOINT:?MATR_CHECKPOINT is required}"
: "${MATR_CHECKPOINT_SHA256:?MATR_CHECKPOINT_SHA256 is required}"
: "${MATR_OPTIONS:?MATR_OPTIONS is required}"
: "${MATR_OPTIONS_SHA256:?MATR_OPTIONS_SHA256 is required}"
: "${MATR_TRAIN_SOURCE_IDENTITY:?MATR_TRAIN_SOURCE_IDENTITY is required}"
: "${MATR_TRAIN_SOURCE_COMMIT:?MATR_TRAIN_SOURCE_COMMIT is required}"
: "${MATR_TRAIN_SOURCE_TREE:?MATR_TRAIN_SOURCE_TREE is required}"
: "${MATR_MECHANISM_GATE_STATUS:?MATR_MECHANISM_GATE_STATUS is required}"
: "${MATR_SCAN_SOURCE_COMMIT:?MATR_SCAN_SOURCE_COMMIT is required}"
: "${MATR_SCAN_SOURCE_TREE:?MATR_SCAN_SOURCE_TREE is required}"
: "${MATR_SCAN_MANIFEST_SHA256:?MATR_SCAN_MANIFEST_SHA256 is required}"

mkdir -p "${MATR_SCAN_ROOT}"
source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_SCAN_ROOT}/pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_SCAN_ROOT}/scan_source_identity_start.json" \
  --expected-commit "${MATR_SCAN_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SCAN_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_SCAN_MANIFEST_SHA256}"

python3 scripts/run_eventmatr_d11_association_scan.py \
  --checkpoint "${MATR_CHECKPOINT}" \
  --options "${MATR_OPTIONS}" \
  --source-identity "${MATR_TRAIN_SOURCE_IDENTITY}" \
  --expected-training-source-commit "${MATR_TRAIN_SOURCE_COMMIT}" \
  --expected-training-source-tree "${MATR_TRAIN_SOURCE_TREE}" \
  --one-epoch-mechanism-gate-status "${MATR_MECHANISM_GATE_STATUS}" \
  --expected-checkpoint-sha256 "${MATR_CHECKPOINT_SHA256}" \
  --expected-options-sha256 "${MATR_OPTIONS_SHA256}" \
  --output "${MATR_SCAN_ROOT}/association_scan.json"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_SCAN_ROOT}/scan_source_identity_final.json" \
  --expected-commit "${MATR_SCAN_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SCAN_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_SCAN_MANIFEST_SHA256}"
