#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d14-param-delta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_AUDIT_ROOT:?MATR_AUDIT_ROOT is required}"
: "${MATR_D14_VARIANT:?MATR_D14_VARIANT is required}"
: "${MATR_CHECKPOINT:?MATR_CHECKPOINT is required}"
: "${MATR_CHECKPOINT_SHA256:?MATR_CHECKPOINT_SHA256 is required}"
: "${MATR_OPTIONS:?MATR_OPTIONS is required}"
: "${MATR_OPTIONS_SHA256:?MATR_OPTIONS_SHA256 is required}"
: "${MATR_INITIALIZATION_SOURCE_COMMIT:?MATR_INITIALIZATION_SOURCE_COMMIT is required}"
: "${MATR_TRAIN_SOURCE_COMMIT:?MATR_TRAIN_SOURCE_COMMIT is required}"
: "${MATR_TRAIN_SOURCE_TREE:?MATR_TRAIN_SOURCE_TREE is required}"
: "${MATR_MECHANISM_GATE_STATUS:?MATR_MECHANISM_GATE_STATUS is required}"
: "${MATR_EXPECTED_OPTIMIZER_STEPS:?MATR_EXPECTED_OPTIMIZER_STEPS is required}"
: "${MATR_METRICS:?MATR_METRICS is required}"
: "${MATR_TRAIN_SOURCE_IDENTITY:?MATR_TRAIN_SOURCE_IDENTITY is required}"
: "${MATR_AUDIT_SOURCE_COMMIT:?MATR_AUDIT_SOURCE_COMMIT is required}"
: "${MATR_AUDIT_SOURCE_TREE:?MATR_AUDIT_SOURCE_TREE is required}"
: "${MATR_AUDIT_MANIFEST_SHA256:?MATR_AUDIT_MANIFEST_SHA256 is required}"

mkdir -p "${MATR_AUDIT_ROOT}"
source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_AUDIT_ROOT}/pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_AUDIT_ROOT}/audit_source_identity_start.json" \
  --expected-commit "${MATR_AUDIT_SOURCE_COMMIT}" \
  --expected-tree "${MATR_AUDIT_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_AUDIT_MANIFEST_SHA256}"

python3 scripts/run_eventmatr_d11_parameter_delta_audit.py \
  --checkpoint "${MATR_CHECKPOINT}" \
  --options "${MATR_OPTIONS}" \
  --metrics "${MATR_METRICS}" \
  --source-identity "${MATR_TRAIN_SOURCE_IDENTITY}" \
  --expected-checkpoint-sha256 "${MATR_CHECKPOINT_SHA256}" \
  --expected-options-sha256 "${MATR_OPTIONS_SHA256}" \
  --initialization-source-commit "${MATR_INITIALIZATION_SOURCE_COMMIT}" \
  --expected-training-source-commit "${MATR_TRAIN_SOURCE_COMMIT}" \
  --expected-training-source-tree "${MATR_TRAIN_SOURCE_TREE}" \
  --one-epoch-mechanism-gate-status "${MATR_MECHANISM_GATE_STATUS}" \
  --d13-variant combined \
  --d14-variant "${MATR_D14_VARIANT}" \
  --expected-optimizer-steps "${MATR_EXPECTED_OPTIMIZER_STEPS}" \
  --output "${MATR_AUDIT_ROOT}/parameter_delta_audit.json"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_AUDIT_ROOT}/audit_source_identity_final.json" \
  --expected-commit "${MATR_AUDIT_SOURCE_COMMIT}" \
  --expected-tree "${MATR_AUDIT_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_AUDIT_MANIFEST_SHA256}"
