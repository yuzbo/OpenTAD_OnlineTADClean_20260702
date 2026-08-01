#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d15-owner-cf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_D15_ROOT:?MATR_D15_ROOT is required}"
: "${MATR_CHECKPOINT:?MATR_CHECKPOINT is required}"
: "${MATR_CHECKPOINT_SHA256:?MATR_CHECKPOINT_SHA256 is required}"
: "${MATR_OPTIONS:?MATR_OPTIONS is required}"
: "${MATR_OPTIONS_SHA256:?MATR_OPTIONS_SHA256 is required}"
: "${MATR_TRAIN_SOURCE_IDENTITY:?MATR_TRAIN_SOURCE_IDENTITY is required}"
: "${MATR_TRAIN_SOURCE_COMMIT:?MATR_TRAIN_SOURCE_COMMIT is required}"
: "${MATR_TRAIN_SOURCE_TREE:?MATR_TRAIN_SOURCE_TREE is required}"
: "${MATR_D14_SOURCE_COMMIT:?MATR_D14_SOURCE_COMMIT is required}"
: "${MATR_D14_SOURCE_TREE:?MATR_D14_SOURCE_TREE is required}"
: "${MATR_D14_STRUCTURE_GATE:?MATR_D14_STRUCTURE_GATE is required}"
: "${MATR_D14_STRUCTURE_GATE_SHA256:?MATR_D14_STRUCTURE_GATE_SHA256 is required}"
: "${MATR_D15_SOURCE_COMMIT:?MATR_D15_SOURCE_COMMIT is required}"
: "${MATR_D15_SOURCE_TREE:?MATR_D15_SOURCE_TREE is required}"
: "${MATR_D15_MANIFEST_SHA256:?MATR_D15_MANIFEST_SHA256 is required}"

mkdir -p "${MATR_D15_ROOT}"
source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_D15_ROOT}/pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
START_IDENTITY="${MATR_D15_ROOT}/source_identity_start.json"
FINAL_IDENTITY="${MATR_D15_ROOT}/source_identity_final.json"
SCAN="${MATR_D15_ROOT}/owner_counterfactual.json"
TRACE="${MATR_D15_ROOT}/owner_counterfactual_trace.jsonl.gz"
RECEIPT="${MATR_D15_ROOT}/owner_counterfactual_receipt.json"

for artifact in \
  "${START_IDENTITY}" \
  "${FINAL_IDENTITY}" \
  "${SCAN}" \
  "${TRACE}" \
  "${RECEIPT}"; do
  if [[ -e "${artifact}" ]]; then
    echo "refusing to overwrite append-only D1.5 artifact: ${artifact}" >&2
    exit 2
  fi
done

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${START_IDENTITY}" \
  --expected-commit "${MATR_D15_SOURCE_COMMIT}" \
  --expected-tree "${MATR_D15_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D15_MANIFEST_SHA256}"

python3 scripts/run_eventmatr_d15_owner_counterfactual.py \
  --checkpoint "${MATR_CHECKPOINT}" \
  --options "${MATR_OPTIONS}" \
  --training-source-identity "${MATR_TRAIN_SOURCE_IDENTITY}" \
  --d14-structure-gate "${MATR_D14_STRUCTURE_GATE}" \
  --manifest "${MANIFEST}" \
  --expected-training-source-commit "${MATR_TRAIN_SOURCE_COMMIT}" \
  --expected-training-source-tree "${MATR_TRAIN_SOURCE_TREE}" \
  --expected-d14-source-commit "${MATR_D14_SOURCE_COMMIT}" \
  --expected-d14-source-tree "${MATR_D14_SOURCE_TREE}" \
  --expected-diagnostic-source-commit "${MATR_D15_SOURCE_COMMIT}" \
  --expected-diagnostic-source-tree "${MATR_D15_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D15_MANIFEST_SHA256}" \
  --expected-checkpoint-sha256 "${MATR_CHECKPOINT_SHA256}" \
  --expected-options-sha256 "${MATR_OPTIONS_SHA256}" \
  --output "${SCAN}" \
  --trace-output "${TRACE}"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${FINAL_IDENTITY}" \
  --expected-commit "${MATR_D15_SOURCE_COMMIT}" \
  --expected-tree "${MATR_D15_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D15_MANIFEST_SHA256}"

python3 scripts/finalize_eventmatr_d15_owner_counterfactual.py \
  --scan "${SCAN}" \
  --source-identity-start "${START_IDENTITY}" \
  --source-identity-final "${FINAL_IDENTITY}" \
  --diagnostic-source-commit "${MATR_D15_SOURCE_COMMIT}" \
  --diagnostic-source-tree "${MATR_D15_SOURCE_TREE}" \
  --training-source-commit "${MATR_TRAIN_SOURCE_COMMIT}" \
  --training-source-tree "${MATR_TRAIN_SOURCE_TREE}" \
  --d14-source-commit "${MATR_D14_SOURCE_COMMIT}" \
  --d14-source-tree "${MATR_D14_SOURCE_TREE}" \
  --manifest-sha256 "${MATR_D15_MANIFEST_SHA256}" \
  --checkpoint-sha256 "${MATR_CHECKPOINT_SHA256}" \
  --options-sha256 "${MATR_OPTIONS_SHA256}" \
  --d14-structure-gate-sha256 "${MATR_D14_STRUCTURE_GATE_SHA256}" \
  --output "${RECEIPT}"
