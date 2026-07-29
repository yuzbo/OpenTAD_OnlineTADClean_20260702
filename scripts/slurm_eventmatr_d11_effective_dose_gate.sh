#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d11-dose-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_GATE_ROOT:?MATR_GATE_ROOT is required}"
: "${MATR_MECHANISM_RECEIPT:?MATR_MECHANISM_RECEIPT is required}"
: "${MATR_ASSOCIATION_SCAN:?MATR_ASSOCIATION_SCAN is required}"
: "${MATR_PARAMETER_DELTA_AUDIT:?MATR_PARAMETER_DELTA_AUDIT is required}"
: "${MATR_CHECKPOINT_SHA256:?MATR_CHECKPOINT_SHA256 is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"

mkdir -p "${MATR_GATE_ROOT}"
source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_GATE_ROOT}/pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_GATE_ROOT}/gate_source_identity_start.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}"

python3 scripts/finalize_eventmatr_d11_effective_dose_gate.py \
  --mechanism-receipt "${MATR_MECHANISM_RECEIPT}" \
  --association-scan "${MATR_ASSOCIATION_SCAN}" \
  --parameter-delta-audit "${MATR_PARAMETER_DELTA_AUDIT}" \
  --expected-checkpoint-sha256 "${MATR_CHECKPOINT_SHA256}" \
  --training-source-commit "${MATR_SOURCE_COMMIT}" \
  --training-source-tree "${MATR_SOURCE_TREE}" \
  --output "${MATR_GATE_ROOT}/effective_dose_gate_receipt.json"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_GATE_ROOT}/gate_source_identity_final.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}"
