#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-v1-d0-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_D0_OUTPUT_ROOT:?MATR_D0_OUTPUT_ROOT is required}"
: "${MATR_D0_SOURCE_RUN:?MATR_D0_SOURCE_RUN is required}"
: "${MATR_D0_SOURCE_COMMIT:?MATR_D0_SOURCE_COMMIT is required}"
: "${MATR_D0_SOURCE_TREE:?MATR_D0_SOURCE_TREE is required}"
: "${MATR_D0_MANIFEST_SHA256:?MATR_D0_MANIFEST_SHA256 is required}"
: "${MATR_D0_TRAINING_COMMIT:?MATR_D0_TRAINING_COMMIT is required}"
: "${MATR_D0_TRAINING_TREE:?MATR_D0_TRAINING_TREE is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
python3 scripts/finalize_eventmatr_v1_d0.py \
  --root "${MATR_D0_OUTPUT_ROOT}" \
  --manifest "${PROJECT_DIR}/experiment_configs/eventmatr_v1_d0_audit.json" \
  --expected-source-run "${MATR_D0_SOURCE_RUN}" \
  --expected-diagnostic-commit "${MATR_D0_SOURCE_COMMIT}" \
  --expected-diagnostic-tree "${MATR_D0_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D0_MANIFEST_SHA256}" \
  --expected-training-commit "${MATR_D0_TRAINING_COMMIT}" \
  --expected-training-tree "${MATR_D0_TRAINING_TREE}"
