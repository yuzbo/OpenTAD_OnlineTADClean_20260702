#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_LANE:?MATR_LANE is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"
: "${MATR_SMOKE_RECEIPT:?MATR_SMOKE_RECEIPT is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_OUTPUT_ROOT}/pycache/${MATR_LANE}"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_bxo_official_thumos14.json"
IDENTITY_DIR="${MATR_OUTPUT_ROOT}/source_identity_lanes"
mkdir -p "${IDENTITY_DIR}"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${IDENTITY_DIR}/${MATR_LANE}.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_SMOKE_RECEIPT}"

export MATR_DEVICE=0
if [[ "${MATR_LANE}" == "native_matr" ]]; then
  exec bash scripts/train_native_matr.sh
fi
case "${MATR_LANE}" in
  b0o0|b1o0|b0o1|b1o1) exec bash "scripts/train_${MATR_LANE}.sh" ;;
  *) echo "invalid MATR_LANE=${MATR_LANE}" >&2; exit 2 ;;
esac
