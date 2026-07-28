#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-v1-d0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=54G
#SBATCH --time=04:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_D0_OUTPUT_ROOT:?MATR_D0_OUTPUT_ROOT is required}"
: "${MATR_D0_SOURCE_RUN:?MATR_D0_SOURCE_RUN is required}"
: "${MATR_D0_SOURCE_COMMIT:?MATR_D0_SOURCE_COMMIT is required}"
: "${MATR_D0_SOURCE_TREE:?MATR_D0_SOURCE_TREE is required}"
: "${MATR_D0_MANIFEST_SHA256:?MATR_D0_MANIFEST_SHA256 is required}"
: "${MATR_D0_TRAINING_COMMIT:?MATR_D0_TRAINING_COMMIT is required}"
: "${MATR_D0_TRAINING_TREE:?MATR_D0_TRAINING_TREE is required}"

if [[ -n "${MATR_TEST_FEATURE:-}" || -n "${MATR_TEST_ANNO:-}" ]]; then
  echo "D0 audit forbids locked-test inputs" >&2
  exit 2
fi

LANES=(b0o0 b1o0 b0o1 b1o1)
if [[ -n "${MATR_D0_LANE:-}" ]]; then
  LANE="${MATR_D0_LANE}"
else
  : "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
  LANE="${LANES[SLURM_ARRAY_TASK_ID]}"
fi
case "${LANE}" in
  b0o0|b1o0|b0o1|b1o1) ;;
  *) echo "invalid D0 lane: ${LANE}" >&2; exit 2 ;;
esac

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_D0_OUTPUT_ROOT}/pycache/${LANE}"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_v1_d0_audit.json"
IDENTITY_DIR="${MATR_D0_OUTPUT_ROOT}/source_identity_lanes"
LANE_OUTPUT="${MATR_D0_OUTPUT_ROOT}/${LANE}"
mkdir -p "${IDENTITY_DIR}" "${LANE_OUTPUT}"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${IDENTITY_DIR}/${LANE}.json" \
  --expected-commit "${MATR_D0_SOURCE_COMMIT}" \
  --expected-tree "${MATR_D0_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D0_MANIFEST_SHA256}"

python3 scripts/run_eventmatr_v1_d0_audit.py \
  --source-run "${MATR_D0_SOURCE_RUN}" \
  --lane "${LANE}" \
  --output-dir "${LANE_OUTPUT}" \
  --expected-training-commit "${MATR_D0_TRAINING_COMMIT}" \
  --expected-training-tree "${MATR_D0_TRAINING_TREE}" \
  --diagnostic-commit "${MATR_D0_SOURCE_COMMIT}" \
  --diagnostic-tree "${MATR_D0_SOURCE_TREE}" \
  --device cuda
