#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_PROTOCOL_ANNO:?official THUMOS14 training annotation is required}"
: "${MATR_TRAIN_FEATURE:?complete official validation/train feature pickle is required}"
: "${MATR_VIDEO_LEN_PATTERN:?video-length pattern is required}"
: "${MATR_LABEL_PATTERN:?label pattern is required}"
: "${MATR_SOURCE_COMMIT:?MATR_SOURCE_COMMIT is required}"
: "${MATR_SOURCE_TREE:?MATR_SOURCE_TREE is required}"
: "${MATR_MANIFEST_SHA256:?MATR_MANIFEST_SHA256 is required}"
: "${MATR_SMOKE_RECEIPT:?MATR_SMOKE_RECEIPT is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_OUTPUT_ROOT}/pycache/smoke"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_bxo_official_thumos14.json"
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_OUTPUT_ROOT}/source_identity_smoke_start.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}"

python3 scripts/verify_official_protocol.py
python3 -m pytest -q -p no:cacheprovider tests
python3 scripts/run_eventmatr_real_smoke.py \
  --video-anno "${MATR_PROTOCOL_ANNO}" \
  --train-feature "${MATR_TRAIN_FEATURE}" \
  --video-len-pattern "${MATR_VIDEO_LEN_PATTERN}" \
  --label-pattern "${MATR_LABEL_PATTERN}" \
  --output "${MATR_SMOKE_RECEIPT}"

# Only an exact-source PASS receipt permits the dependent five-lane release.
python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_OUTPUT_ROOT}/source_identity_smoke.json" \
  --expected-commit "${MATR_SOURCE_COMMIT}" \
  --expected-tree "${MATR_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_MANIFEST_SHA256}" \
  --smoke-receipt "${MATR_SMOKE_RECEIPT}"
