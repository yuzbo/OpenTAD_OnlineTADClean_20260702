#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d151-margin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_D15_ROOT:?MATR_D15_ROOT is required}"
: "${MATR_D151_ROOT:?MATR_D151_ROOT is required}"
: "${MATR_D15_RECEIPT_SHA256:?MATR_D15_RECEIPT_SHA256 is required}"
: "${MATR_D15_SCAN_SHA256:?MATR_D15_SCAN_SHA256 is required}"
: "${MATR_D15_TRACE_SHA256:?MATR_D15_TRACE_SHA256 is required}"
: "${MATR_D15_SOURCE_COMMIT:?MATR_D15_SOURCE_COMMIT is required}"
: "${MATR_D15_SOURCE_TREE:?MATR_D15_SOURCE_TREE is required}"
: "${MATR_D151_SOURCE_COMMIT:?MATR_D151_SOURCE_COMMIT is required}"
: "${MATR_D151_SOURCE_TREE:?MATR_D151_SOURCE_TREE is required}"
: "${MATR_D151_MANIFEST_SHA256:?MATR_D151_MANIFEST_SHA256 is required}"

mkdir -p "${MATR_D151_ROOT}"
source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_D151_ROOT}/pycache"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_d1_preexperiments.json"
RECEIPT="${MATR_D15_ROOT}/owner_counterfactual_receipt.json"
SCAN="${MATR_D15_ROOT}/owner_counterfactual.json"
TRACE="${MATR_D15_ROOT}/owner_counterfactual_trace.jsonl.gz"
START_IDENTITY="${MATR_D151_ROOT}/source_identity_start.json"
FINAL_IDENTITY="${MATR_D151_ROOT}/source_identity_final.json"
OUTPUT="${MATR_D151_ROOT}/endpoint_margin_receipt.json"
OUTPUT_SHA="${MATR_D151_ROOT}/endpoint_margin_receipt.sha256"

for artifact in "${START_IDENTITY}" "${FINAL_IDENTITY}" "${OUTPUT}" "${OUTPUT_SHA}"; do
  if [[ -e "${artifact}" ]]; then
    echo "refusing to overwrite append-only D1.5.1 artifact: ${artifact}" >&2
    exit 2
  fi
done

printf '%s  %s\n' "${MATR_D15_RECEIPT_SHA256}" "${RECEIPT}" | sha256sum --check --strict
printf '%s  %s\n' "${MATR_D15_SCAN_SHA256}" "${SCAN}" | sha256sum --check --strict
printf '%s  %s\n' "${MATR_D15_TRACE_SHA256}" "${TRACE}" | sha256sum --check --strict

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${START_IDENTITY}" \
  --expected-commit "${MATR_D151_SOURCE_COMMIT}" \
  --expected-tree "${MATR_D151_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D151_MANIFEST_SHA256}"

python3 -m scripts.analyze_eventmatr_d151_endpoint_trace \
  --project-dir "${PROJECT_DIR}" \
  --receipt "${RECEIPT}" \
  --scan "${SCAN}" \
  --trace "${TRACE}" \
  --expected-receipt-sha256 "${MATR_D15_RECEIPT_SHA256}" \
  --expected-scan-sha256 "${MATR_D15_SCAN_SHA256}" \
  --expected-trace-sha256 "${MATR_D15_TRACE_SHA256}" \
  --expected-d15-source-commit "${MATR_D15_SOURCE_COMMIT}" \
  --expected-d15-source-tree "${MATR_D15_SOURCE_TREE}" \
  --expected-analyzer-source-commit "${MATR_D151_SOURCE_COMMIT}" \
  --expected-analyzer-source-tree "${MATR_D151_SOURCE_TREE}" \
  --output "${OUTPUT}"

python3 scripts/verify_source_identity.py \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${FINAL_IDENTITY}" \
  --expected-commit "${MATR_D151_SOURCE_COMMIT}" \
  --expected-tree "${MATR_D151_SOURCE_TREE}" \
  --expected-manifest-sha256 "${MATR_D151_MANIFEST_SHA256}"

sha256sum "${OUTPUT}" > "${OUTPUT_SHA}"
