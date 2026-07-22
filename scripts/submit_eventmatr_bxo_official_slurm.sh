#!/usr/bin/env bash
# Slurm DAG: Linux smoke -> native MATR + four eventized 100-epoch lanes -> receipt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MATR_PROTOCOL_ANNO:?official THUMOS14 training annotation is required}"
: "${MATR_TRAIN_FEATURE:?complete official validation/train feature pickle is required}"
: "${MATR_VIDEO_LEN_PATTERN:?video-length pattern is required}"
: "${MATR_LABEL_PATTERN:?label pattern is required}"
: "${MATR_RUN_BASE:?durable Slurm run root is required}"

if [[ -z "${MATR_ENV_ACTIVATE:-}" && -n "${BASE:-}" && \
      -f "${BASE%/}/conda_envs/opentad/bin/activate" ]]; then
  export MATR_ENV_ACTIVATE="${BASE%/}/conda_envs/opentad/bin/activate"
fi
if [[ -z "${MATR_ENV_ACTIVATE:-}" && \
      ( -z "${MATR_CONDA_SH:-}" || -z "${MATR_CONDA_ENV:-}" ) ]]; then
  echo "set MATR_ENV_ACTIVATE, or both MATR_CONDA_SH and MATR_CONDA_ENV" >&2
  exit 2
fi

if [[ -n "${MATR_TEST_FEATURE:-}" || -n "${MATR_TEST_ANNO:-}" ]]; then
  echo "training DAG forbids locked-test inputs" >&2
  exit 2
fi

export MATR_RUN_TAG="${MATR_RUN_TAG:-eventmatr_bxo_official_$(date +%Y%m%d_%H%M%S)}"
export MATR_OUTPUT_ROOT="${MATR_RUN_BASE%/}/${MATR_RUN_TAG}"
mkdir -p "${MATR_OUTPUT_ROOT}/logs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_OUTPUT_ROOT}/pycache"

MATR_MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_bxo_official_thumos14.json"
python3 "${SCRIPT_DIR}/verify_source_identity.py" \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MATR_MANIFEST}" \
  --output "${MATR_OUTPUT_ROOT}/source_identity.json"

export MATR_SOURCE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
export MATR_SOURCE_TREE="$(git -C "${PROJECT_DIR}" rev-parse 'HEAD^{tree}')"
export MATR_MANIFEST_SHA256="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "${MATR_MANIFEST}")"
export MATR_SMOKE_RECEIPT="${MATR_OUTPUT_ROOT}/eventmatr_real_smoke.json"

SBATCH_COMMON=()
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_COMMON+=(--partition "${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_COMMON+=(--account "${SLURM_ACCOUNT}")
fi

SMOKE_JOB=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
  --output "${MATR_OUTPUT_ROOT}/logs/%x.%j.out" \
  --error "${MATR_OUTPUT_ROOT}/logs/%x.%j.err" \
  --export=ALL \
  "${SCRIPT_DIR}/slurm_eventmatr_smoke.sh")

TRAIN_JOBS=()
for LANE in native_matr b0o0 b1o0 b0o1 b1o1; do
  JOB_ID=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency="afterok:${SMOKE_JOB}" \
    --job-name="matr-${LANE}" \
    --output "${MATR_OUTPUT_ROOT}/logs/%x.%j.out" \
    --error "${MATR_OUTPUT_ROOT}/logs/%x.%j.err" \
    --export="ALL,MATR_LANE=${LANE}" \
    "${SCRIPT_DIR}/slurm_eventmatr_train.sh")
  TRAIN_JOBS+=("${JOB_ID}")
done

DEPENDENCY=$(IFS=:; echo "${TRAIN_JOBS[*]}")
FINAL_JOB=$(sbatch --parsable "${SBATCH_COMMON[@]}" \
  --dependency="afterok:${DEPENDENCY}" \
  --output "${MATR_OUTPUT_ROOT}/logs/%x.%j.out" \
  --error "${MATR_OUTPUT_ROOT}/logs/%x.%j.err" \
  --export=ALL \
  "${SCRIPT_DIR}/slurm_eventmatr_finalize.sh")

printf 'run_tag=%s\noutput_root=%s\nsmoke=%s\nlanes=%s\nfinalizer=%s\n' \
  "${MATR_RUN_TAG}" "${MATR_OUTPUT_ROOT}" "${SMOKE_JOB}" \
  "${TRAIN_JOBS[*]}" "${FINAL_JOB}"
