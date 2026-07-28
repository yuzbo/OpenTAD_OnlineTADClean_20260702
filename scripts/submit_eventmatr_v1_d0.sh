#!/usr/bin/env bash
# Submit four checkpoint-only D0 audits in parallel, then summarize them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MATR_D0_SOURCE_RUN:?MATR_D0_SOURCE_RUN is required}"
: "${MATR_D0_RUN_BASE:?MATR_D0_RUN_BASE is required}"
: "${MATR_D0_TRAINING_COMMIT:?MATR_D0_TRAINING_COMMIT is required}"

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
  echo "D0 audit forbids locked-test inputs" >&2
  exit 2
fi
if [[ ! -f "${MATR_D0_SOURCE_RUN}/source_identity.json" ]]; then
  echo "invalid source run: ${MATR_D0_SOURCE_RUN}" >&2
  exit 2
fi

export MATR_D0_RUN_TAG="${MATR_D0_RUN_TAG:-eventmatr_v1_d0_$(date +%Y%m%d_%H%M%S)}"
export MATR_D0_OUTPUT_ROOT="${MATR_D0_RUN_BASE%/}/${MATR_D0_RUN_TAG}"
if [[ -e "${MATR_D0_OUTPUT_ROOT}" ]]; then
  echo "D0 output root already exists: ${MATR_D0_OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${MATR_D0_OUTPUT_ROOT}/logs"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${MATR_D0_OUTPUT_ROOT}/pycache/submit"

MANIFEST="${PROJECT_DIR}/experiment_configs/eventmatr_v1_d0_audit.json"
MANIFEST_TRAINING_COMMIT="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_training_commit"])' \
    "${MANIFEST}"
)"
export MATR_D0_TRAINING_TREE="$(
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_training_tree"])' \
    "${MANIFEST}"
)"
if [[ "${MATR_D0_TRAINING_COMMIT}" != "${MANIFEST_TRAINING_COMMIT}" ]]; then
  echo "training commit does not match the frozen D0 manifest" >&2
  exit 2
fi
python3 "${SCRIPT_DIR}/verify_source_identity.py" \
  --project-dir "${PROJECT_DIR}" \
  --manifest "${MANIFEST}" \
  --output "${MATR_D0_OUTPUT_ROOT}/source_identity.json"

export MATR_D0_SOURCE_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
export MATR_D0_SOURCE_TREE="$(git -C "${PROJECT_DIR}" rev-parse 'HEAD^{tree}')"
export MATR_D0_MANIFEST_SHA256="$(
  python3 -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "${MANIFEST}"
)"

SBATCH_COMMON=()
if [[ -n "${SLURM_PARTITION:-}" ]]; then
  SBATCH_COMMON+=(--partition "${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_COMMON+=(--account "${SLURM_ACCOUNT}")
fi

ARRAY_JOB="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --array=0-3%4 \
    --output "${MATR_D0_OUTPUT_ROOT}/logs/%x.%A_%a.out" \
    --error "${MATR_D0_OUTPUT_ROOT}/logs/%x.%A_%a.err" \
    --export=ALL \
    "${SCRIPT_DIR}/slurm_eventmatr_v1_d0.sh"
)"
FINAL_JOB="$(
  sbatch --parsable "${SBATCH_COMMON[@]}" \
    --dependency="afterok:${ARRAY_JOB}" \
    --output "${MATR_D0_OUTPUT_ROOT}/logs/%x.%j.out" \
    --error "${MATR_D0_OUTPUT_ROOT}/logs/%x.%j.err" \
    --export=ALL \
    "${SCRIPT_DIR}/slurm_eventmatr_v1_d0_finalize.sh"
)"

python3 - "${MATR_D0_OUTPUT_ROOT}" "${ARRAY_JOB}" "${FINAL_JOB}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
payload = {
    "status": "SUBMITTED",
    "protocol_id": "eventmatr_v1_d0_checkpoint_replay_v1",
    "output_root": str(root),
    "array_job": sys.argv[2],
    "array_lanes": ["b0o0", "b1o0", "b0o1", "b1o1"],
    "finalizer_job": sys.argv[3],
    "test_access": False,
    "diagnostic_source": {
        "commit": __import__("os").environ["MATR_D0_SOURCE_COMMIT"],
        "tree": __import__("os").environ["MATR_D0_SOURCE_TREE"],
        "manifest_sha256": __import__("os").environ["MATR_D0_MANIFEST_SHA256"],
    },
    "source_training": {
        "run": __import__("os").environ["MATR_D0_SOURCE_RUN"],
        "commit": __import__("os").environ["MATR_D0_TRAINING_COMMIT"],
        "tree": __import__("os").environ["MATR_D0_TRAINING_TREE"],
    },
}
(root / "d0_launch.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
PY
