#!/usr/bin/env bash
# Source this file from Slurm workers. Prefer the server's direct environment
# activation path; retain a conda.sh + environment-name fallback.

if [[ -n "${MATR_ENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "${MATR_ENV_ACTIVATE}" ]]; then
    echo "MATR_ENV_ACTIVATE does not exist: ${MATR_ENV_ACTIVATE}" >&2
    return 2 2>/dev/null || exit 2
  fi
  source "${MATR_ENV_ACTIVATE}"
elif [[ -n "${MATR_CONDA_SH:-}" && -n "${MATR_CONDA_ENV:-}" ]]; then
  if [[ ! -f "${MATR_CONDA_SH}" ]]; then
    echo "MATR_CONDA_SH does not exist: ${MATR_CONDA_SH}" >&2
    return 2 2>/dev/null || exit 2
  fi
  source "${MATR_CONDA_SH}"
  conda activate "${MATR_CONDA_ENV}"
else
  echo "set MATR_ENV_ACTIVATE, or both MATR_CONDA_SH and MATR_CONDA_ENV" >&2
  return 2 2>/dev/null || exit 2
fi
