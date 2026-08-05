#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-d16-pair-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_MECHANISM_ROOT:?MATR_MECHANISM_ROOT is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"
python3 scripts/finalize_eventmatr_d16_pair.py \
  --pair-root "${MATR_MECHANISM_ROOT}"
