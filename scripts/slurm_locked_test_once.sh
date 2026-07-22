#!/usr/bin/env bash
#SBATCH --job-name=matr-locked-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${MATR_LANE:?MATR_LANE is required}"

source "${PROJECT_DIR}/scripts/activate_matr_env.sh"
cd "${PROJECT_DIR}"

export MATR_DEVICE=0
exec bash scripts/eval_locked_test_once.sh "${MATR_LANE}"
