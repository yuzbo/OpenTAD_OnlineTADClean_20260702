#!/usr/bin/env bash
#SBATCH --job-name=eventmatr-cofie
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

SOURCE_ROOT="${EVENTMATR_SOURCE_ROOT:?set EVENTMATR_SOURCE_ROOT to a clean checkout}"
RUN_ROOT="${EVENTMATR_RUN_ROOT:?set EVENTMATR_RUN_ROOT to an empty result root}"
PYTHON="${EVENTMATR_PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"

arms=(matr matr matr cofie cofie cofie)
seeds=(52 3407 5801 52 3407 5801)
configs=(
  configs/causaltad/thumos_i3d_matr_n16r4.py
  configs/causaltad/thumos_i3d_matr_n16r4.py
  configs/causaltad/thumos_i3d_matr_n16r4.py
  configs/causaltad/thumos_i3d_cofie_n16r4.py
  configs/causaltad/thumos_i3d_cofie_n16r4.py
  configs/causaltad/thumos_i3d_cofie_n16r4.py
)

task_id="${SLURM_ARRAY_TASK_ID}"
arm="${arms[$task_id]}"
seed="${seeds[$task_id]}"
config="${configs[$task_id]}"
work_dir="${RUN_ROOT}/${arm}_seed${seed}"

test -x "${PYTHON}"
test -f "${SOURCE_ROOT}/${config}"
mkdir -p "${RUN_ROOT}/slurm"
cd "${SOURCE_ROOT}"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="$((29500 + task_id))" \
  tools/train.py "${config}" \
  --seed "${seed}" \
  --id 0 \
  --cfg-options "work_dir=${work_dir}"
