#!/usr/bin/env bash

# Source this file before CausalTAD checks, Slurm launch scripts, or training on
# N16R4. It keeps the clean repo code separate from data/checkpoints while
# exposing the Video-Mamba bundled mamba API required by CausalProj.

set -euo pipefail

BASE=${BASE:-/data/run01/sczc063/yuzibo}
ENV_DIR=${ENV_DIR:-$BASE/conda_envs/opentad}
VIDEO_MAMBA_MAMBA=${VIDEO_MAMBA_MAMBA:-$BASE/external_official_action_segmentation_repos_20260702/video-mamba-suite/mamba}

if [ ! -f "$ENV_DIR/bin/activate" ]; then
    echo "Missing conda env: $ENV_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

if [ ! -d "$VIDEO_MAMBA_MAMBA/mamba_ssm" ]; then
    echo "Missing Video-Mamba bundled mamba source: $VIDEO_MAMBA_MAMBA" >&2
    return 1 2>/dev/null || exit 1
fi

source "$ENV_DIR/bin/activate"

export HOME=${HOME:-$BASE/tmp/home}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$BASE/tmp/xdg_cache}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-$BASE/tmp/xdg_config}
export HF_HOME=${HF_HOME:-$BASE/hf_cache}
export PYTHONPATH="$VIDEO_MAMBA_MAMBA:${PYTHONPATH:-}"
