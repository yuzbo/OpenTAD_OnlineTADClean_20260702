#!/usr/bin/env bash
# Exact native MATR model lane under the same terminal-epoch study protocol.
# It preserves the official architecture/head/loss but never mounts test data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_PROTOCOL_ANNO:?official THUMOS14 training annotation is required}"
: "${MATR_TRAIN_FEATURE:?complete official THUMOS14 validation/train feature pickle is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"

for PATH_VAR in MATR_OUTPUT_ROOT MATR_PROTOCOL_ANNO MATR_TRAIN_FEATURE \
  MATR_VIDEO_LEN_PATTERN MATR_LABEL_PATTERN; do
  if [[ "${!PATH_VAR}" != /* ]]; then
    echo "${PATH_VAR} must be an absolute path" >&2
    exit 2
  fi
done

export MATR_LANE=native_matr
LOCKED_TEST_SENTINEL="${MATR_OUTPUT_ROOT}/LOCKED_TEST_NOT_MOUNTED.pickle"
python3 "${SCRIPT_DIR}/run_eventmatr_main.py" \
  --device "${MATR_DEVICE:-0}" \
  --mode train \
  --video_anno "${MATR_PROTOCOL_ANNO}" \
  --video_feature_all_train "${MATR_TRAIN_FEATURE}" \
  --video_feature_all_test "${LOCKED_TEST_SENTINEL}" \
  --video_len_file "${MATR_VIDEO_LEN_PATTERN}" \
  --ontal_label_file "${MATR_LABEL_PATTERN}" \
  --rgb \
  --flow \
  --feat_dim 4096 \
  --num_frame 64 \
  --num_queries 10 \
  --max_memory_len 7 \
  --memory_sampler gap2 \
  --batch 64 \
  --epochs 100 \
  --min_lr 1e-8 \
  --max_lr 1e-5 \
  --weight_decay 1e-4 \
  --lr_Tup 3 \
  --lr_Tcycle 10 \
  --lr_gamma 0.9 \
  --random_seed 52 \
  --use_focal \
  --use_flag \
  --flag_threshold 0.5 \
  --cls_threshold 0.1 \
  --nms_threshold 0.3 \
  --reduce 1 \
  --make_output \
  --study_protocol matched_study \
  --model_variant native_matr
