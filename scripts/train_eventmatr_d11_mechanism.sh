#!/usr/bin/env bash
set -euo pipefail

: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_PROTOCOL_ANNO:?MATR_PROTOCOL_ANNO is required}"
: "${MATR_TRAIN_FEATURE:?MATR_TRAIN_FEATURE is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"

if [[ "${MATR_LANE:-}" != "TH" ]]; then
  echo "D1.1 mechanism run is restricted to MATR_LANE=TH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCKED_TEST_SENTINEL="${MATR_OUTPUT_ROOT}/LOCKED_TEST_NOT_MOUNTED.pickle"
ARGS=(
  --device "${MATR_DEVICE:-0}"
  --mode train
  --video_anno "${MATR_PROTOCOL_ANNO}"
  --video_feature_all_train "${MATR_TRAIN_FEATURE}"
  --video_feature_all_test "${LOCKED_TEST_SENTINEL}"
  --video_len_file "${MATR_VIDEO_LEN_PATTERN}"
  --ontal_label_file "${MATR_LABEL_PATTERN}"
  --rgb
  --flow
  --feat_dim 4096
  --num_frame 64
  --num_queries 10
  --max_memory_len 7
  --memory_sampler gap2
  --batch 64
  --epochs 1
  --train_eval_step 1
  --min_lr 1e-8
  --max_lr 1e-5
  --weight_decay 1e-4
  --lr_Tup 3
  --lr_Tcycle 10
  --lr_gamma 0.9
  --random_seed 52
  --use_focal
  --use_flag
  --flag_threshold 0.5
  --cls_threshold 0.1
  --nms_threshold 0.3
  --reduce 1
  --make_output
  --study_protocol d11_mechanism
  --model_variant eventmatr
  --event_lifecycle_version d1_censored
  --event_d1_lane th
  --event_teacher_forcing_ratio 0.5
  --d11_effective_dose
  --birth_mode instant_transition
  --ownership_mode sticky_owner
  --event_arm b1o1
)

exec python3 "${SCRIPT_DIR}/run_eventmatr_main.py" "${ARGS[@]}"
