#!/usr/bin/env bash
set -euo pipefail

: "${MATR_LANE:?MATR_LANE is required}"
: "${MATR_PILOT_EPOCHS:?MATR_PILOT_EPOCHS is required}"
: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_PROTOCOL_ANNO:?MATR_PROTOCOL_ANNO is required}"
: "${MATR_TRAIN_FEATURE:?MATR_TRAIN_FEATURE is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"

case "${MATR_LANE}" in
  N|R|T|H|TH) ;;
  *) echo "invalid D1 pilot lane: ${MATR_LANE}" >&2; exit 2 ;;
esac
case "${MATR_PILOT_EPOCHS}" in
  5|10|20) ;;
  *) echo "invalid D1 pilot horizon: ${MATR_PILOT_EPOCHS}" >&2; exit 2 ;;
esac

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
  --epochs "${MATR_PILOT_EPOCHS}"
  --train_eval_step "${MATR_PILOT_EPOCHS}"
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
  --study_protocol d1_preexperiment
)

if [[ "${MATR_LANE}" == "N" ]]; then
  ARGS+=(--model_variant native_matr --event_lifecycle_version v1_dense)
else
  D1_LANE="$(printf '%s' "${MATR_LANE}" | tr '[:upper:]' '[:lower:]')"
  OWNERSHIP_MODE=fresh_rematch
  EVENT_ARM=b1o0
  if [[ "${MATR_LANE}" == "T" || "${MATR_LANE}" == "TH" ]]; then
    OWNERSHIP_MODE=sticky_owner
    EVENT_ARM=b1o1
  fi
  ARGS+=(
    --model_variant eventmatr
    --event_lifecycle_version d1_censored
    --event_d1_lane "${D1_LANE}"
    --event_teacher_forcing_ratio 0.5
    --birth_mode instant_transition
    --ownership_mode "${OWNERSHIP_MODE}"
    --event_arm "${EVENT_ARM}"
  )
fi

exec python3 "${SCRIPT_DIR}/run_eventmatr_main.py" "${ARGS[@]}"
