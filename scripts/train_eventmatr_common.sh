#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 ARM --birth_mode MODE --ownership_mode MODE [extra native MATR args...]" >&2
  exit 2
fi

ARM="$1"
shift

if [[ "$1" != "--birth_mode" || "$3" != "--ownership_mode" ]]; then
  echo "the common launcher accepts only --birth_mode then --ownership_mode as arm factors" >&2
  exit 2
fi
BIRTH_MODE="$2"
OWNERSHIP_MODE="$4"
shift 4

if [[ $# -ne 0 ]]; then
  echo "formal EventMATR arms forbid ad-hoc argument overrides" >&2
  exit 2
fi

case "${ARM}:${BIRTH_MODE}:${OWNERSHIP_MODE}" in
  b0o0:matr_delayed:fresh_rematch|\
  b1o0:instant_transition:fresh_rematch|\
  b0o1:matr_delayed:sticky_owner|\
  b1o1:instant_transition:sticky_owner) ;;
  *)
    echo "invalid factorial arm/mode tuple: ${ARM}:${BIRTH_MODE}:${OWNERSHIP_MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_PROTOCOL_ANNO:?MATR_PROTOCOL_ANNO must point to the official training annotation}"
: "${MATR_TRAIN_FEATURE:?MATR_TRAIN_FEATURE must point to the complete official validation/train feature pickle}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"

for PATH_VAR in MATR_OUTPUT_ROOT MATR_PROTOCOL_ANNO MATR_TRAIN_FEATURE \
  MATR_VIDEO_LEN_PATTERN MATR_LABEL_PATTERN; do
  if [[ "${!PATH_VAR}" != /* ]]; then
    echo "${PATH_VAR} must be an absolute path because outputs run in an isolated directory" >&2
    exit 2
  fi
done

export MATR_LANE="${ARM}"
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
  --model_variant eventmatr \
  --event_arm "${ARM}" \
  --birth_mode "${BIRTH_MODE}" \
  --ownership_mode "${OWNERSHIP_MODE}"
