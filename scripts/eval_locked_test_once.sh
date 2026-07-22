#!/usr/bin/env bash
# One-shot THUMOS14 test evaluation, separated from all training and selection.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 LANE" >&2
  exit 2
fi
LANE="$1"

case "${LANE}" in
  native_matr)
    MODEL_ARGS=(--model_variant native_matr)
    ;;
  b0o0)
    MODEL_ARGS=(--model_variant eventmatr --event_arm b0o0 --birth_mode matr_delayed --ownership_mode fresh_rematch)
    ;;
  b1o0)
    MODEL_ARGS=(--model_variant eventmatr --event_arm b1o0 --birth_mode instant_transition --ownership_mode fresh_rematch)
    ;;
  b0o1)
    MODEL_ARGS=(--model_variant eventmatr --event_arm b0o1 --birth_mode matr_delayed --ownership_mode sticky_owner)
    ;;
  b1o1)
    MODEL_ARGS=(--model_variant eventmatr --event_arm b1o1 --birth_mode instant_transition --ownership_mode sticky_owner)
    ;;
  *) echo "invalid lane: ${LANE}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MATR_LOCKED_TEST_ACK:?set MATR_LOCKED_TEST_ACK=1 for the final one-shot evaluation}"
if [[ "${MATR_LOCKED_TEST_ACK}" != "1" ]]; then
  echo "locked-test acknowledgement must equal 1" >&2
  exit 2
fi
: "${MATR_OUTPUT_ROOT:?MATR_OUTPUT_ROOT is required}"
: "${MATR_RUN_TAG:?MATR_RUN_TAG is required}"
: "${MATR_MODEL_PATH:?terminal checkpoint is required}"
: "${MATR_TEST_FEATURE:?locked THUMOS14 test feature pickle is required}"
: "${MATR_TEST_ANNO:?locked THUMOS14 test annotation is required}"
: "${MATR_VIDEO_LEN_PATTERN:?MATR_VIDEO_LEN_PATTERN is required}"
: "${MATR_LABEL_PATTERN:?MATR_LABEL_PATTERN is required}"

for PATH_VAR in MATR_OUTPUT_ROOT MATR_MODEL_PATH MATR_TEST_FEATURE \
  MATR_TEST_ANNO MATR_VIDEO_LEN_PATTERN MATR_LABEL_PATTERN; do
  if [[ "${!PATH_VAR}" != /* ]]; then
    echo "${PATH_VAR} must be an absolute path" >&2
    exit 2
  fi
done

COMPLETION="${MATR_OUTPUT_ROOT}/eventmatr_bxo_completion.json"
if [[ ! -f "${COMPLETION}" ]]; then
  echo "matched-study completion receipt is required before locked test" >&2
  exit 2
fi
EXPECTED_MODEL=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["lanes"][sys.argv[2]]["checkpoint"])' \
  "${COMPLETION}" "${LANE}")
if [[ "$(readlink -f "${MATR_MODEL_PATH}")" != "$(readlink -f "${EXPECTED_MODEL}")" ]]; then
  echo "MATR_MODEL_PATH does not match the terminal checkpoint receipt for ${LANE}" >&2
  exit 2
fi

LOCK_DIR="${MATR_OUTPUT_ROOT}/locked_test_consumed/${LANE}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "locked THUMOS14 test has already been consumed or reserved for ${LANE}" >&2
  exit 3
fi
printf 'reserved_before_eval=true\nlane=%s\ncheckpoint=%s\n' \
  "${LANE}" "${MATR_MODEL_PATH}" > "${LOCK_DIR}/receipt.txt"

TRAIN_RUN_TAG="${MATR_RUN_TAG}"
export MATR_RUN_TAG="${TRAIN_RUN_TAG}_locked_test"
export MATR_LANE="${LANE}"
python3 "${SCRIPT_DIR}/run_eventmatr_main.py" \
  --device "${MATR_DEVICE:-0}" \
  --mode eval \
  --video_anno "${MATR_TEST_ANNO}" \
  --video_feature_all_test "${MATR_TEST_FEATURE}" \
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
  --load_model \
  --model_path "${MATR_MODEL_PATH}" \
  --study_protocol locked_test \
  "${MODEL_ARGS[@]}"

printf 'completed=true\n' >> "${LOCK_DIR}/receipt.txt"
