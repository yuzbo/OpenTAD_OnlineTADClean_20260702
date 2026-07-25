#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_eventmatr_common.sh" b1o1 \
  --birth_mode instant_transition \
  --ownership_mode sticky_owner \
  "$@"
