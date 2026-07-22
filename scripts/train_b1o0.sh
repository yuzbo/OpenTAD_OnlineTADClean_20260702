#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_eventmatr_common.sh" b1o0 \
  --birth_mode instant_transition \
  --ownership_mode fresh_rematch \
  "$@"
