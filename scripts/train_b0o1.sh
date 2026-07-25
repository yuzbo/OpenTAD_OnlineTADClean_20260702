#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_eventmatr_common.sh" b0o1 \
  --birth_mode matr_delayed \
  --ownership_mode sticky_owner \
  "$@"
