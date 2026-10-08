#!/bin/bash
# warm-retune/apply.sh apply|undo|check
#
# Carries the runtime-retune patch in the llama.cpp that this checkout vendors
# (third_party/llama.cpp @ the revision pinned by setup.py's LLAMA_CPP_COMMIT).
#
#   check  - is the retune present?
#   apply  - apply it (idempotent)
#   undo   - reverse it (idempotent)
#
# The vendored tree is .gitignore'd (setup.py extracts it), so git cannot track or revert this;
# the patch file in this directory is the source of truth and `undo` reverse-applies it.
#
# NOTE: setup.py re-extracts third_party/llama.cpp whenever its revision changes - run `apply`
# again afterwards (setup.py with this fork does that automatically).
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VEND=$ROOT/third_party/llama.cpp
PATCH=$ROOT/warm-retune/retune-llama-server-3cf0325.patch
MARKER=$VEND/tools/server/server-task.h

patched() { grep -q "SERVER_TASK_TYPE_RECONFIGURE" "$MARKER" 2>/dev/null; }

case "${1:-check}" in
  check)
    if patched; then echo "warm-retune: present in $VEND"; else echo "warm-retune: absent from $VEND"; fi ;;
  apply)
    [ -r "$PATCH" ] || { echo "missing patch: $PATCH"; exit 1; }
    [ -d "$VEND" ] || { echo "no vendored tree at $VEND - run setup.py first"; exit 1; }
    if patched; then echo "warm-retune: already applied"; exit 0; fi
    git -C "$ROOT" apply --directory=third_party/llama.cpp "$PATCH" \
      && echo "warm-retune: applied $(basename "$PATCH")" ;;
  undo)
    patched || { echo "warm-retune: not applied"; exit 0; }
    git -C "$ROOT" apply -R --directory=third_party/llama.cpp "$PATCH" \
      && echo "warm-retune: reverted" ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
