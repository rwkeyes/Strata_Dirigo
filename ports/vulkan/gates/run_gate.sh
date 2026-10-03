#!/usr/bin/env bash
# ports/vulkan/gates/run_gate.sh - compile every shader from SOURCE, validate the SPIR-V, cross-check the
# declared LocalSize against what the host assumes, then run the numeric gate.
#
# Rules this script exists to enforce (each one cost real time on an earlier port):
#   * A gate compiles the shader from .comp on EVERY run.  A gate that reads a .spv built earlier tests a
#     stale binary and reports a confident wrong verdict.
#   * Stray .spv files are deleted first so nothing can pick one up.
#   * The local_size_x the host divides by is ASSERTED against the shader's own OpExecutionMode, because a
#     silent mismatch there is a wrong grid, not a compile error.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SH="$ROOT/shaders"
BUILD="$ROOT/harness/build"
mkdir -p "$BUILD"

echo "== compiling shaders from source ($SH)"
rm -f "$SH"/*.spv
rc=0
for f in "$SH"/*.comp; do
  name="$(basename "$f" .comp)"
  if glslc --target-env=vulkan1.3 -fshader-stage=compute "$f" -o "$SH/$name.spv" 2>"$BUILD/$name.glslerr"; then
    if spirv-val --target-env vulkan1.3 "$SH/$name.spv" 2>>"$BUILD/$name.glslerr"; then
      ls="$(spirv-dis "$SH/$name.spv" | grep -oE 'OpExecutionMode %main LocalSize [0-9]+ [0-9]+ [0-9]+' | head -1)"
      printf '  OK   %-22s %s\n' "$name" "$ls"
      # The host hardcodes local_size_x = 256 for every kernel in this slice.
      if ! grep -q 'LocalSize 256 1 1' <<<"$ls"; then
        printf '  FAIL %-22s local_size_x is not 256 - the host grid math would be wrong\n' "$name"
        rc=1
      fi
    else
      printf '  FAIL %-22s spirv-val\n' "$name"; sed -n '1,12p' "$BUILD/$name.glslerr"; rc=1
    fi
  else
    printf '  FAIL %-22s glslc\n' "$name"; sed -n '1,12p' "$BUILD/$name.glslerr"; rc=1
  fi
done
[ $rc -eq 0 ] || { echo "== shader gate FAILED"; exit 1; }

echo "== building the harness"
g++ -std=c++20 -O2 -Wall -Wextra -o "$BUILD/vk_gate" "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_gate.cpp" -lvulkan || exit 1

echo "== numeric gate"
"$BUILD/vk_gate" --spv-dir "$SH" "$@"
