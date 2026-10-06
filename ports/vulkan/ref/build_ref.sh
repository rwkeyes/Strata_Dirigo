#!/usr/bin/env bash
# ports/vulkan/ref/build_ref.sh - build the INDEPENDENT REFERENCE HARNESS (measurement-only).
#
# It links the port's own device layer (harness/vk_compute.cpp + vk_compat.cpp + vk_stack.cpp, the same TUs the
# gate drives) against the llama.cpp build's ggml (`libggml-cpu` for the reference `vec_dot`, `libggml-base` for
# `ggml_get_type_traits`/`to_float`).  It compiles the mmvq shaders it needs from SOURCE, exactly as the gate
# does, so a stale .spv cannot be measured.  It does NOT touch gates/run_gate.sh, the gate's build, or the
# engine's build tree.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # ports/vulkan
SH="$ROOT/shaders"
BUILD="$ROOT/ref/build"
SPV="$BUILD/spv"
LLAMA="${LLAMA_BUILD:-/home/bob/llama-050/build-vulkan}"
GGML_INC="${GGML_INCLUDE:-/home/bob/llama-050/ggml/include}"
mkdir -p "$SPV"

KERNELS=(iq3xxs_mmvq iq2s_mmvq iq3s_mmvq iq4xs_mmvq iq4nl_mmvq q2_0_mmvq native_k_mmvq q8_0_mmvq)
rc=0
echo "== compiling the measured shaders from source -> $SPV"
for k in "${KERNELS[@]}"; do
  if ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$SH/$k.comp" -o "$SPV/$k.spv" 2>"$BUILD/$k.err"; then
    echo "  FAIL glslc $k"; sed -n '1,8p' "$BUILD/$k.err"; rc=1
  else
    printf '  OK   %s\n' "$k"
  fi
done
[ $rc -eq 0 ] || { echo "== shader build FAILED"; exit 1; }

echo "== building ref_vs_ggml"
g++ -std=c++20 -O2 -Wall -Wextra -Werror \
    -I"$ROOT/harness" -I"$GGML_INC" \
    -o "$BUILD/ref_vs_ggml" "$ROOT/ref/ref_vs_ggml.cpp" \
    "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_compat.cpp" "$ROOT/harness/vk_stack.cpp" \
    -L"$LLAMA/bin" -lggml-cpu -lggml-base -lggml \
    -Wl,-rpath,"$LLAMA/bin" -lvulkan -lpthread || { echo "== ref build FAILED"; exit 1; }
echo "== built $BUILD/ref_vs_ggml"
