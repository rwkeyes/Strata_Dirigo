#!/usr/bin/env bash
# ports/vulkan/bench/run_bench.sh - build and run the port-side throughput harness.
#
# It does NOT touch gates/run_gate.sh, harness/build, or the gate's own .spv files: it compiles the shaders it
# needs from SOURCE into bench/build/spv/ and builds the binary into bench/build/.  So it can be run while a
# verification gate is running on either box.
#
# What it does, in order:
#   1. compile the measured kernels from .comp into bench/build/spv/ (glslc, same flags as the gate) and
#      validate each with spirv-val;
#   2. build vk_bench (-O2 -Werror) against the port's own device layer;
#   3. run it once under EVERY Vulkan ICD that reports a device - a correctness gate runs every implementation
#      because a kernel can be right on one and wrong on another; a THROUGHPUT harness runs every
#      implementation because a kernel number means nothing without the device it was taken on, and the
#      cross-ICD gap is also the harness's own proof that it measures something real;
#   4. print a side-by-side comparison of the kernels both ICDs measured.
#
# Usage: bash ports/vulkan/bench/run_bench.sh [extra vk_bench args...]
#   env (as the gate uses them): STRATA_VK_DESKTOP_RESERVE_MIB / STRATA_VK_RESERVE_FLOOR_MIB / STRATA_VK_MAX_BUDGET_MIB
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # ports/vulkan
SH="$ROOT/shaders"
BUILD="$ROOT/bench/build"
SPV="$BUILD/spv"
mkdir -p "$SPV"

KERNELS=(gdn_conv_step gdn_l2_norm gdn_beta_gate gdn_gate gdn_step gdn_out_norm scale add
         iq_dequant_f32 iq1m_mmvq iq2s_mmvq sampler_kernel_f32 sampler_split
         quantize_q8_0 quantize_q8_1 quantize_q8_K
         rope_neox native_rope_apply router_top10_f32 native_router_top10
         moe_combine_f32 native_moe_combine rms_norm native_qsa_rms_norm_weighted
         qsa_gate_apply_f32 native_qsa_gate_apply
         silu_f32 native_gdn_conv_silu native_gdn_l2_norm native_gdn_beta_gate
         native_gdn_gate native_gdn_out_norm native_gdn_step bf16_mmvf_f32
         fused_gdn_conv_l2 fused_gdn_ab fused_gdn_step_norm bf16_gemv qsa_decode_attn s_gemv_q8_split
         gemm_prefill_f16_m8 gemm_prefill_f16_m8_staged gemm_prefill_fma gemm_prefill_fma_small
         native_gu_any native_down_any fused_gr_rs fused_gr_down fused_gr_mix fused_gr_inject
         native_k_mmvq swiglu_f32)

rc=0
echo "== compiling the measured kernels from source -> $SPV"
for k in "${KERNELS[@]}"; do
  if ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$SH/$k.comp" -o "$SPV/$k.spv" 2>"$BUILD/$k.err"; then
    echo "  FAIL glslc $k"; sed -n '1,8p' "$BUILD/$k.err"; rc=1; continue
  fi
  if ! spirv-val --target-env vulkan1.3 "$SPV/$k.spv" 2>>"$BUILD/$k.err"; then
    echo "  FAIL spirv-val $k"; sed -n '1,8p' "$BUILD/$k.err"; rc=1; continue
  fi
  printf '  OK   %s\n' "$k"
done
# THE UNROLL A/B VARIANTS: one source, three builds.  `gdn_step_unroll` compares the rolled order (KU=1) and a
# KU=8 variant against the SHIPPED KU, so the arm never needs a saved .spv and cannot drift from the source it
# claims to measure.  A failure here is a NAMED failure, not a missing row.
for ku in 1 8; do
  if ! glslc --target-env=vulkan1.3 -fshader-stage=compute -DKU=$ku "$SH/native_gdn_step.comp" \
        -o "$SPV/native_gdn_step_u$ku.spv" 2>"$BUILD/native_gdn_step_u$ku.err"; then
    echo "  FAIL glslc native_gdn_step -DKU=$ku"; sed -n '1,8p' "$BUILD/native_gdn_step_u$ku.err"; rc=1; continue
  fi
  if ! spirv-val --target-env vulkan1.3 "$SPV/native_gdn_step_u$ku.spv" 2>>"$BUILD/native_gdn_step_u$ku.err"; then
    echo "  FAIL spirv-val native_gdn_step_u$ku"; rc=1; continue
  fi
  printf '  OK   %s\n' "native_gdn_step_u$ku (-DKU=$ku)"
done
[ $rc -eq 0 ] || { echo "== shader build FAILED"; exit 1; }

# THE BENCH-LOCAL SHADERS.  The scatter-vs-stream arm needs three kernels the GATE never sees (one of them has
# an atomic, which the gate's census would fail by design), so they live in bench/shaders/ and are compiled here,
# beside the kernels compiled above.  Same flags as the gate.
echo "== compiling the bench-local shaders (bench/shaders/*.comp) -> $SPV"
BSH="$ROOT/bench/shaders"
shopt -s nullglob; bcomps=("$BSH"/*.comp); shopt -u nullglob
for f in "${bcomps[@]}"; do
  name="$(basename "$f" .comp)"
  if ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$f" -o "$SPV/$name.spv" 2>"$BUILD/$name.err"; then
    echo "  FAIL glslc $name"; sed -n '1,8p' "$BUILD/$name.err"; rc=1; continue
  fi
  if ! spirv-val --target-env vulkan1.3 "$SPV/$name.spv" 2>>"$BUILD/$name.err"; then
    echo "  FAIL spirv-val $name"; sed -n '1,8p' "$BUILD/$name.err"; rc=1; continue
  fi
  printf '  OK   %s\n' "$name"
done
[ $rc -eq 0 ] || { echo "== bench shader build FAILED"; exit 1; }

echo "== building vk_bench (-O2 -Werror)"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$ROOT/harness" \
    -o "$BUILD/vk_bench" "$ROOT/bench/vk_bench.cpp" \
    "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_compat.cpp" "$ROOT/harness/vk_stack.cpp" \
    -lvulkan || { echo "== vk_bench build FAILED"; exit 1; }

echo "== running vk_bench under every Vulkan ICD that reports a device"
ran=0
declare -a names=()
for icd in /usr/share/vulkan/icd.d/*.json; do
  [ -e "$icd" ] || continue
  name="$(basename "$icd" .json)"
  first="$(VK_ICD_FILENAMES="$icd" "$BUILD/vk_bench" --list 2>/dev/null | head -1)"
  if [ -z "$first" ]; then printf '  --   %-16s no device\n' "$name"; continue; fi
  log="$BUILD/icd-$name.log"
  VK_ICD_FILENAMES="$icd" "$BUILD/vk_bench" --spv-dir "$SPV" "$@" >"$log" 2>&1
  st=$?
  printf '  %s   %-16s %s\n' "$([ $st -eq 0 ] && echo OK || echo FAIL)" "$name" "$first"
  grep -E '^ROW ' "$log" | sed 's/^/      /'
  ran=$((ran+1)); names+=("$name")
done
if [ "$ran" -lt 2 ]; then
  echo "  note: only $ran implementation(s) exercised - a single-ICD run is not a cross-device proof"
fi

# ---- the comparison: for kernels BOTH ICDs measured, the ratio (the harness's own distinguishability proof) --
if command -v python3 >/dev/null && [ "$ran" -ge 2 ]; then
  echo "== cross-ICD comparison (median ms; ratio = slowest/fastest)"
  python3 - "$BUILD" "${names[@]}" <<'PY'
import re, sys, os
build, names = sys.argv[1], sys.argv[2:]
rows = {}
for n in names:
    p = os.path.join(build, f"icd-{n}.log")
    if not os.path.exists(p): continue
    for line in open(p, errors="replace"):
        m = re.match(r"ROW (\S+)\s*\|", line)
        if not m: continue
        k = m.group(1)
        mm = re.search(r"med\s+([0-9.]+) ms", line)
        if mm: rows.setdefault(k, {})[n] = float(mm.group(1))
for k in rows:
    d = rows[k]
    vals = [(n, v) for n, v in d.items()]
    if len(vals) < 2: continue
    vals.sort(key=lambda x: x[1])
    fast, slow = vals[0], vals[-1]
    ratio = slow[1] / fast[1] if fast[1] > 0 else 0
    print(f"  {k:<24} " + "  ".join(f"{n}={v:.4f}" for n, v in vals) + f"   ratio {ratio:.2f}x")
PY
fi

echo "== logs: $BUILD/icd-*.log"
exit $rc
