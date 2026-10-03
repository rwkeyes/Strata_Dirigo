#!/usr/bin/env bash
# ports/vulkan/gates/run_gate.sh - THE audit and verification entry point for the Vulkan port.
#
# It compiles every shader from SOURCE, validates the SPIR-V, cross-checks each shader's declared workgroup
# size against the host's constant, censuses the SPIR-V operations for embellishment, then runs the numeric
# gate.  Every check below exists because the opposite failure was observed somewhere:
#
#   * shaders are compiled from .comp on EVERY run, and stale .spv files are deleted first - a gate that reads
#     a binary built before the last interface change tests the wrong kernel and reports a confident verdict.
#   * the workgroup size is checked against the HOST's own constant (one source of truth, checked for
#     agreement) - the host sizes every dispatch from it, so a silent mismatch is a wrong grid, not a compile
#     error.
#   * the operation census is checked - a "compiled and validated" shader is not evidence that the kernel
#     lowered what the source meant (a reduction that vanished compiles cleanly and returns one lane's value).
#   * an empty or all-skipped run FAILS - an assertion over an empty input passes vacuously.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # ports/vulkan
TREE="$(cd "$ROOT/../.." && pwd)"                 # the engine tree (for the real bf16/f16 headers)
SH="$ROOT/shaders"
BUILD="$ROOT/harness/build"
mkdir -p "$BUILD"

rc=0
fail() { printf '  FAIL %s\n' "$*"; rc=1; }

# The host's workgroup size, read out of the source rather than repeated here.
HOST_LOCAL="$(sed -n 's/.*constexpr uint32_t kLocalSize = \([0-9]*\).*/\1/p' "$ROOT/harness/vk_gate.cpp" | head -1)"
if [ -z "$HOST_LOCAL" ]; then
  echo "== cannot read kLocalSize from harness/vk_gate.cpp"; exit 1
fi
echo "== compiling shaders from source ($SH); host workgroup size = $HOST_LOCAL"
rm -f "$SH"/*.spv

shopt -s nullglob
comps=("$SH"/*.comp)
shopt -u nullglob
if [ "${#comps[@]}" -eq 0 ]; then
  echo "== no shaders found in $SH - nothing to verify, refusing to report success"; exit 1
fi

for f in "${comps[@]}"; do
  name="$(basename "$f" .comp)"
  if ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$f" -o "$SH/$name.spv" 2>"$BUILD/$name.glslerr"; then
    fail "$name (glslc)"; sed -n '1,12p' "$BUILD/$name.glslerr"; continue
  fi
  # NOTE: this spirv-tools build wants `--target-env vulkan1.3`; the `=` form prints usage and exits 1.
  if ! spirv-val --target-env vulkan1.3 "$SH/$name.spv" 2>>"$BUILD/$name.glslerr"; then
    fail "$name (spirv-val)"; sed -n '1,12p' "$BUILD/$name.glslerr"; continue
  fi

  ls_line="$(spirv-dis "$SH/$name.spv" | grep -oE 'OpExecutionMode %main LocalSize [0-9]+ [0-9]+ [0-9]+' | head -1)"
  if [ -z "$ls_line" ]; then
    fail "$name (no LocalSize execution mode - the host cannot size a grid for it)"; continue
  fi
  if ! grep -q "LocalSize $HOST_LOCAL 1 1" <<<"$ls_line"; then
    fail "$name ($ls_line does not match the host's kLocalSize=$HOST_LOCAL)"; continue
  fi

  # OPERATION CENSUS.  Count the ops that carry meaning; every kernel's expected set is derived from its CUDA
  # source.  The KUDA sources in this wave contain no subgroup op, no barrier and no atomic except rms_norm's
  # reduction, so any of those appearing in the others is an embellishment (the model-port analogue: a plain
  # gather came back with a subgroup reduction plus an atomicAdd, commented as "safer").
  census="$(spirv-dis "$SH/$name.spv" | grep -oE 'OpGroupNonUniform[A-Za-z]*|OpControlBarrier|OpAtomic[A-Za-z]*' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')"
  case "$name" in
    rms_norm)
      grep -q 'OpGroupNonUniformFAdd' <<<"$census" || fail "$name (no subgroup reduction in the SPIR-V - the kernel did not lower its sum)"
      ;;
    *)
      if [ -n "$census" ]; then
        fail "$name (unexpected ops: $census - the CUDA source has no subgroup op, barrier or atomic here)"
      fi
      ;;
  esac
  printf '  OK   %-22s %s | census: %s\n' "$name" "$ls_line" "${census:-none}"
done

[ $rc -eq 0 ] || { echo "== shader checks FAILED"; exit 1; }

# -----------------------------------------------------------------------------------------------------------
# TOOLCHAIN PROBES.  These do not decide whether the port RUNS - they decide what it CAN do, and every one of
# them is a property of a tool version rather than of this code, so they will change under us.  Two matter:
#   * cooperative matrix: the only route from Vulkan to the matrix units (Intel XMX / AMD WMMA).  It IS supported
#     by this toolchain - the earlier "absent" verdict came from a broken probe shader, not from glslc.
#   * fp64: double ARITHMETIC compiles; exp(double) does not, which is what keeps the silu reference host-side.
echo "== toolchain probes (capability, not correctness)"
TP="$BUILD/toolprobe"; mkdir -p "$TP"
missing=""
for t in glslc spirv-val g++; do command -v "$t" >/dev/null || missing="$missing $t"; done
if [ -n "$missing" ]; then echo "  FAIL required tool(s) absent:$missing"; exit 1; fi

# Cooperative matrix: compile the REAL kernel, not a throwaway fixture.  The fixture this replaces was itself
# broken - it omitted GL_KHR_memory_scope_semantics, called `coopmatMulAdd` (the extension spells it
# `coopMatMulAdd`) and used `gl_MatrixLayoutRowMajor` (it is `gl_CooperativeMatrixLayoutRowMajor`) - so the
# answer to "does it compile?" was reported as a toolchain limitation that did not exist.  A probe that IS the
# shipping shader cannot drift away from the capability it claims.
if glslc --target-env=vulkan1.3 -fshader-stage=compute "$SH/gemm_coopmat.comp" -o "$TP/coopmat.spv" 2>"$TP/coopmat.err"; then
  coop="SUPPORTED - OpCooperativeMatrixMulAddKHR is emitted"
else
  coop="ABSENT - $(head -1 "$TP/coopmat.err" | cut -c1-64)"
fi

# fp64: arithmetic and transcendentals are DIFFERENT questions, and the old single probe conflated them.  Basic
# double arithmetic compiles fine on this toolchain; what it cannot express is exp(double), which is the actual
# reason the double-precision silu reference is still a host-side oracle.
cat > "$TP/fp64_arith.comp" <<'EOF'
#version 450
#extension GL_ARB_gpu_shader_fp64 : require
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) buffer O { float v[]; } o;
void main() { double d = double(o.v[gl_LocalInvocationIndex]) * 1.5 + 2.0; o.v[0] = float(d); }
EOF
cat > "$TP/fp64_exp.comp" <<'EOF'
#version 450
#extension GL_ARB_gpu_shader_fp64 : require
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) buffer O { float v[]; } o;
void main() { double d = double(o.v[gl_LocalInvocationIndex]); o.v[0] = float(exp(-d)); }
EOF
a_ok=no; if glslc --target-env=vulkan1.3 -fshader-stage=compute "$TP/fp64_arith.comp" -o "$TP/fp64a.spv" 2>"$TP/fp64a.err"; then a_ok=yes; fi
e_ok=no; if glslc --target-env=vulkan1.3 -fshader-stage=compute "$TP/fp64_exp.comp" -o "$TP/fp64e.spv" 2>"$TP/fp64e.err"; then e_ok=yes; fi
if [ "$a_ok" = yes ] && [ "$e_ok" = no ]; then
  fp64="arithmetic yes, exp(double) NO (host-side oracle stays)"
elif [ "$a_ok" = yes ]; then
  fp64="SUPPORTED"
elif [ "$e_ok" = no ]; then
  fp64="absent (no fp64 at all)"
else
  fp64="odd: exp(double) compiles but arithmetic does not"
fi

if spirv-val --target-env vulkan1.3 "$ROOT/shaders/copy.spv" >/dev/null 2>&1; then valform="--target-env <env>"; else valform="unknown"; fi

printf '  %-42s %s\n' "glslc (shader compile)"        "$(glslc --version 2>/dev/null | head -1)"
printf '  %-42s %s\n' "spirv-val (SPIR-V validation)" "$(spirv-val --version 2>/dev/null | head -1)"
printf '  %-42s %s\n' "g++ (harness build, C++20)"    "$(g++ --version 2>/dev/null | head -1)"
printf '  %-42s %s\n' "spirv-val accepted arg form"   "$valform"
printf '  %-42s %s\n' "cooperative matrix (matrix-unit path)" "$coop"
printf '  %-42s %s\n' "fp64 arithmetic / transcendentals" "$fp64"
echo "  (a capability reported absent here is a TOOLCHAIN limit, not a port defect - see STACK-COMPAT.md)"

echo "== building the harness (-Werror: hygiene is part of the gate)"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$TREE/include" \
    -o "$BUILD/vk_gate" "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_compat.cpp" \
    "$ROOT/harness/vk_stack.cpp" \
    "$ROOT/harness/vk_gate.cpp" -lvulkan || exit 1

echo "== numeric gate"
# The gate needs a few MiB of buffers, so it runs with the reserve and its floor at 0: this box's resident local
# model already holds the card, and the real policy would (CORRECTLY) refuse to allocate beside it.  The POLICY
# itself is still verified - case_reserve_policy tests the real defaults (1024 MiB, 256 MiB floor, 25% cap) as a
# pure function, and case_memory_budget checks the live driver figures - so the escape hatch does not weaken the
# contract; it only lets a small correctness harness run on a card someone else is using.
export STRATA_VK_DESKTOP_RESERVE_MIB=0
export STRATA_VK_RESERVE_FLOOR_MIB=0
out="$("$BUILD/vk_gate" --spv-dir "$SH" "$@")"
status=$?
printf '%s\n' "$out"
[ $status -eq 0 ] || exit 1

# The gate's own summary is asserted, so that a run which somehow skipped everything cannot come out green
# even if the binary's exit status were wrong.
summary="$(grep -E '^== [0-9]+ passed, [0-9]+ failed, [0-9]+ skipped$' <<<"$out" | tail -1)"
[ -n "$summary" ] || { echo "== no summary line from the gate - refusing to report success"; exit 1; }
passed="$(sed -n 's/^== \([0-9]*\) passed.*/\1/p' <<<"$summary")"
skipped="$(sed -n 's/.* \([0-9]*\) skipped$/\1/p' <<<"$summary")"
if [ "${passed:-0}" -eq 0 ]; then echo "== zero cases passed"; exit 1; fi
if [ "${skipped:-0}" -ne 0 ]; then echo "== $skipped case(s) SKIPPED - a skipped case is not a passing one"; exit 1; fi

# -----------------------------------------------------------------------------------------------------------
# CROSS-IMPLEMENTATION ARM.  Every ICD that reports a device is run, because a kernel can be correct on one
# implementation and wrong on another: the reduction below was written for a 64-lane subgroup and silently
# dropped 24 of 32 subgroup sums at 8 lanes (llvmpipe), where every value came out ~2x off.  That was invisible
# on the only device it had been tested on.  An ICD with no device is reported as such and skipped - skipping a
# MISSING HARDWARE is honest; skipping a present implementation is not.
echo "== cross-implementation arm (every Vulkan ICD that reports a device)"
impls=0
for icd in /usr/share/vulkan/icd.d/*.json; do
  [ -e "$icd" ] || continue
  name="$(basename "$icd" .json)"
  first="$(VK_ICD_FILENAMES="$icd" "$BUILD/vk_gate" --list 2>/dev/null | head -1)"
  if [ -z "$first" ]; then
    printf '  --   %-16s no device on this machine\n' "$name"
    continue
  fi
  impls=$((impls+1))
  icd_log="$BUILD/icd-$name.log"
  if VK_ICD_FILENAMES="$icd" "$BUILD/vk_gate" --spv-dir "$SH" >"$icd_log" 2>&1; then
    printf '  OK   %-16s %s\n' "$name" "$(grep -E '^== [0-9]+ passed' "$icd_log" | tail -1)"
  else
    printf '  FAIL %-16s %s\n' "$name" "$(grep -E '^== [0-9]+ passed' "$icd_log" | tail -1)"
    grep -E '^FAIL' "$icd_log" | head -5
    rc=1
  fi
done
if [ "$impls" -lt 2 ]; then
  echo "  note: only $impls implementation(s) exercised - a width-dependent defect can hide in a single one"
fi
[ $rc -eq 0 ] || { echo "== cross-implementation arm FAILED"; exit 1; }

echo "== shader checks + numeric gate: $summary ($impls implementation(s))"
