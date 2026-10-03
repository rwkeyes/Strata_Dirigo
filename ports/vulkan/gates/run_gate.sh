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

echo "== building the harness (-Werror: hygiene is part of the gate)"
g++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$TREE/include" \
    -o "$BUILD/vk_gate" "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_gate.cpp" -lvulkan || exit 1

echo "== numeric gate"
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
