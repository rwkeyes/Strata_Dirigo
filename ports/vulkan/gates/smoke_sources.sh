#!/usr/bin/env bash
# ports/vulkan/gates/smoke_sources.sh - BUILD AND RUN `strata_vk_cudart_smoke` for the gate.
#
# WHY THIS EXISTS.  `strata_vk_cudart_smoke` is a CMake target (`vulkan/CMakeLists.txt:120`) that exercises the
# CUDA-runtime shim end to end on a real device - and NOTHING ran it, so the shipped `cudaStreamQuery` fix
# (a pending live batch must NOT be reported as `cudaSuccess`) was UNGUARDED: a future change could restore the
# wrong answer with every other arm green.  `run_gate.sh` sources this file to build the target and run it, and
# `inject-verify.sh` sources it for the falsification in the other direction (`STRATA_VK_QUERY_NOFIX=1`).
#
# THE BUILD DIR IS NAMED, NOT GUESSED, AND A FAILURE IS LOUD.  The target lives in a CMake configuration, so the
# helper configures one (only if `CMakeCache.txt` is absent) and builds the ONE target.  A missing toolchain or a
# build error returns non-zero and the caller reports it - never a silent skip, which is the failure mode this
# whole file is about.
#
#   smoke_build <engine-tree-root>   -> 0 on success (configures STRATA_ENABLE_VULKAN=ON if needed)
#   smoke_bin                        -> the binary's path
set -uo pipefail

STRATA_VK_BUILD="${STRATA_VK_BUILD:-$HOME/vkbuild-vulkan}"

smoke_build() {
    local tree="$1"
    if [ ! -f "$STRATA_VK_BUILD/CMakeCache.txt" ]; then
        cmake -S "$tree" -B "$STRATA_VK_BUILD" -DSTRATA_ENABLE_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
            >"$STRATA_VK_BUILD.smoke.configure.log" 2>&1 || { cat "$STRATA_VK_BUILD.smoke.configure.log"; return 1; }
    fi
    cmake --build "$STRATA_VK_BUILD" --target strata_vk_cudart_smoke -j"$(nproc)" \
        >"$STRATA_VK_BUILD.smoke.build.log" 2>&1 || { tail -20 "$STRATA_VK_BUILD.smoke.build.log"; return 1; }
    [ -x "$STRATA_VK_BUILD/vulkan/strata_vk_cudart_smoke" ] || return 1
    return 0
}

smoke_bin() { printf '%s' "$STRATA_VK_BUILD/vulkan/strata_vk_cudart_smoke"; }
