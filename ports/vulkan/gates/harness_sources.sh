#!/usr/bin/env bash
# ports/vulkan/gates/harness_sources.sh - the ONE place the gate's harness source list is built.
#
# WHY THIS FILE EXISTS.  `run_gate.sh` and `inject-verify.sh` each compiled the gate from a HAND-KEPT literal TU
# list, and that list went stale THREE batches running (most recently missing `iq_vk.cpp`, `moe_vk.cpp`,
# `sampler_vk.cpp`): an injection into a backend TU that was not in the list rebuilt a gate that could not link,
# ran the OLD binary, and reported "NOT FALSIFIED" - silently weakening every injection in that run.  A list a
# human has to remember to extend WILL be forgotten; a list derived from the tree cannot.  So the engine-side TUs
# are GLOBBED, and `build_harness` fails LOUDLY if any `vulkan/src/**/*.cpp` is missing from the list it hands
# the compiler.
#
# The caller must set ROOT (ports/vulkan), TREE (the engine root) and GATE (the output binary path).

# Print the harness source list, one path per line.  The harness's own TUs (its private `portvk` namespace) plus
# EVERY `vulkan/src/**/*.cpp` plus the engine host TUs the harness links.
harness_sources() {
  local f
  for f in "$ROOT"/harness/*.cpp; do printf '%s\n' "$f"; done
  find "$TREE/vulkan/src" -name '*.cpp' | LC_ALL=C sort
  printf '%s\n' "$TREE/src/kernels/ngram.cpp" "$TREE/src/ngram/ple_reader.cpp" "$TREE/src/platform/direct_file.cpp"
}

# Build the gate.  Guard first: the count of `vulkan/src/**/*.cpp` on disk must EQUAL the count the list carries -
# if a backend TU is not handed to the compiler the build silently tests a stale backend.
build_harness() {
  local list n_disk n_list
  list="$(harness_sources)"
  n_disk="$(find "$TREE/vulkan/src" -name '*.cpp' | wc -l)"
  n_list="$(printf '%s\n' "$list" | grep -c "^$TREE/vulkan/src/")"
  if [ "$n_disk" -eq 0 ] || [ "$n_disk" -ne "$n_list" ]; then
    printf 'build_harness: %s engine TU(s) under %s/vulkan/src but %s in the source list - REFUSING (a stale list silently weakens every injection and every gate run)\n' \
      "$n_disk" "$TREE" "$n_list" >&2
    return 1
  fi
  # shellcheck disable=SC2086
  g++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$TREE/include" \
      -I"$TREE/vulkan/include" -I"$TREE/vulkan/include/cuda_compat" -I"$TREE/vulkan/src/device" \
      -DSTRATA_ENABLE_VULKAN=1 -o "$GATE" $list \
      -lpthread -lvulkan
}
