# vulkan/CMakeLists.txt - the Vulkan backend, a SEPARATE backend tree (like sycl/), configured only when the
# top-level `-DSTRATA_ENABLE_VULKAN=ON` is set.  The integration plan and the increments are in
# `ports/vulkan/plan/BACKEND-INTEGRATION.md`; the ported kernels, gates and harness are in `ports/vulkan/`.
#
# HOW THE ENGINE'S SOURCES JOIN THIS CONFIGURATION (I1's build-system decision, and it follows sycl/, not an
# invention): the SYCL port is the precedent for "a separate backend tree that takes the engine's own sources"
# (../sycl/CMakeLists.txt).  Its `strata_resolve()` prefers a migrated copy under `sycl/src/` when one exists
# and otherwise compiles the file from the engine tree.  This backend does the same through
# `strata_vulkan_resolve()` below.  The engine's HEADERS are never edited; the engine's SOURCES are named
# explicitly, one increment at a time, and resolved from `${STRATA_ROOT}/` unless `vulkan/` carries a migrated
# copy.  I1 needs NO engine-root source: the fwht256 entry point is the header wrapper in
# `include/strata/kernels/kv_q4.hpp` answered by `vulkan/src/kernels/fwht_vk.cpp`.  I2's glue (the mapped
# copies, rope, kv_stream) are the first engine-root sources and are added through this same function.
#
# The top-level `STRATA_ENABLE_VULKAN` block still returns before the CUDA engine is configured, and that is
# deliberate: src/'s source list assumes CUDA and is not well-formed without it.  What changes in I1 is that the
# Vulkan configuration now builds a REAL backend (device layer + kernel TU + an engine-side target that runs the
# entry point), not just a skeleton that compiled.  A full engine build under this option is a later increment
# and is priced in the plan.

cmake_minimum_required(VERSION 3.24)
find_package(Vulkan REQUIRED)

set(STRATA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# --- source resolution: a migrated copy under vulkan/ when there is one, the engine tree otherwise -----------
# (Same rule as ../sycl/CMakeLists.txt's strata_resolve, so the two backend trees read alike.)
function(strata_vulkan_resolve out)
  set(r "")
  foreach(f IN LISTS ARGN)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${f})
      list(APPEND r ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    elseif(EXISTS ${STRATA_ROOT}/${f})
      list(APPEND r ${STRATA_ROOT}/${f})
    else()
      message(FATAL_ERROR "strata_vulkan_resolve: no such source: ${f}")
    endif()
  endforeach()
  set(${out} ${r} PARENT_SCOPE)
endfunction()

# --- the device layer (the port's harness/vk_compute.* ADOPTED; see I1) --------------------------------------
strata_vulkan_resolve(_device
  src/device/vk_compat.cpp
  src/device/vk_stack.cpp
  src/device/vk_compute.cpp
  src/device/vk_arena.cpp
  src/device/sync.cpp)          # I2: the fence/host-handoff replacement for the CUDA doorbell
add_library(strata_vulkan_device STATIC ${_device})
target_compile_features(strata_vulkan_device PRIVATE cxx_std_20)
target_include_directories(strata_vulkan_device PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include          # strata/vulkan/vk_backend.hpp (the seam)
  ${CMAKE_CURRENT_SOURCE_DIR}/src/device        # vk_arena.hpp / the adopted vk_compute.hpp
  ${STRATA_ROOT}/include)                       # strata/kernels/*.hpp (the engine's wrappers)
target_compile_definitions(strata_vulkan_device PUBLIC STRATA_ENABLE_VULKAN=1)
target_link_libraries(strata_vulkan_device PUBLIC Vulkan::Vulkan)

# --- the kernel translation units: one engine entry point per symbol, behind the macro -----------------------
strata_vulkan_resolve(_kernels src/kernels/fwht_vk.cpp src/kernels/native_caps_vk.cpp
  src/kernels/elementwise_vk.cpp
  src/kernels/doorbell_vk.cpp
  src/kernels/gdn_vk.cpp
  src/kernels/matvec_vk.cpp
  src/kernels/qsa_vk.cpp
  src/kernels/ple_vk.cpp
  src/kernels/shared_expert_vk.cpp
  src/kernels/refusals_vk.cpp)  # shared_expert (the MoE shared stage) + the LOUD REFUSALS (unreachable symbols)
                            # + the PLE/GR shared stages and the MoE routing rows (ple_vk.cpp)
add_library(strata_vulkan_kernels STATIC ${_kernels})
target_link_libraries(strata_vulkan_kernels PUBLIC strata_vulkan_device)

# --- the engine-side target: build + run the entry point through the engine's OWN wrapper --------------------
# This is the "something that links the device layer + the Vulkan TU against the engine's own code and exercises
# the entry point" the plan asks for.  It is NOT the numeric oracle - the port's gate is (ports/vulkan/harness).
add_executable(strata_vk_entry_smoke tests/entry_point_smoke.cpp)
target_link_libraries(strata_vk_entry_smoke PRIVATE strata_vulkan_kernels)

message(STATUS "Strata: Vulkan backend (experimental) - device layer + sync (the doorbell replacement) + "
               "the engine entry points (fwht256 + 11 elementwise/glue + 5 doorbell + the cvec module + "
               "all 14 GDN/DeltaNet mixer kernels) + strata_vk_entry_smoke; see "
               "ports/vulkan/plan/BACKEND-INTEGRATION.md")

# ============================================================================================================
# THE CUDA-RUNTIME COMPATIBILITY SHIM (deliverable A of the approved hybrid route)
# ============================================================================================================
# The engine's host TUs call the CUDA runtime directly (`cudaMalloc`, `cudaMemcpy`, `cudaHostAlloc`, ...).  On
# this backend those names are answered HERE, over the port's device layer.  `cuda_compat/` is put FIRST on the
# include path of every engine-facing target (PUBLIC below), so `#include <cuda_runtime.h>` resolves to the
# shim - exactly the mechanism `include/strata/platform/hip_compat/` uses on the HIP build.  The engine's
# headers and sources are NOT edited.
strata_vulkan_resolve(_cudart src/compat/cuda_runtime.cpp)
add_library(strata_vulkan_cudart STATIC ${_cudart})
target_include_directories(strata_vulkan_cudart PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_compat     # <cuda_runtime.h> resolves HERE on an engine TU
  ${CMAKE_CURRENT_SOURCE_DIR}/src/device)
target_compile_definitions(strata_vulkan_cudart PUBLIC STRATA_ENABLE_VULKAN=1)
target_link_libraries(strata_vulkan_cudart PUBLIC strata_vulkan_device)

# The shim RUNNING (not merely linking): every entry point exercised end to end on a real device.
add_executable(strata_vk_cudart_smoke tests/cudart_smoke.cpp)
target_link_libraries(strata_vk_cudart_smoke PRIVATE strata_vulkan_cudart)

# ============================================================================================================
# THE ENGINE'S OWN TARGETS (deliverable B) - so a -DSTRATA_ENABLE_VULKAN=ON configure has an engine binary and
# not just the backend library.  This mirrors `../sycl/CMakeLists.txt`'s shape (the established precedent): the
# engine's sources join through `strata_vulkan_resolve()`, a migrated copy under `vulkan/` when one exists and
# `${STRATA_ROOT}/` otherwise.  The engine's HEADERS are never edited.
#
# WHAT IS EXCLUDED, and why (reported, per the increment):
#   * the three `src/core/*.cu` files (device.cu, pinned.cu, remote_expert_opt.cu) - CUDA device code; on this
#     backend the device layer is `vulkan/src/device/`, so they are not compiled.
#   * the PREFILL path (`src/prefill/*`) - not ported (BACKEND-INTEGRATION.md §4); its symbols are simply not
#     provided, so a prefill-calling TU links later or not at all.
#   * the parity/bench TUs (`src/kernels/*_parity.cpp`, `bench/`) - they are oracles against the CUDA path.
#   * the native-expert / ggml-cpu half of `strata_kernels_cpu` (`src/kernels/cpu/native_expert.cpp`,
#     `iq_avx*.cpp`) - it needs the ggml sources the Vulkan configuration does not build.
# The targets below CONFIGURE; whether a given TU links is the plan's own increments (I4/I5) and is measured,
# not asserted, in ports/vulkan/NEXT.md.
# ============================================================================================================
find_package(Threads REQUIRED)

# The engine targets are EXCLUDE_FROM_ALL: they CONFIGURE (the deliverable) and build by NAME, but their
# graph-API-dependent TUs (graph.cpp, session.cpp) do not compile until I5 declares the CUDA graph API, so
# sweeping them into the default `all` would break `cmake --build` for the backend + smokes that DO build.

strata_vulkan_resolve(_vspec src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
add_library(strata_vulkan_spec STATIC EXCLUDE_FROM_ALL ${_vspec})
target_include_directories(strata_vulkan_spec PUBLIC ${STRATA_ROOT}/include)
target_link_libraries(strata_vulkan_spec PUBLIC strata_vulkan_cudart)

strata_vulkan_resolve(_vcpu src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
                      src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp)
add_library(strata_vulkan_kernels_cpu STATIC EXCLUDE_FROM_ALL ${_vcpu})
target_include_directories(strata_vulkan_kernels_cpu PUBLIC ${STRATA_ROOT}/include)
target_link_libraries(strata_vulkan_kernels_cpu PUBLIC strata_vulkan_cudart Threads::Threads)
# The AVX-512 / AVX2 source flags, mirrored from the top-level `strata_kernels_cpu`: without them
# `expert.cpp`'s VNNI/VBMI intrinsics do not compile (measured - "target specific option mismatch").
if(MSVC)
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/expert.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/expert.cpp
                              PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set_source_files_properties(${STRATA_ROOT}/src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
endif()

strata_vulkan_resolve(_vcore src/core/graph.cpp src/core/weights.cpp src/core/layout.cpp src/platform/memory.cpp)
add_library(strata_vulkan_core STATIC EXCLUDE_FROM_ALL ${_vcore})
target_include_directories(strata_vulkan_core PUBLIC ${STRATA_ROOT}/include)
target_link_libraries(strata_vulkan_core PUBLIC strata_vulkan_cudart strata_vulkan_kernels)

strata_vulkan_resolve(_vengine
  src/core/layer.cpp src/core/session.cpp src/core/expert_source.cpp src/core/remote_experts.cpp
  src/core/expert_cache.cpp src/core/native_head.cpp src/core/native_dense.cpp src/core/verify.cpp
  src/core/mtp.cpp src/core/conversation_snapshot.cpp src/core/conversation_state.cpp
  src/core/conversation_memory.cpp src/core/peer_experts.cpp)
add_library(strata_vulkan_engine STATIC EXCLUDE_FROM_ALL ${_vengine})
target_link_libraries(strata_vulkan_engine PUBLIC strata_vulkan_core strata_vulkan_kernels_cpu)

strata_vulkan_resolve(_vmain src/program/generate.cpp)
add_executable(strata_vulkan EXCLUDE_FROM_ALL ${_vmain})
target_link_libraries(strata_vulkan PRIVATE strata_vulkan_engine strata_vulkan_spec)

message(STATUS "Strata: Vulkan engine targets (deliverable B) - strata_vulkan_core / strata_vulkan_engine / "
               "strata_vulkan_spec / strata_vulkan_kernels_cpu / strata_vulkan (executable) + the CUDA-runtime "
               "shim strata_vulkan_cudart + strata_vk_cudart_smoke")
