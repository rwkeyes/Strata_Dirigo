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
  src/kernels/doorbell_vk.cpp)   # I2: the engine's doorbell_* (the split submission; sync.* is the design)
add_library(strata_vulkan_kernels STATIC ${_kernels})
target_link_libraries(strata_vulkan_kernels PUBLIC strata_vulkan_device)

# --- the engine-side target: build + run the entry point through the engine's OWN wrapper --------------------
# This is the "something that links the device layer + the Vulkan TU against the engine's own code and exercises
# the entry point" the plan asks for.  It is NOT the numeric oracle - the port's gate is (ports/vulkan/harness).
add_executable(strata_vk_entry_smoke tests/entry_point_smoke.cpp)
target_link_libraries(strata_vk_entry_smoke PRIVATE strata_vulkan_kernels)

message(STATUS "Strata: Vulkan backend (experimental) - device layer + sync (the doorbell replacement) + "
               "4 entry points (fwht256 + silu/scale/f32_to_bf16 glue) + strata_vk_entry_smoke; see "
               "ports/vulkan/plan/BACKEND-INTEGRATION.md")
