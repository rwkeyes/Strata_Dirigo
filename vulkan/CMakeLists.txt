# vulkan/CMakeLists.txt - the Vulkan backend, a SEPARATE backend tree (like sycl/), configured only when the
# top-level `-DSTRATA_ENABLE_VULKAN=ON` is set.  The integration plan and the increments are in
# `ports/vulkan/plan/BACKEND-INTEGRATION.md`; the ported kernels, gates and harness are in `ports/vulkan/`.
#
# INCREMENT 0 (this skeleton) has ONE translation unit: the first engine entry point behind the macro
# (`strata::kernels::fwht256_cuda`, whose header wrapper is fwht256_inplace_cuda in
# include/strata/kernels/kv_q4.hpp).  The device layer it calls - the port's `harness/vk_compute.*` is the seed -
# and the other 27 decode-path kernel entry points land in increments 1+, so this target is deliberately NOT
# link-complete yet.  It CONFIGURES, and its one TU COMPILES; that is what the skeleton was asked to prove.
# Do NOT wire this into setup.py until M-B (one layer, end to end) is green.

find_package(Vulkan REQUIRED)

add_library(strata_vulkan_kernels STATIC
  src/kernels/fwht_vk.cpp)
target_compile_features(strata_vulkan_kernels PRIVATE cxx_std_20)
target_include_directories(strata_vulkan_kernels PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include          # strata/vulkan/vk_backend.hpp (the device-layer seam)
  ${CMAKE_SOURCE_DIR}/include                  # strata/kernels/*.hpp (the engine's wrappers)
  ${CMAKE_SOURCE_DIR}/ports/vulkan/harness)    # vk_compute.hpp (the device layer, adopted in increment 1)
# One macro, the same name as the top-level option, so a TU cannot be built into the wrong backend.
target_compile_definitions(strata_vulkan_kernels PUBLIC STRATA_ENABLE_VULKAN=1)
target_link_libraries(strata_vulkan_kernels PUBLIC Vulkan::Vulkan)

message(STATUS "Strata: Vulkan backend (experimental) - 1 entry point compiled; "
               "see ports/vulkan/plan/BACKEND-INTEGRATION.md")
