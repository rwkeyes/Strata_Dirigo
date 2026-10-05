// vulkan/src/kernels/fwht_vk.cpp - the Vulkan backend's ONE engine entry point, for the increment-0 skeleton.
//
// This is the whole engine-facing surface of the skeleton: `strata::kernels::fwht256_cuda`, the symbol
// include/strata/kernels/kv_q4.hpp declares and wraps as `fwht256_inplace_cuda(...)`.  A CUDA/HIP build answers
// that symbol from src/kernels/cuda/kv_q4.cu; a Vulkan build answers it HERE, and only here.
//
// The body is deliberately the SEAM, not the dispatch: dispatching needs the arena (engine device pointer ->
// VkBuffer/offset), which increment 1 builds.  What this TU proves is exactly what the increment was scoped to
// prove - the macro, the tree, and one entry point wired at the build-system level, compiling behind
// -DSTRATA_ENABLE_VULKAN and refusing to compile without it.

#if !defined(STRATA_ENABLE_VULKAN)
#error "fwht_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/kv_q4.hpp"      // the engine's wrapper: fwht256_inplace_cuda -> fwht256_cuda
#include "strata/vulkan/vk_backend.hpp"  // the backend's device-layer seam

namespace strata::kernels {

// kv_q4.hpp declares:
//     void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream);
// and wraps it:
//     fwht256_inplace_cuda(data, n_rows, stream) { fwht256_cuda(data, data, n_rows, stream); }
// Same four arguments as the CUDA one; `stream` is the engine's opaque handle, which on this backend is a
// strata::vulkan::Stream*.
void fwht256_cuda(const float* src, float* dst, int64_t n_rows, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        // TODO(increment 1): the backend's own refusal, naming the handle.  Refusing here (rather than
        // dereferencing) is the port's rule, kept even in the skeleton.
        return;
    }
    strata::vulkan::fwht256(*s, src, dst, n_rows);
}

}  // namespace strata::kernels
