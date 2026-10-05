// vulkan/src/kernels/fwht_vk.cpp - the Vulkan backend's fwht256 entry point and the shader it drives.
//
// This is the whole engine-facing surface for fwht256: `strata::kernels::fwht256_cuda`, the symbol
// include/strata/kernels/kv_q4.hpp declares and wraps as `fwht256_inplace_cuda(...)`.  A CUDA/HIP build
// answers that symbol from src/kernels/cuda/kv_q4.cu; a Vulkan build answers it HERE, and only here.
//
// INCREMENT 0 left the body as the SEAM (declared, not run).  INCREMENT 1 implements `strata::vulkan::fwht256`
// against the arena (vulkan/src/device/), so the entry point RUNS: the engine's raw device pointers are
// resolved to arena views, the pipeline is the one the device layer caches for this dispatch's signature, and
// the shader (ports/vulkan/shaders/fwht256.comp) computes the rows.  The numeric proof lives in the port's
// gate (ports/vulkan/harness/vk_gate.cpp, case_fwht256_entry): the ENGINE wrapper's output is compared
// BITWISE against the port's already-green shader case.

#if !defined(STRATA_ENABLE_VULKAN)
#error "fwht_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/kv_q4.hpp"      // the engine's wrapper: fwht256_inplace_cuda -> fwht256_cuda
#include "strata/vulkan/vk_backend.hpp"  // the backend's seam: Stream, stream_of, fwht256
#include "vk_arena.hpp"                  // the arena + pointer->buffer resolution (vulkan/src/device)

#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

// The shader's row is a FIXED 256 floats (fwht256.comp, local_size_x = 256).  The engine's contract is the
// same: `fwht256_cuda(const float* src, float* dst, int64_t n_rows)` rotates `n_rows` rows of 256.
static constexpr uint64_t kFwhtRowFloats = 256;

void fwht256(Stream& s, const float* src, float* dst, int64_t n_rows) {
    if (n_rows <= 0) return;

    // RAW DEVICE POINTERS -> (arena, byte offset).  The engine hands in pointers it got from an arena
    // allocation (or a slice of one); a pointer outside the arena is refused rather than bound.
    const uint64_t bytes = (uint64_t) n_rows * kFwhtRowFloats * sizeof(float);
    Buf sv{}, dv{};
    if (!arena_resolve(s, src, bytes, sv) || !arena_resolve(s, dst, bytes, dv)) {
        std::fprintf(stderr,
                     "strata::vulkan::fwht256: src or dst is not inside this stream's arena (n_rows=%lld) - "
                     "refusing rather than binding a wrong view\n",
                     (long long) n_rows);
        std::exit(2);
    }

    // THE PIPELINE CACHE, keyed as the engine dispatches: 2 storage-buffer bindings (SRC, DST) and a 4-byte
    // push constant ({int n_rows}) - the shader's own interface.  Re-asking the device layer for this key
    // returns the cached VkPipeline (the dispatch path also caches per signature), so a decode step that calls
    // fwht256 every token creates one pipeline, not one per token.
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/fwht256.spv", 2, 4);

    struct Push {
        int n_rows;
    } pc{};
    if (n_rows > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::fwht256: n_rows=%lld overflows the shader's int\n", (long long) n_rows);
        std::exit(2);
    }
    pc.n_rows = (int) n_rows;

    s.ctx->dispatch(pipe, {&sv, &dv}, &pc, sizeof(pc), (uint32_t) n_rows);
}

}  // namespace strata::vulkan

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
        // The backend's own refusal, naming the handle.  Dereferencing is never the answer.
        std::fprintf(stderr, "fwht256_cuda: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::fwht256(*s, src, dst, n_rows);
}

}  // namespace strata::kernels
