// vulkan/src/kernels/elementwise_vk.cpp - the Vulkan backend's ELEMENTWISE GLUE entry points (I2).
//
// ============================================================================================================
// WHICH THREE, AND WHY THESE THREE (derived from the layer body, not from the plan's list)
// ============================================================================================================
//
// The plan's I2 list (`plan/BACKEND-INTEGRATION.md` §3) names eleven kernel rows.  This TU wires the first
// three the layer chain reaches IN EXECUTION ORDER, read out of `src/core/layer.cpp`'s layer body:
//
//   block_layer (layer.cpp:1344) -> block_layer_pre -> the MIXER -> ... -> block_layer_post
//
// and the mixer for 36 of the 48 layers is `gdn_layer` (layer.cpp:223).  Its body reaches the glue in this
// order, and the first three are:
//
//   1. `silu_inplace`      layer.cpp:257   gdn_layer stage 10 - SiLU of the conv output (`b.h`), the
//                                          legacy branch of the conv+SiLU pair (`!native_gdn_enabled()`).
//   2. `scale_inplace`     layer.cpp:276   gdn_layer stage 11 - the recurrence's extra 1/sqrt(S) on q
//                                          (`b.h`, qk floats), again the legacy branch.  Its own comment
//                                          says the scale is a backend-specific boundary: the native CUDA
//                                          applies it after the readout dot, this legacy path before.
//   3. `f32_to_bf16_bulk`  layer.cpp:290   gdn_layer stage 12 - the alpha/beta activation, reached by
//                                          DEFAULT (`native_bf16_projections` defaults false, layer.cpp:91)
//                                          and again at moe_route:357, moe_shared:430 and qsa_layer:900.
//
// The plan's list is alphabetical in part and is NOT an order: reading it as one would start with
// `add_inplace` (which the layer body never calls - its call sites are `expert_source.cpp:2353` and
// `remote_expert_opt.cu:127`, the R4 hit/miss split) then `embedding_gather` (a token-level lookup, and at
// `layer.cpp:1083` inside `embed_row`, before layer 0).  The three above are the first three the LAYER body
// itself reaches.
//
// THE NEXT THREE, in the same reading, are `gdn_gate` (layer.cpp:300, stage 12), `rms_norm_weighted`
// (layer.cpp:880, the QSA mixer's norm) and `embedding_gather` (layer.cpp:1083).  They are I2's remainder.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each entry point is the thin wrapper already declared in
// `include/strata/kernels/elementwise.hpp`; this TU answers the `strata::kernels::` symbol the wrapper
// calls.  On a CUDA/HIP build `src/kernels/cuda/elementwise.cu` answers them; on this build THIS file does.
// The bodies resolve the engine's raw device pointers to arena views (vk_arena.hpp), take the pipeline the
// device layer caches for the shader's own signature, and dispatch - the same shape as `fwht_vk.cpp`, and
// the shader each one drives is the shader the port's numeric gate has already gated:
//
//     scale_inplace      -> scale.spv        (case_scale,       gate verdict "scale_inplace")
//     silu_inplace       -> silu_f32.spv     (case_silu,        gate verdict "silu_inplace (f32 vs double ref)")
//     f32_to_bf16_bulk   -> f32_to_bf16.spv  (run_conversion<>,"f32_to_bf16 (bit-exact)")
//
// The gate's `case_*_entry` cases (I2) re-run each with the ENGINE WRAPPER and compare BITWISE to that same
// shader path, so this file's claim is not "it compiles" but "the wrapper's answer equals the ported
// shader's answer".
#if !defined(STRATA_ENABLE_VULKAN)
#error "elementwise_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/elementwise.hpp"  // the engine's wrappers: silu_inplace / scale_inplace / f32_to_bf16_bulk
#include "strata/vulkan/vk_backend.hpp"    // the backend's seam: Stream, stream_of, the elementwise decls
#include "vk_arena.hpp"                    // the arena + pointer->buffer resolution

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

// The port's elementwise shaders are all `local_size_x = 256` with one element per invocation (checked
// against the host's constant by the gate's shader census), so a grid is the element count over 256.
static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// `x[i] /= 1 + exp(-x[i])`, in place (shader silu_f32.spv: 1 storage buffer, push {int n}).
void silu_inplace(Stream& s, float* x, int64_t n) {
    if (n <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv)) {
        std::fprintf(stderr, "strata::vulkan::silu_inplace: x is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::silu_inplace: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/silu_f32.spv", 1, sizeof(int32_t));
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `x[i] *= s`, in place (shader scale.spv: 1 storage buffer, push {int n; float s}).
void scale_inplace(Stream& s, float* x, int64_t n, float factor) {
    if (n <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv)) {
        std::fprintf(stderr, "strata::vulkan::scale_inplace: x is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::scale_inplace: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/scale.spv", 1, 8);
    struct Push {
        int32_t n;
        float s;
    } pc{};
    pc.n = (int32_t) n;
    pc.s = factor;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `y[i] = bf16(x[i])` (shader f32_to_bf16.spv: X float[] read, Y uint16_t[] write, push {int n}).
void f32_to_bf16_bulk(Stream& s, const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv) ||
        !arena_resolve(s, y, (uint64_t) n * sizeof(uint16_t), yv)) {
        std::fprintf(stderr, "strata::vulkan::f32_to_bf16_bulk: x or y is not inside this stream's arena "
                             "(n=%lld) - refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::f32_to_bf16_bulk: n=%lld overflows the shader's int\n",
                     (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/f32_to_bf16.spv", 2, sizeof(int32_t));
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/elementwise.hpp declares ---------------
namespace strata::kernels {

// elementwise.hpp: `void silu_inplace(float* x, int64_t n, void* stream);`
void silu_inplace(float* x, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "silu_inplace: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::silu_inplace(*s, x, n);
}

// elementwise.hpp: `void scale_inplace(float* x, int64_t n, float s, void* stream);`
void scale_inplace(float* x, int64_t n, float s, void* stream) {
    strata::vulkan::Stream* st = strata::vulkan::stream_of(stream);
    if (st == nullptr) {
        std::fprintf(stderr, "scale_inplace: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::scale_inplace(*st, x, n, s);
}

// elementwise.hpp: `void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);`
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "f32_to_bf16_bulk: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::f32_to_bf16_bulk(*s, x, y, n);
}

}  // namespace strata::kernels
