// vulkan/src/kernels/gdn_vk.cpp - the Vulkan backend's GDN / DeltaNet MIXER entry points.
//
// ============================================================================================================
// WHICH SIX, AND WHY THESE SIX (derived from the mixer's own body, not from the plan's list)
// ============================================================================================================
//
// The GDN / DeltaNet mixer is `gdn_layer` (`src/core/layer.cpp:223`), the block every GDN layer runs - 36 of
// the model's 48 layers - reached from `block_layer_pre` (`layer.cpp:1259`: `if (qsa) qsa_layer(...) else
// gdn_layer(...)`).  Its body reaches the GDN kernels in this order; the first six are these (the previous
// glue batches read the SAME body for their own subset, so this is the established method, not a new one):
//
//   1. `fused_gdn_conv_l2`   layer.cpp:250   stage 3 (conv): the FUSED conv+SiLU+two L2 norms, taken when
//                                            `fused_pre` holds (g_fused_gdn && native_gdn_enabled() &&
//                                            native_bf16_projections && d_conv==4 && S==128).
//   2. `native_gdn_conv_silu` layer.cpp:253  stage 3, the `else` native branch (native_gdn_enabled()).
//   3. `gdn_conv_step`       layer.cpp:255   stage 3, the legacy `else` (its SiLU is the separate
//                                            `silu_inplace`, already wired in elementwise_vk.cpp at :257).
//   4. `native_gdn_l2_norm`  layer.cpp:266   stage 4 (q/k L2), the native branch, called twice (:266/:267).
//   5. `gdn_l2_norm`         layer.cpp:269   stage 4, the legacy `else`, called twice (:269/:270).
//   6. `fused_gdn_ab`        layer.cpp:287   stage 5 (alpha/beta), the fused branch - the first GDN call of the
//                                            beta/gate stage, BEFORE `native_gdn_beta_gate` (:296).
//
// WHY SOURCE ORDER IS THE ORDER.  Within a stage the branches are alternatives, so "the order the body reaches
// them" is the order the call sites appear: fused, then native, then legacy.  The plan's I2 list
// (`plan/BACKEND-INTEGRATION.md` §3) names the mixer's kernel rows only in the class-B/class-A prose; it is
// NOT an order.  The executed branch under the shipped defaults (native_gdn_enabled() true, g_fused_gdn true,
// native_bf16_projections false) is a SUBSET - `native_gdn_conv_silu`, the two `native_gdn_l2_norm`, then
// `fused_gdn_step_norm` - which is fewer than six symbols and skips the fused/native/legacy siblings this
// batch is meant to cover, so the source order is the reading that yields six.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each symbol below is the thin wrapper already declared in
// `include/strata/kernels/{fused_gdn,gdn,native_gdn_preprocess}.hpp`; this TU answers the `strata::kernels::`
// symbol the wrapper calls.  On a CUDA/HIP build `src/kernels/cuda/{fused_gdn,gdn,native_gdn_preprocess}.cu`
// answer them; on this build THIS file does.  The bodies resolve the engine's raw device pointers to arena
// views (vk_arena.hpp), take the pipeline the device layer caches for the shader's own signature, and dispatch
// - the same shape as `fwht_vk.cpp` and `elementwise_vk.cpp` - and the shader each drives is the one the
// port's numeric gate has already gated:
//
//     fused_gdn_conv_l2   -> fused_gdn_conv_l2.spv   (case_fused_gdn_conv_l2)
//     native_gdn_conv_silu-> native_gdn_conv_silu.spv(case_native_gdn_conv_silu)
//     gdn_conv_step       -> gdn_conv_step.spv       (case_gdn_conv_step)
//     native_gdn_l2_norm  -> native_gdn_l2_norm.spv  (case_native_gdn_l2_norm)
//     gdn_l2_norm         -> gdn_l2_norm.spv         (case_gdn_l2_norm)
//     fused_gdn_ab        -> fused_gdn_ab.spv        (case_fused_gdn_ab)
//
// The gate's `case_*_entry` cases re-run each with the ENGINE WRAPPER and compare BITWISE to that same shader
// path AND against the case's explicit oracle, so this file's claim is not "it compiles" but "the wrapper's
// answer equals the ported shader's answer".
//
// NO `host` ROW IS NEEDED FOR THIS BATCH.  The only `host` row the mixer's six reach is `native_gdn_enabled()`
// (`layer.cpp:247`/`:253`/`:265`/`:295`/`:306`/`:324`), which `vulkan/src/kernels/native_caps_vk.cpp` already
// answers - and the GDN headers carry NO `*_scratch_bytes`-style symbol and no shape accessor (the only shared
// shape, `GdnShapes`, is a by-value POD passed to `gdn_step`, which this batch does not reach).  So the six
// are self-contained; if a later GDN batch (the step/norm pair) pulls one in, that is where it belongs.
#if !defined(STRATA_ENABLE_VULKAN)
#error "gdn_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/fused_gdn.hpp"           // fused_gdn_conv_l2 / fused_gdn_ab
#include "strata/kernels/gdn.hpp"                 // gdn_conv_step / gdn_l2_norm
#include "strata/kernels/native_gdn_preprocess.hpp"  // native_gdn_conv_silu / native_gdn_l2_norm
#include "strata/vulkan/vk_backend.hpp"           // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                           // the arena + pointer->buffer resolution

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

// Every GDN shader is `local_size_x = 256` (checked against the host's constant by the gate's census).
static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// The port's rule: refuse, never degrade.  A pointer outside the arena, or a shape the engine's own wrapper
// contract rejects, is a loud exit rather than a silently wrong binding.
[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

// ---- 1. `fused_gdn_conv_l2` -> fused_gdn_conv_l2.spv (HIST rw, QKV ro, W ro, H rw; push {int channels; int
//        qk_heads; float eps}; one thread per channel).  The engine wrapper's contract (fused_gdn.cu:127-136):
//        `channels` is a multiple of the 128-channel head (`S`), and `qk_heads` selects the first heads that
//        receive the per-head L2 norm.  Those two checks are the CUDA body's; they are kept here (the shader
//        assumes a 256-lane group covers two 128-channel heads).
void fused_gdn_conv_l2(Stream& s, float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps) {
    if (channels <= 0) return;
    if (channels % 128 != 0 || qk_heads < 0 || qk_heads > channels / 128)
        refuse("fused_gdn_conv_l2", "channels is not a multiple of 128 or qk_heads is out of range");
    Buf hv{}, qv{}, wv{}, ov{};
    if (!arena_resolve(s, history, (uint64_t) channels * 3 * 4, hv) ||
        !arena_resolve(s, qkv, (uint64_t) channels * 4, qv) ||
        !arena_resolve(s, conv_w, (uint64_t) channels * 4 * 4, wv) ||
        !arena_resolve(s, h, (uint64_t) channels * 4, ov))
        refuse("fused_gdn_conv_l2", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/fused_gdn_conv_l2.spv", 4, 12);
    struct Push {
        int32_t channels;
        int32_t qk_heads;
        float eps;
    } pc{};
    pc.channels = channels;
    pc.qk_heads = qk_heads;
    pc.eps = eps;
    s.ctx->dispatch(pipe, {&hv, &qv, &wv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) channels));
}

// ---- 2. `native_gdn_conv_silu` -> native_gdn_conv_silu.spv (HIST rw, XI ro, W ro, RAW rw, SILU rw; push
//        {int channels; int d_conv}; one thread per channel).  The CUDA wrapper (native_gdn_preprocess.cu:142)
//        refuses anything but four taps; d_conv is carried to the shader for interface parity even though the
//        body is four taps / three carried rows.
void native_gdn_conv_silu(Stream& s, float* history, const float* input, const float* weights, float* raw_output,
                          float* silu_output, int64_t channels, int64_t d_conv) {
    if (channels <= 0) return;
    if (d_conv != 4) refuse("native_gdn_conv_silu", "the native convolution requires four taps");
    Buf hv{}, iv{}, wv{}, rv{}, sv{};
    if (!arena_resolve(s, history, (uint64_t) channels * 3 * 4, hv) ||
        !arena_resolve(s, input, (uint64_t) channels * 4, iv) ||
        !arena_resolve(s, weights, (uint64_t) channels * 4 * 4, wv) ||
        !arena_resolve(s, raw_output, (uint64_t) channels * 4, rv) ||
        !arena_resolve(s, silu_output, (uint64_t) channels * 4, sv))
        refuse("native_gdn_conv_silu", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_conv_silu.spv", 5, 8);
    struct Push {
        int32_t channels;
        int32_t d_conv;
    } pc{};
    pc.channels = (int32_t) channels;
    pc.d_conv = (int32_t) d_conv;
    s.ctx->dispatch(pipe, {&hv, &iv, &wv, &rv, &sv}, &pc, sizeof(pc), groups_for((uint64_t) channels));
}

// ---- 3. `gdn_conv_step` -> gdn_conv_step.spv (CS rw, XI ro, W ro, OUT rw; push {int channels; int d_conv};
//        one thread per channel).  The carried state is (channels, d_conv-1); d_conv is the shader's own index.
void gdn_conv_step(Stream& s, float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
                   int64_t d_conv) {
    if (channels <= 0 || d_conv < 1) return;
    const uint64_t hist = (uint64_t) channels * (uint64_t) (d_conv - 1);
    Buf cv{}, xv{}, wv{}, ov{};
    if (!arena_resolve(s, conv_state, hist * 4, cv) || !arena_resolve(s, x, (uint64_t) channels * 4, xv) ||
        !arena_resolve(s, kW, (uint64_t) channels * (uint64_t) d_conv * 4, wv) ||
        !arena_resolve(s, out, (uint64_t) channels * 4, ov))
        refuse("gdn_conv_step", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_conv_step.spv", 4, 8);
    struct Push {
        int32_t channels;
        int32_t d_conv;
    } pc{};
    pc.channels = (int32_t) channels;
    pc.d_conv = (int32_t) d_conv;
    s.ctx->dispatch(pipe, {&cv, &xv, &wv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) channels));
}

// ---- 4. `native_gdn_l2_norm` -> native_gdn_l2_norm.spv (X rw; push {int rows; int cols; float eps; float
//        inv_sqrt_cols}; one workgroup per row).  The engine's `epsilon` is the RAW engine epsilon: the shader
//        forms `rsqrt(partial/cols + eps/cols)` itself (native_gdn_preprocess.cu:163 passes `epsilon/S` and
//        `1/sqrt(S)` with S=128, and the port moved the `/cols` into the shader - the gate's case pushes the
//        raw eps and its oracle divides by cols, and this wrapper matches the case bit for bit).
void native_gdn_l2_norm(Stream& s, float* input, int64_t rows, int64_t cols, float epsilon) {
    if (rows <= 0 || cols <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, input, (uint64_t) rows * (uint64_t) cols * 4, xv))
        refuse("native_gdn_l2_norm", "input is not inside this stream's arena");
    const float inv_sqrt_cols = 1.0f / std::sqrt((float) cols);
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_l2_norm.spv", 1, 16);
    struct Push {
        int32_t rows;
        int32_t cols;
        float eps;
        float inv_sqrt_cols;
    } pc{};
    pc.rows = (int32_t) rows;
    pc.cols = (int32_t) cols;
    pc.eps = epsilon;
    pc.inv_sqrt_cols = inv_sqrt_cols;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), (uint32_t) rows);
}

// ---- 5. `gdn_l2_norm` -> gdn_l2_norm.spv (X rw; push {int rows; int cols; float eps}; one workgroup per row).
//        The eps is an ABSOLUTE floor on the SQUARED NORM (gdn.hpp:75-80), passed straight through.
void gdn_l2_norm(Stream& s, float* x, int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, x, (uint64_t) rows * (uint64_t) cols * 4, xv))
        refuse("gdn_l2_norm", "x is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_l2_norm.spv", 1, 12);
    struct Push {
        int32_t rows;
        int32_t cols;
        float eps;
    } pc{};
    pc.rows = (int32_t) rows;
    pc.cols = (int32_t) cols;
    pc.eps = eps;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), (uint32_t) rows);
}

// ---- 6. `fused_gdn_ab` -> fused_gdn_ab.spv (X ro, WA ro uint, WB ro uint, DT ro, SSM_A ro, GATE rw, BETA rw;
//        push {int n; int h_v}; ONE WORKGROUP PER ROW, 2*h_v rows: rows [0,h_v) are beta, [h_v,2h_v) are the
//        gate).  The BF16 weights are bound as 32-bit pairs, as the shader reads them (the CUDA's own pairing:
//        LOW half = element 2p, HIGH half = element 2p+1).  The CUDA wrapper (fused_gdn.cu:140) requires
//        `n_embd % 8 == 0` (the shader strides the activation in 8-element chunks); kept here.
void fused_gdn_ab(Stream& s, const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt,
                  const float* ssm_a, float* gate, float* beta, int n_embd, int h_v) {
    if (n_embd <= 0 || h_v <= 0) return;
    if (n_embd % 8 != 0) refuse("fused_gdn_ab", "n_embd is not a multiple of 8 (the shader's activation chunk)");
    unsigned bad = 0;
    Buf xv{}, av{}, bv{}, dv{}, sv{}, gv{}, ov{};
    bad += !arena_resolve(s, x, (uint64_t) n_embd * 4, xv);
    bad += !arena_resolve(s, w_alpha, (uint64_t) n_embd * (uint64_t) h_v * 2, av);
    bad += !arena_resolve(s, w_beta, (uint64_t) n_embd * (uint64_t) h_v * 2, bv);
    bad += !arena_resolve(s, dt, (uint64_t) h_v * 4, dv);
    bad += !arena_resolve(s, ssm_a, (uint64_t) h_v * 4, sv);
    bad += !arena_resolve(s, gate, (uint64_t) h_v * 4, gv);
    bad += !arena_resolve(s, beta, (uint64_t) h_v * 4, ov);
    if (bad != 0) refuse("fused_gdn_ab", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/fused_gdn_ab.spv", 7, 8);
    struct Push {
        int32_t n;
        int32_t h_v;
    } pc{};
    pc.n = n_embd;
    pc.h_v = h_v;
    s.ctx->dispatch(pipe, {&xv, &av, &bv, &dv, &sv, &gv, &ov}, &pc, sizeof(pc), (uint32_t) (2 * h_v));
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/*.hpp declare --------------------------
namespace strata::kernels {

// fused_gdn.hpp: `void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h,
//                                        int channels, int qk_heads, float eps, void* stream);`  (layer.cpp:250)
void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "fused_gdn_conv_l2: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::fused_gdn_conv_l2(*s, history, qkv, conv_w, h, channels, qk_heads, eps);
}

// native_gdn_preprocess.hpp: `void native_gdn_conv_silu(float* history, const float* input, const float* weights,
//     float* raw_output, float* silu_output, int64_t channels, int64_t d_conv, void* stream);`  (layer.cpp:253)
void native_gdn_conv_silu(float* history, const float* input, const float* weights, float* raw_output,
                          float* silu_output, int64_t channels, int64_t d_conv, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_conv_silu: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_conv_silu(*s, history, input, weights, raw_output, silu_output, channels, d_conv);
}

// gdn.hpp: `void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels,
//                              int64_t d_conv, void* stream);`  (layer.cpp:255)
void gdn_conv_step(float* conv_state, const float* x, const float* kW, float* out, int64_t channels, int64_t d_conv,
                   void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_conv_step: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_conv_step(*s, conv_state, x, kW, out, channels, d_conv);
}

// native_gdn_preprocess.hpp: `void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon,
//                                                      void* stream);`  (layer.cpp:266/267)
void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_l2_norm: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_l2_norm(*s, input, rows, cols, epsilon);
}

// gdn.hpp: `void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream);`  (layer.cpp:269/270)
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_l2_norm: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_l2_norm(*s, x, rows, cols, eps);
}

// fused_gdn.hpp: `void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
//     const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, void* stream);`
//     (layer.cpp:287)
void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "fused_gdn_ab: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::fused_gdn_ab(*s, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v);
}

}  // namespace strata::kernels
