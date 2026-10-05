// vulkan/src/kernels/gdn_vk.cpp - the Vulkan backend's GDN / DeltaNet MIXER entry points.
//
// ============================================================================================================
// WHICH FOURTEEN, AND WHY THESE FOURTEEN (derived from the mixer's own body, not from the plan's list)
// ============================================================================================================
//
// The GDN / DeltaNet mixer is `gdn_layer` (`src/core/layer.cpp:223`), the block every GDN layer runs - 36 of
// the model's 48 layers - reached from `block_layer_pre` (`layer.cpp:1259`: `if (qsa) qsa_layer(...) else
// gdn_layer(...)`).  Its body reaches the GDN kernels in the order below.  I2d wired the FIRST SIX (conv and
// the q/k L2 norms and the fused alpha/beta); I2e wires the REMAINING EIGHT - the beta/gate stage and the
// step/norm stage, including `fused_gdn_step_norm`.
//
// THE FULL ORDER, as `gdn_layer`'s call sites appear (fused -> native -> legacy within each stage; the plan's
// I2 list names the mixer's rows in prose and is NOT an order):
//
//   1. `fused_gdn_conv_l2`     layer.cpp:250   stage 3 (conv), fused (needs `fused_pre`)
//   2. `native_gdn_conv_silu`  layer.cpp:253   stage 3, native (`native_gdn_enabled()`)
//   3. `gdn_conv_step`         layer.cpp:255   stage 3, legacy (its SiLU is the separate `silu_inplace`)
//   4. `native_gdn_l2_norm`    layer.cpp:266/267  stage 4 (q/k L2), native
//   5. `gdn_l2_norm`           layer.cpp:269/270  stage 4, legacy
//   6. `fused_gdn_ab`          layer.cpp:287   stage 5 (alpha/beta), fused
//   7. `native_gdn_beta_gate`  layer.cpp:296   stage 5, native
//   8. `native_gdn_gate`       layer.cpp:297   stage 5, native
//   9. `gdn_beta_gate`         layer.cpp:299   stage 5, legacy
//  10. `native_gdn_step`       layer.cpp:308   stage 6 (recurrence), native (folds the decay + readout scale)
//  11. `gdn_step`              layer.cpp:309   stage 6, legacy (decay-first; the scale is a separate launch)
//  12. `fused_gdn_step_norm`   layer.cpp:322   stage 7 (z-gate/norm), fused (needs `fused_gdn`)
//  13. `native_gdn_out_norm`   layer.cpp:324   stage 7, native
//  14. `gdn_out_norm`          layer.cpp:325   stage 7, legacy
//
// WHY SOURCE ORDER IS THE ORDER.  Within a stage the branches are alternatives, so "the order the body reaches
// them" is the order the call sites appear: fused, then native, then legacy.  The executed branch under the
// shipped defaults (native_gdn_enabled() true, g_fused_gdn true, native_bf16_projections false) is a SUBSET -
// `native_gdn_conv_silu`, the two `native_gdn_l2_norm`, `native_gdn_beta_gate`, `native_gdn_gate`,
// `native_gdn_step`, `fused_gdn_step_norm` - which skips the fused/native/legacy siblings this file is meant
// to cover, so the source order is the reading that yields all fourteen.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each symbol below is the thin wrapper already declared in
// `include/strata/kernels/{fused_gdn,gdn,native_gdn,native_gdn_preprocess}.hpp`; this TU answers the
// `strata::kernels::` symbol the wrapper calls.  On a CUDA/HIP build `src/kernels/cuda/*.cu` answer them; on
// this build THIS file does.  The bodies resolve the engine's raw device pointers to arena views
// (vk_arena.hpp), take the pipeline the device layer caches for the shader's own signature, and dispatch -
// the same shape as `fwht_vk.cpp` and `elementwise_vk.cpp` - and the shader each drives is the one the port's
// numeric gate has already gated:
//
//     fused_gdn_conv_l2   -> fused_gdn_conv_l2.spv   (case_fused_gdn_conv_l2)
//     native_gdn_conv_silu-> native_gdn_conv_silu.spv(case_native_gdn_conv_silu)
//     gdn_conv_step       -> gdn_conv_step.spv       (case_gdn_conv_step)
//     native_gdn_l2_norm  -> native_gdn_l2_norm.spv  (case_native_gdn_l2_norm)
//     gdn_l2_norm         -> gdn_l2_norm.spv         (case_gdn_l2_norm)
//     fused_gdn_ab        -> fused_gdn_ab.spv        (case_fused_gdn_ab)
//     native_gdn_beta_gate-> native_gdn_beta_gate.spv(case_native_gdn_beta_gate)
//     native_gdn_gate     -> native_gdn_gate.spv     (case_native_gdn_gate)
//     gdn_beta_gate       -> gdn_beta_gate.spv       (case_gdn_beta_gate)
//     native_gdn_step     -> native_gdn_step.spv     (case_native_gdn_step)
//     gdn_step            -> gdn_step.spv            (case_gdn_step)
//     fused_gdn_step_norm -> fused_gdn_step_norm.spv (case_fused_gdn_step_norm)
//     native_gdn_out_norm -> native_gdn_out_norm.spv (case_native_gdn_out_norm)
//     gdn_out_norm        -> gdn_out_norm.spv        (case_gdn_out_norm)
//
// (`gdn_gate`, layer.cpp:300, is the SEVENTH symbol of the beta/gate stage and is answered in
// `elementwise_vk.cpp` with the other glue kernels, so it is not repeated here.)
//
// The gate's `case_*_entry` cases re-run each with the ENGINE WRAPPER and compare BITWISE to that same shader
// path AND against the case's explicit oracle, so this file's claim is not "it compiles" but "the wrapper's
// answer equals the ported shader's answer".
//
// NO `host` ROW IS NEEDED FOR THIS FILE, AND THIS IS THE CHECK.  The only `host` row the mixer reaches is
// `native_gdn_enabled()` (layer.cpp:247/253/265/295/306/324), which `vulkan/src/kernels/native_caps_vk.cpp`
// already answers (TRUE - every gated symbol now has a shader).  The GDN headers carry NO `*_scratch_bytes`-
// style symbol and no shape ACCESSOR: the only shared shape, `GdnShapes`, is a by-value POD passed to
// `gdn_step`/`native_gdn_step` (a struct, not a symbol to link), and the FUSED `fused_gdn_step_norm` takes the
// head counts as ints and fixes S=128 internally (fused_gdn.cu's `o = (S^T q)/sqrt(128)`).  This is verified
// rather than asserted: the I2e link measurement in `ports/vulkan/NEXT.md` drops by exactly the eight symbols
// this file adds and introduces NO new undefined reference.
#if !defined(STRATA_ENABLE_VULKAN)
#error "gdn_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/fused_gdn.hpp"               // fused_gdn_conv_l2 / fused_gdn_ab / fused_gdn_step_norm
#include "strata/kernels/gdn.hpp"                     // gdn_conv_step / gdn_l2_norm / gdn_beta_gate / gdn_step /
                                                      //   gdn_out_norm / GdnShapes
#include "strata/kernels/native_gdn.hpp"              // native_gdn_step
#include "strata/kernels/native_gdn_preprocess.hpp"   // native_gdn_conv_silu / native_gdn_l2_norm /
                                                      //   native_gdn_beta_gate / native_gdn_gate / native_gdn_out_norm
#include "strata/vulkan/vk_backend.hpp"               // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                               // the arena + pointer->buffer resolution

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

// ============================================================================================================
// I2e - THE REMAINING EIGHT: the beta/gate stage (#7-#9) and the step/norm stage (#10-#14, `gdn_gate` at
// layer.cpp:300 being the seventh of the beta/gate stage and answered in elementwise_vk.cpp).
// ============================================================================================================

// ---- 7. `native_gdn_beta_gate` -> native_gdn_beta_gate.spv (B rw; push {int n}; one thread per head).  In
//        place `beta = sigmoid(beta)`; the native body (native_gdn_preprocess.cu's `beta_sigmoid`) is the SAME
//        expression as the legacy `gdn_beta_gate` (#9), and the layer needs the fraction before `gdn_step`.
void native_gdn_beta_gate(Stream& s, float* beta, int64_t heads) {
    if (heads <= 0) return;
    Buf bv{};
    if (!arena_resolve(s, beta, (uint64_t) heads * 4, bv))
        refuse("native_gdn_beta_gate", "beta is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_beta_gate.spv", 1, 4);
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) heads;
    s.ctx->dispatch(pipe, {&bv}, &pc, sizeof(pc), groups_for((uint64_t) heads));
}

// ---- 8. `native_gdn_gate` -> native_gdn_gate.spv (ALPHA ro, DT ro, SSMA ro, GATE rw; push {int heads}; one
//        thread per head).  `gate[h] = softplus(alpha[h] + dt[h]) * ssm_a[h]`, the threshold-20 branch
//        (native_gdn_preprocess.cu's `gate_softplus`).  Per-head, one token - the layer calls it with
//        `heads = ssm_v_heads` (layer.cpp:297).
void native_gdn_gate(Stream& s, const float* alpha, const float* dt, const float* ssm_a, float* gate,
                     int64_t heads) {
    if (heads <= 0) return;
    Buf av{}, dv{}, sv{}, gv{};
    if (!arena_resolve(s, alpha, (uint64_t) heads * 4, av) || !arena_resolve(s, dt, (uint64_t) heads * 4, dv) ||
        !arena_resolve(s, ssm_a, (uint64_t) heads * 4, sv) || !arena_resolve(s, gate, (uint64_t) heads * 4, gv))
        refuse("native_gdn_gate", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_gate.spv", 4, 4);
    struct Push {
        int32_t heads;
    } pc{};
    pc.heads = (int32_t) heads;
    s.ctx->dispatch(pipe, {&av, &dv, &sv, &gv}, &pc, sizeof(pc), groups_for((uint64_t) heads));
}

// ---- 9. `gdn_beta_gate` -> gdn_beta_gate.spv (B rw; push {int n}; one thread per head).  The LEGACY sibling
//        of #7; the same in-place sigmoid (`sigmoid_f`).
void gdn_beta_gate(Stream& s, float* beta, int64_t h_v) {
    if (h_v <= 0) return;
    Buf bv{};
    if (!arena_resolve(s, beta, (uint64_t) h_v * 4, bv))
        refuse("gdn_beta_gate", "beta is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_beta_gate.spv", 1, 4);
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) h_v;
    s.ctx->dispatch(pipe, {&bv}, &pc, sizeof(pc), groups_for((uint64_t) h_v));
}

// ---- 10. `native_gdn_step` -> native_gdn_step.spv (STATE rw, Q ro, K ro, V ro, G ro, B ro, O rw; push
//         {int S; int h_k; int h_v; float scale}; one thread per (h,j) column).  The native body folds the
//         decay into the rank-1 update and FUSES the `1/sqrt(S)` readout scale (which the legacy branch
//         applies in a separate `scale_inplace`, layer.cpp:276), taking the `sk` contract against the
//         UNDECAYED state.  The wrapper requires S == 128 and h_v % h_k == 0 (native_gdn.hpp:11-13).
void native_gdn_step(Stream& s, float* state, const float* q, const float* k, const float* v, const float* gate,
                     const float* beta, float* output, const strata::kernels::GdnShapes& sh) {
    if (sh.S <= 0 || sh.h_k <= 0 || sh.h_v <= 0) return;
    if (sh.S != 128) refuse("native_gdn_step", "the native recurrence requires S == 128");
    if (sh.h_v % sh.h_k != 0) refuse("native_gdn_step", "h_v is not a multiple of h_k");
    const uint64_t S = (uint64_t) sh.S, hk = (uint64_t) sh.h_k, hv = (uint64_t) sh.h_v;
    Buf st{}, qb{}, kb{}, vb{}, gb{}, bb{}, ob{};
    if (!arena_resolve(s, state, S * hv * S * 4, st) || !arena_resolve(s, q, hk * S * 4, qb) ||
        !arena_resolve(s, k, hk * S * 4, kb) || !arena_resolve(s, v, hv * S * 4, vb) ||
        !arena_resolve(s, gate, hv * 4, gb) || !arena_resolve(s, beta, hv * 4, bb) ||
        !arena_resolve(s, output, hv * S * 4, ob))
        refuse("native_gdn_step", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_step.spv", 7, 16);
    struct Push {
        int32_t S;
        int32_t h_k;
        int32_t h_v;
        float scale;
    } pc{};
    pc.S = (int32_t) sh.S;
    pc.h_k = (int32_t) sh.h_k;
    pc.h_v = (int32_t) sh.h_v;
    pc.scale = 1.0f / std::sqrt((float) sh.S);   // the folded readout scale
    s.ctx->dispatch(pipe, {&st, &qb, &kb, &vb, &gb, &bb, &ob}, &pc, sizeof(pc), groups_for(hv * S));
}

// ---- 11. `gdn_step` -> gdn_step.spv (STATE rw, Q ro, K ro, V ro, G ro, B ro, O rw; push {int S; int h_k;
//         int h_v}; one thread per (h,j) column).  The LEGACY sibling of #10: the decay is applied FIRST and
//         the readout scale is NOT applied here (the layer's separate `scale_inplace` carries it).  The head
//         pairing is `src = h % h_k`, so `h_v % h_k == 0` is required for the pairing to cover each head
//         exactly; a geometry that violated it would be a silent wrong answer, so it is a loud refusal.
void gdn_step(Stream& s, float* state, const float* q, const float* k, const float* v, const float* gate,
              const float* beta, float* o, const strata::kernels::GdnShapes& sh) {
    if (sh.S <= 0 || sh.h_k <= 0 || sh.h_v <= 0) return;
    if (sh.h_v % sh.h_k != 0) refuse("gdn_step", "h_v is not a multiple of h_k");
    const uint64_t S = (uint64_t) sh.S, hk = (uint64_t) sh.h_k, hv = (uint64_t) sh.h_v;
    Buf st{}, qb{}, kb{}, vb{}, gb{}, bb{}, ob{};
    if (!arena_resolve(s, state, S * hv * S * 4, st) || !arena_resolve(s, q, hk * S * 4, qb) ||
        !arena_resolve(s, k, hk * S * 4, kb) || !arena_resolve(s, v, hv * S * 4, vb) ||
        !arena_resolve(s, gate, hv * 4, gb) || !arena_resolve(s, beta, hv * 4, bb) ||
        !arena_resolve(s, o, hv * S * 4, ob))
        refuse("gdn_step", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_step.spv", 7, 12);
    struct Push {
        int32_t S;
        int32_t h_k;
        int32_t h_v;
    } pc{};
    pc.S = (int32_t) sh.S;
    pc.h_k = (int32_t) sh.h_k;
    pc.h_v = (int32_t) sh.h_v;
    s.ctx->dispatch(pipe, {&st, &qb, &kb, &vb, &gb, &bb, &ob}, &pc, sizeof(pc), groups_for(hv * S));
}

// ---- 12. `fused_gdn_step_norm` -> fused_gdn_step_norm.spv (STATE rw, Q ro, K ro, V ro, G ro, B ro, Z ro,
//         GAMMA ro, Y rw; push {int S; int h_k; int h_v; float eps}; one thread per (h,j) column).  The fused
//         step+closing-norm (fused_gdn.cu's `gdn_step_norm_kernel`); S is FIXED at 128 (the fused operator's
//         `o = (S^T q)/sqrt(128)`), so the wrapper takes only the head counts and derives S here.
void fused_gdn_step_norm(Stream& s, float* state, const float* q, const float* k, const float* v,
                         const float* gate, const float* beta, const float* z, const float* gamma, float eps,
                         float* y, int h_k, int h_v) {
    if (h_k <= 0 || h_v <= 0) return;
    if (h_v % h_k != 0) refuse("fused_gdn_step_norm", "h_v is not a multiple of h_k");
    const uint64_t S = 128, hk = (uint64_t) h_k, hv = (uint64_t) h_v;
    Buf st{}, qb{}, kb{}, vb{}, gb{}, bb{}, zb{}, gmb{}, yb{};
    if (!arena_resolve(s, state, S * hv * S * 4, st) || !arena_resolve(s, q, hk * S * 4, qb) ||
        !arena_resolve(s, k, hk * S * 4, kb) || !arena_resolve(s, v, hv * S * 4, vb) ||
        !arena_resolve(s, gate, hv * 4, gb) || !arena_resolve(s, beta, hv * 4, bb) ||
        !arena_resolve(s, z, hv * S * 4, zb) || !arena_resolve(s, gamma, S * 4, gmb) ||
        !arena_resolve(s, y, hv * S * 4, yb))
        refuse("fused_gdn_step_norm", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/fused_gdn_step_norm.spv", 9, 16);
    struct Push {
        int32_t S;
        int32_t h_k;
        int32_t h_v;
        float eps;
    } pc{};
    pc.S = 128;
    pc.h_k = h_k;
    pc.h_v = h_v;
    pc.eps = eps;
    s.ctx->dispatch(pipe, {&st, &qb, &kb, &vb, &gb, &bb, &zb, &gmb, &yb}, &pc, sizeof(pc), groups_for(hv * S));
}

// ---- 13. `native_gdn_out_norm` -> native_gdn_out_norm.spv (O ro, Z ro, G ro, Y rw; push {int heads; int S;
//         float eps}; ONE WORKGROUP PER ROW).  `y = rms_norm(o, eps) * gamma * sigmoid(z)`, eps on the MEAN;
//         the native wrapper requires cols == 128 (gamma is 128 floats, native_gdn_preprocess.hpp:31-35).
void native_gdn_out_norm(Stream& s, const float* output, const float* z, const float* gamma, float* destination,
                         int64_t heads, int64_t cols, float epsilon) {
    if (heads <= 0 || cols <= 0) return;
    if (cols != 128) refuse("native_gdn_out_norm", "the native wrapper requires cols == 128 (gamma is 128 floats)");
    const uint64_t n = (uint64_t) heads * (uint64_t) cols;
    Buf ob{}, zb{}, gb{}, yb{};
    if (!arena_resolve(s, output, n * 4, ob) || !arena_resolve(s, z, n * 4, zb) ||
        !arena_resolve(s, gamma, (uint64_t) cols * 4, gb) || !arena_resolve(s, destination, n * 4, yb))
        refuse("native_gdn_out_norm", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gdn_out_norm.spv", 4, 12);
    struct Push {
        int32_t heads;
        int32_t S;
        float eps;
    } pc{};
    pc.heads = (int32_t) heads;
    pc.S = (int32_t) cols;
    pc.eps = epsilon;
    s.ctx->dispatch(pipe, {&ob, &zb, &gb, &yb}, &pc, sizeof(pc), (uint32_t) heads);
}

// ---- 14. `gdn_out_norm` -> gdn_out_norm.spv (O ro, Z ro, SN ro, Y rw; push {int h_v; int S; float eps}; one
//         workgroup per row).  The LEGACY sibling of #13: the SAME expression, but eps on the MEAN and the
//         multiply order `(o*inv)*ssm_norm*sigmoid(z)` as the legacy CUDA body (gdn.cu:132-149).  There is no
//         `cols == 128` restriction here (the legacy case runs S=16/8 too).
void gdn_out_norm(Stream& s, const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v,
                  int64_t S, float eps) {
    if (h_v <= 0 || S <= 0) return;
    const uint64_t n = (uint64_t) h_v * (uint64_t) S;
    Buf ob{}, zb{}, sb{}, yb{};
    if (!arena_resolve(s, o, n * 4, ob) || !arena_resolve(s, z, n * 4, zb) ||
        !arena_resolve(s, ssm_norm, (uint64_t) S * 4, sb) || !arena_resolve(s, y, n * 4, yb))
        refuse("gdn_out_norm", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_out_norm.spv", 4, 12);
    struct Push {
        int32_t h_v;
        int32_t S;
        float eps;
    } pc{};
    pc.h_v = (int32_t) h_v;
    pc.S = (int32_t) S;
    pc.eps = eps;
    s.ctx->dispatch(pipe, {&ob, &zb, &sb, &yb}, &pc, sizeof(pc), (uint32_t) h_v);
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

// ------------------------------------------------------------------------------------------------------------
// I2e: the remaining eight.  Each is a bare bind-and-dispatch over the shader its gate case already proved; NONE
// reads module state (unlike `cvec_apply`, which needed the `cvec` host row).  `native_gdn_step` / `gdn_step`
// carry the by-value `GdnShapes` POD; `native_gdn_step` derives the folded readout scale, `fused_gdn_step_norm`
// fixes S=128 - those are the only arithmetic the wrappers add.
// ------------------------------------------------------------------------------------------------------------

// native_gdn_preprocess.hpp: `void native_gdn_beta_gate(float* beta, int64_t heads, void* stream);` (layer.cpp:296)
void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_beta_gate: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_beta_gate(*s, beta, heads);
}

// native_gdn_preprocess.hpp: `void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a,
//     float* gate, int64_t heads, void* stream);` (layer.cpp:297)
void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads,
                     void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_gate: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_gate(*s, alpha, dt, ssm_a, gate, heads);
}

// gdn.hpp: `void gdn_beta_gate(float* beta, int64_t h_v, void* stream);` (layer.cpp:299)
void gdn_beta_gate(float* beta, int64_t h_v, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_beta_gate: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_beta_gate(*s, beta, h_v);
}

// native_gdn.hpp: `void native_gdn_step(float* state, const float* q, const float* k, const float* v,
//     const float* gate, const float* beta, float* output, const GdnShapes& shapes, void* stream);`
//     (layer.cpp:308)
void native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
                     const float* beta, float* output, const GdnShapes& shapes, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_step: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_step(*s, state, q, k, v, gate, beta, output, shapes);
}

// gdn.hpp: `void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
//     const float* beta, float* o, const GdnShapes& s, void* stream);` (layer.cpp:309)
void gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate, const float* beta,
              float* o, const GdnShapes& shapes, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_step: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_step(*s, state, q, k, v, gate, beta, o, shapes);
}

// fused_gdn.hpp: `void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v,
//     const float* gate, const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
//     int h_v, void* stream);` (layer.cpp:322)
void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k,
                         int h_v, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "fused_gdn_step_norm: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::fused_gdn_step_norm(*s, state, q, k, v, gate, beta, z, gamma, eps, y, h_k, h_v);
}

// native_gdn_preprocess.hpp: `void native_gdn_out_norm(const float* output, const float* z, const float* gamma,
//     float* destination, int64_t heads, int64_t cols, float epsilon, void* stream);` (layer.cpp:324)
void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination, int64_t heads,
                         int64_t cols, float epsilon, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "native_gdn_out_norm: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::native_gdn_out_norm(*s, output, z, gamma, destination, heads, cols, epsilon);
}

// gdn.hpp: `void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v,
//     int64_t S, float eps, void* stream);` (layer.cpp:325)
void gdn_out_norm(const float* o, const float* z, const float* ssm_norm, float* y, int64_t h_v, int64_t S,
                  float eps, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_out_norm: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_out_norm(*s, o, z, ssm_norm, y, h_v, S, eps);
}

}  // namespace strata::kernels
