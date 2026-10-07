// vulkan/src/kernels/fused_gr_vk.cpp - the Vulkan backend's FUSED hyper-connection read
// (`fused_gr_read` / `fused_gr_read_multi`).
//
// ============================================================================================================
// WHY THIS EXISTS, AND WHY IT IS NOT OPTIONAL
// ============================================================================================================
//
// `fused_gr_supported(n_embd, hc, hc_lr)` selects the hyper-connection read on BOTH the decode path
// (`layer.cpp:1188` `fused = g_fused_gr && fused_gr_supported(...)`, `:1328`) AND the P6 verify window
// (`verify.cpp:336`).  The verify window is a NATIVE PACK'S ONLY DECODE PATH (`generate.cpp:7578-7579` breaks
// the token loop for a native pack), and its per-layer GR read calls `fused_gr_read_multi` UNCONDITIONALLY
// (`verify.cpp:693`, `:1142`).  So a native pack's first token REQUIRES this read: answering
// `fused_gr_supported` TRUE without it would move the refusal from the verify window's door to its per-layer
// body, and answering FALSE (the previous batch) refuses the window at `verifiy.cpp:336`'s first disjunct.
//
// ============================================================================================================
// WHAT THE FUSED READ IS (read from src/kernels/cuda/fused_gr.cu, not inferred)
// ============================================================================================================
//
// `fused_gr_down` (fused_gr.cu:49-105) + `fused_gr_up` (:107-138) compute, for one token:
//
//     gw[c] = apply ? 2*sigmoid(inj_prev[c]/hc) : 0                  (the fold's gate)
//     R'[i] = R[i] + bo_prev[d]*gw[c]                                (i = c*n_embd+d; the previous half's write)
//     ss[c] = sum_d R'[c][d]^2 ;  rs[c] = rsqrt(ss[c]/n_embd + eps)  (RMS, PER STREAM)
//     xn[i] = R'[i] * w_norm[i] * rs[c]
//     lo[k] = silu( (w_down[k] . xn) / hc )                          (k < hc_lr; `/hc` INSIDE the silu)
//     inject[j] = w_inject[j] . xn                                   (only when w_inject != null)
//     mixed[d] = mean_c xn[c][d] * sigmoid( w_up[c*n_embd+d] . lo )
//     R_out[i] = R'[i]                                               (only when apply)
//
// THE DIFFERENCE FROM THE UNFUSED `gr_read` (`gr.cu`) IS TWO THINGS, and neither is cosmetic:
//   1. THE WRITE IS FOLDED IN.  Unfused: `gr_write` materialises R' into R, then `gr_read` reads it.  Fused:
//      R' is computed on the way into the norm (`fused_gr_rs.comp`).  Same arithmetic, one fewer pass.
//   2. FP32 ACTIVATIONS.  The fused read's declared contract is "FP32 activations and BF16 weights"
//      (`fused_gr.hpp:15`), so `xn` and `lo` are UNROUNDED: `gr_down_kernel`'s `dot8(w4, xn)` reads the f32
//      `xn`, and `a.lo[row] = x/(1+exp(-x))` stores f32.  This is `gr.cu`'s `gr_norm_kernel<true>` /
//      `gr_down_kernel<float>` / `gr_gate_kernel<float>` / `gr_inject_kernel<float>` branch (`gr.cu:373-395`),
//      NOT its default `<uint16_t>` branch.  So the fused read is NOT bitwise-equal to the port's unfused
//      `gr_read` (which rounds the activations to bf16, `gr_set_fp32_activations(false)`); the gate case says
//      so and measures the two apart rather than asserting an equality that does not hold.
//
// ============================================================================================================
// HOW IT IS PORTED (the standing preference: ONE mechanism, reuse where the arithmetic is the same)
// ============================================================================================================
//
// FOUR dispatches per token, all `local_size_x = 256`, all reductions the port's one barrier tree
// (`common/wg_reduce.glsl`), all BF16 weights widened by the port's one `common/bf16.glsl`:
//
//     fused_gr_rs.spv      fold + per-stream RMS + `rs`        (hc workgroups; writes R_out when apply)
//     fused_gr_down.spv    lo[k] = silu((w_down[k] . xn)/hc)   (hc_lr workgroups)
//     fused_gr_mix.spv     mixed[d] = mean_c xn*sigmoid(...)   (n_embd workgroups; gate+mean FUSED, as the CUDA)
//     fused_gr_inject.spv  inject[c] = w_inject[c] . xn        (hc workgroups; skipped when w_inject == null)
//
// THE ACTIVATION IS NOT MATERIALISED.  The fused read's argument set has NO buffer for `xn` (the CUDA keeps it
// in the down kernel's SHARED memory, 40 KB at the artifact's geometry, and the up kernel RECOMPUTES it as
// `rv * w_norm[i] * rs[c]`).  This port follows the CUDA's recompute: every consumer derives `xn` from `R_out`,
// `w_norm` and `rs`.  That is what keeps this read inside the engine's argument set - no hidden allocation, no
// dependence on `xn_scratch` (which the header describes as the CUDA multi's staging; a caller may pass null).
//
// GEOMETRY IS THE ARTIFACT'S AND IS COMPILE-TIME IN THE REFERENCE: `fused_gr.cu:20-23` hard-codes
// N=2560 / HC=4 / LR=320, which is exactly what `fused_gr_supported` tests.  This TU carries the same three
// constants, so the shaders' push constants are always the artifact's numbers.
//
// `fused_gr_read_multi` IS A LOOP OVER `fused_gr_read`'s four dispatches, one token at a time.  The header's
// correctness contract is "every token's outputs are bitwise `fused_gr_read(a[t])`" (`fused_gr.hpp:47`); a loop
// makes that true BY CONSTRUCTION rather than by a shared-weight kernel that must reproduce the single-token
// summation order.  The CUDA's single weight read for all tokens is a COST optimisation, and it is the one
// property this port does not carry.
#if !defined(STRATA_ENABLE_VULKAN)
#error "fused_gr_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/fused_gr.hpp"        // FusedGrArgs / fused_gr_read / fused_gr_read_multi
#include "strata/kernels/verify_kernels.hpp"  // gpu_stamp (the verifier's stage profiler)
#include "strata/vulkan/vk_backend.hpp"       // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                       // the arena + pointer->buffer resolution

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

// The artifact's geometry, the same three numbers `fused_gr_supported` tests (`fused_gr.cu:20-23`).
static constexpr int64_t FG_N = 2560, FG_HC = 4, FG_LR = 320;

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

// A Vulkan descriptor cannot be null: `apply == false` binds the fold's BO/INJ inputs and the final mixer binds
// its WI/INJ pair even though neither is read.  The sentinel lives with the stream (a per-dispatch allocation
// would exhaust the arena) - the `ple_vk.cpp` / `qsa_vk.cpp` precedent.
static Buf& dummy_buf(Stream& s) {
    if (s.dummy.buffer == VK_NULL_HANDLE) {
        s.dummy = s.ctx->alloc(64);
        const int32_t zero[16] = {0};
        s.ctx->write(s.dummy, zero, sizeof(zero));
    }
    return s.dummy;
}

// ONE TOKEN, four dispatches.  This is the whole read; the multi is a loop over it.
static void fused_gr_read_one(Stream& s, const strata::kernels::FusedGrArgs& a) {
    const int64_t n_embd = FG_N, hc = FG_HC, hc_lr = FG_LR;
    const int64_t hc_dim = hc * n_embd;
    if ((n_embd % 2) != 0 || (hc_lr % 2) != 0)
        refuse("fused_gr_read", "the artifact's geometry is odd, which these bf16-pair dots cannot address");
    if (!a.R || !a.R_out || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed)
        refuse("fused_gr_read", "a required pointer is null");
    if (a.apply && (!a.bo_prev || !a.inj_prev))
        refuse("fused_gr_read", "apply is set but bo_prev/inj_prev is null");
    if (a.w_inject != nullptr && a.inject_out == nullptr)
        refuse("fused_gr_read", "w_inject is set but inject_out is null");

    Buf rv{}, ov{}, nv{}, sv{}, dv{}, uv{}, lv{}, mv{}, bv{}, iv{}, wiv{}, jv{};
    if (!arena_resolve(s, a.R, (uint64_t) hc_dim * 4, rv) ||
        !arena_resolve(s, a.R_out, (uint64_t) hc_dim * 4, ov) ||
        !arena_resolve(s, a.w_norm, (uint64_t) hc_dim * 4, nv) ||
        !arena_resolve(s, a.w_down, (uint64_t) hc_lr * (uint64_t) (hc_dim / 2) * 4, dv) ||
        !arena_resolve(s, a.w_up, (uint64_t) hc_dim * (uint64_t) (hc_lr / 2) * 4, uv) ||
        !arena_resolve(s, a.lo, (uint64_t) hc_lr * 4, lv) ||
        !arena_resolve(s, a.rs, (uint64_t) hc * 4, sv) ||
        !arena_resolve(s, a.mixed, (uint64_t) n_embd * 4, mv))
        refuse("fused_gr_read", "a pointer is not inside this stream's arena");
    // THE FOLDED RESIDUAL THE CONSUMERS READ.  When `apply`, `fused_gr_rs` writes R' into `a.R_out` and every
    // later stage reads THAT; when not, `R_out` is left untouched (the CUDA writes it only inside
    // `if (a.apply)`) and the stages must read `a.R`, which IS R'.  The engine passes R_out == R, so the two
    // agree there - but a caller that passes a distinct R_out would otherwise have the down projection read
    // whatever the untouched output buffer happens to hold.  Bound explicitly, not assumed.
    Buf rf = a.apply ? ov : rv;
    if (a.apply) {
        if (!arena_resolve(s, a.bo_prev, (uint64_t) n_embd * 4, bv) ||
            !arena_resolve(s, a.inj_prev, (uint64_t) hc * 4, iv))
            refuse("fused_gr_read", "the fold's bo_prev/inj_prev is not inside this stream's arena");
    } else {
        bv = iv = dummy_buf(s);   // bound but never read (apply == false)
    }
    if (a.w_inject != nullptr) {
        if (!arena_resolve(s, a.w_inject, (uint64_t) hc * (uint64_t) (hc_dim / 2) * 4, wiv) ||
            !arena_resolve(s, a.inject_out, (uint64_t) hc * 4, jv))
            refuse("fused_gr_read", "the inject pair is not inside this stream's arena");
    } else {
        wiv = jv = dummy_buf(s);
    }

    // ---- 1. fold + per-stream RMS -> R_out (when apply) and rs ----
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_rs.spv", 5, 20);
        struct { int32_t n_embd, hc; float eps; int32_t apply; } pc{(int32_t) n_embd, (int32_t) hc, a.eps,
                                                                   a.apply ? 1 : 0};
        s.ctx->dispatch(p, {&rv, &bv, &iv, &ov, &sv}, &pc, sizeof(pc), (uint32_t) hc);
    }
    // ---- 2. down: lo[k] = silu((w_down[k] . xn) / hc), xn = R_out*w_norm*rs ----
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_down.spv", 5, 12);
        struct { int32_t n_embd, hc, hc_lr; } pc{(int32_t) n_embd, (int32_t) hc, (int32_t) hc_lr};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &dv, &lv}, &pc, sizeof(pc), (uint32_t) hc_lr);
    }
    // ---- 3. gate + mean: mixed[d] = mean_c xn*sigmoid(w_up . lo) ----
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_mix.spv", 6, 12);
        struct { int32_t n_embd, hc, hc_lr; } pc{(int32_t) n_embd, (int32_t) hc, (int32_t) hc_lr};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &lv, &uv, &mv}, &pc, sizeof(pc), (uint32_t) n_embd);
    }
    // ---- 4. inject (SKIPPED for the final mixer: w_inject == null leaves the caller's buffer untouched) ----
    if (a.w_inject != nullptr) {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_inject.spv", 5, 8);
        struct { int32_t n_embd, hc; } pc{(int32_t) n_embd, (int32_t) hc};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &wiv, &jv}, &pc, sizeof(pc), (uint32_t) hc);
    }
}

// ---- THE TOKEN-BATCHED MULTI: the dispatch-count reduction on the RECORDED decode chain ---------------------
// The multi loops tokens and issues FOUR dispatches per token.  On the recorded decode arm each dispatch is a
// SERIAL LINK, and the chain's cost is the per-link DEVICE latency (measured: ~114 us of the ~120 us window per
// dispatch, against a 5.5 us kernel - `vk kernel time SPLIT [PAIR]`), so FOUR links for n_tok tokens instead of
// 4*n_tok is the lever.  Every (token, workgroup) of every stage writes DISJOINT elements from SHARED weights,
// so the whole loop is ONE grid whose y dimension is the token: BITWISE IDENTICAL by construction (the shaders
// index the per-token buffers by `t = gl_WorkGroupID.y` and `t == 0` on the single read).  Taken ONLY when the
// per-token buffers are contiguous with the strides the shaders already compute - VERIFIED here, never assumed;
// otherwise the loop runs unchanged.  `STRATA_FGR_TOKEN_BATCH=0` restores the loop (the control).
static bool fused_gr_batch_ok(const strata::kernels::FusedGrArgs* a, int n_tok) {
    const uint64_t sR = (uint64_t) FG_HC * (uint64_t) FG_N, sHC = (uint64_t) FG_HC, sN = (uint64_t) FG_N,
                     sLo = (uint64_t) FG_LR;
    for (int t = 1; t < n_tok; ++t) {
        const strata::kernels::FusedGrArgs& x = a[t];
        if ((uintptr_t) x.R != (uintptr_t) a[0].R + (uint64_t) t * sR * 4) return false;
        if ((uintptr_t) x.R_out != (uintptr_t) a[0].R_out + (uint64_t) t * sR * 4) return false;
        if ((uintptr_t) x.bo_prev != (uintptr_t) a[0].bo_prev + (uint64_t) t * sN * 4) return false;
        if ((uintptr_t) x.inj_prev != (uintptr_t) a[0].inj_prev + (uint64_t) t * sHC * 4) return false;
        if ((uintptr_t) x.lo != (uintptr_t) a[0].lo + (uint64_t) t * sLo * 4) return false;
        if ((uintptr_t) x.rs != (uintptr_t) a[0].rs + (uint64_t) t * sHC * 4) return false;
        if ((uintptr_t) x.mixed != (uintptr_t) a[0].mixed + (uint64_t) t * sN * 4) return false;
        if (a[0].inject_out != nullptr &&
            (uintptr_t) x.inject_out != (uintptr_t) a[0].inject_out + (uint64_t) t * sHC * 4) return false;
    }
    return true;
}

static void fused_gr_read_batch(Stream& s, const strata::kernels::FusedGrArgs* a, int n_tok) {
    const int64_t n_embd = FG_N, hc = FG_HC, hc_lr = FG_LR;
    const int64_t hc_dim = hc * n_embd;
    const uint64_t nt = (uint64_t) n_tok;
    const strata::kernels::FusedGrArgs& a0 = a[0];
    if ((n_embd % 2) != 0 || (hc_lr % 2) != 0)
        refuse("fused_gr_read_batch", "the artifact's geometry is odd, which these bf16-pair dots cannot address");
    Buf rv{}, ov{}, nv{}, sv{}, dv{}, uv{}, lv{}, mv{}, bv{}, iv{}, wiv{}, jv{};
    if (!arena_resolve(s, a0.R, (uint64_t) hc_dim * 4 * nt, rv) ||
        !arena_resolve(s, a0.R_out, (uint64_t) hc_dim * 4 * nt, ov) ||
        !arena_resolve(s, a0.w_norm, (uint64_t) hc_dim * 4, nv) ||
        !arena_resolve(s, a0.w_down, (uint64_t) hc_lr * (uint64_t) (hc_dim / 2) * 4, dv) ||
        !arena_resolve(s, a0.w_up, (uint64_t) hc_dim * (uint64_t) (hc_lr / 2) * 4, uv) ||
        !arena_resolve(s, a0.lo, (uint64_t) hc_lr * 4 * nt, lv) ||
        !arena_resolve(s, a0.rs, (uint64_t) hc * 4 * nt, sv) ||
        !arena_resolve(s, a0.mixed, (uint64_t) n_embd * 4 * nt, mv))
        refuse("fused_gr_read_batch", "a pointer is not inside this stream's arena");
    Buf rf = a0.apply ? ov : rv;
    if (a0.apply) {
        if (!arena_resolve(s, a0.bo_prev, (uint64_t) n_embd * 4 * nt, bv) ||
            !arena_resolve(s, a0.inj_prev, (uint64_t) hc * 4 * nt, iv))
            refuse("fused_gr_read_batch", "the fold's bo_prev/inj_prev is not inside this stream's arena");
    } else {
        bv = iv = dummy_buf(s);
    }
    if (a0.w_inject != nullptr) {
        if (!arena_resolve(s, a0.w_inject, (uint64_t) hc * (uint64_t) (hc_dim / 2) * 4, wiv) ||
            !arena_resolve(s, a0.inject_out, (uint64_t) hc * 4 * nt, jv))
            refuse("fused_gr_read_batch", "the inject pair is not inside this stream's arena");
    } else {
        wiv = jv = dummy_buf(s);
    }
    const uint32_t gy = (uint32_t) n_tok;
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_rs.spv", 5, 20);
        struct { int32_t n_embd, hc; float eps; int32_t apply; } pc{(int32_t) n_embd, (int32_t) hc, a0.eps,
                                                                   a0.apply ? 1 : 0};
        s.ctx->dispatch(p, {&rv, &bv, &iv, &ov, &sv}, &pc, sizeof(pc), (uint32_t) hc, gy);
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_down.spv", 5, 12);
        struct { int32_t n_embd, hc, hc_lr; } pc{(int32_t) n_embd, (int32_t) hc, (int32_t) hc_lr};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &dv, &lv}, &pc, sizeof(pc), (uint32_t) hc_lr, gy);
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_mix.spv", 6, 12);
        struct { int32_t n_embd, hc, hc_lr; } pc{(int32_t) n_embd, (int32_t) hc, (int32_t) hc_lr};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &lv, &uv, &mv}, &pc, sizeof(pc), (uint32_t) n_embd, gy);
    }
    if (a0.w_inject != nullptr) {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fused_gr_inject.spv", 5, 8);
        struct { int32_t n_embd, hc; } pc{(int32_t) n_embd, (int32_t) hc};
        s.ctx->dispatch(p, {&rf, &nv, &sv, &wiv, &jv}, &pc, sizeof(pc), (uint32_t) hc, gy);
    }
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/fused_gr.hpp declares -------------------
namespace strata::kernels {

static strata::vulkan::Stream* need_stream(const char* who, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return s;
}

// fused_gr.hpp: `void fused_gr_read(const FusedGrArgs& a, void* stream);`  (layer.cpp:1253/:1276, the fused
//     branch of `block_layer_pre`, and the head mixer).
void fused_gr_read(const FusedGrArgs& a, void* stream) {
    strata::vulkan::fused_gr_read_one(*need_stream("fused_gr_read", stream), a);
}

// fused_gr.hpp: `void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
//     unsigned long long* stamp_buf, int stamp_i0);`  (verify.cpp:693/:1142 - THE P6 VERIFY WINDOW, a native
//     pack's only decode path - and mtp.cpp:510/:571/:620).
//
// The multi's contract is per-token bitwise equality with the single read (fused_gr.hpp:47), so it is a LOOP.
// The CUDA's other properties are checked and either honoured or refused: the tokens must share the four weight
// pointers and `eps` (fused_gr.cu:1066-1071, a loud exit there and here), and `n_tok` must be 1..kFusedGrMaxT.
// `xn_scratch` is the CUDA's per-token `xn` staging; this port recomputes `xn` instead of staging it, so the
// argument is accepted and NOT used (a caller may pass null) - stated rather than silently assumed.
void fused_gr_read_multi(const FusedGrArgs* a, int n_tok, float* xn_scratch, void* stream,
                         unsigned long long* stamp_buf, int stamp_i0) {
    (void) xn_scratch;
    if (a == nullptr || n_tok < 1 || n_tok > kFusedGrMaxT) {
        std::fprintf(stderr, "fused_gr_read_multi: invalid arguments\n");
        std::exit(1);
    }
    for (int t = 0; t < n_tok; ++t) {
        const FusedGrArgs& x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed ||
            (x.w_inject && !x.inject_out) || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            x.w_down != a[0].w_down || x.w_up != a[0].w_up || x.w_inject != a[0].w_inject ||
            x.w_norm != a[0].w_norm || x.eps != a[0].eps) {
            std::fprintf(stderr, "fused_gr_read_multi: invalid arguments for token %d\n", t);
            std::exit(1);
        }
    }
    strata::vulkan::Stream* s = need_stream("fused_gr_read_multi", stream);
    // The token-batched multi (see fused_gr_read_batch): one grid per stage over every token, bitwise identical,
    // taken only when the buffers are contiguous. `STRATA_FGR_TOKEN_BATCH=0` restores the per-token loop.
    static const bool tb_env = [] {
        const char* v = std::getenv("STRATA_FGR_TOKEN_BATCH");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (stamp_buf != nullptr) gpu_stamp(stamp_buf, stamp_i0, stream);
    if (tb_env && n_tok > 1 && strata::vulkan::fused_gr_batch_ok(a, n_tok)) {
        strata::vulkan::fused_gr_read_batch(*s, a, n_tok);
    } else {
        for (int t = 0; t < n_tok; ++t) strata::vulkan::fused_gr_read_one(*s, a[t]);
    }
    if (stamp_buf != nullptr) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
}

}  // namespace strata::kernels
