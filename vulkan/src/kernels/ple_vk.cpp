// vulkan/src/kernels/ple_vk.cpp - the Vulkan backend's PLE / GR SHARED STAGES and the MoE ROUTING rows.
//
// ============================================================================================================
// WHICH SIX, AND WHY THESE (derived from the engine's own body, not from the plan's list)
// ============================================================================================================
//
// `block_layer_pre` (`src/core/layer.cpp:1170`) runs the PLE stage and the two GR stages BEFORE either layer
// body (`qsa_layer` / `gdn_layer`), and `block_layer_post` (:1319) runs the closing `gr_write`.  The GR pair is
// the SHARED stage (every layer, 48 of 48); the PLE stage is layer-1 only; the MoE routing rows complete the
// block's stage 4/5.  The order below is the SOURCE order of the call sites - within a native/legacy pair the
// branches are alternatives, so "the order the body reaches them" is the order the calls appear:
//
//   1. `gr_write`            layer.cpp:1261 (run2, unfused), :1195/:1329 (fused / head)     -> gr_write.spv
//   2. `ple_block`           layer.cpp:1208 (the PLE block, layer 1)                        -> q8_0+s2 / bf16 + gnorm/gate/bcast/gnorm/conv/add3
//   3. `ple_history_advance` layer.cpp:1222 (the PLE history shift)                         -> ple_history_advance.spv
//   4. `gr_read`             layer.cpp:1255 (run0, unfused)                                 -> gr_norm/gr_down/gr_gate/gr_mean/gr_inject
//   5. `router_top10`        layer.cpp:373  (`moe_route`'s generic router)                  -> router_top10_f32.spv
//   6. `native_moe_combine`  layer.cpp:463  (`moe_combine_parts`, the DEFAULT combine)       -> native_moe_combine.spv
//
// `gr_write` is reached on BOTH branches of the fused/unfused GR choice (:1261 unfused, :1328 post/head) so it
// cannot be dodged; `gr_read` is the UNFUSED member of the pair the `gr_set_native_mmvf(false)` contract
// selects.  The fused `fused_gr_read`/`fused_gr_supported` are the flag-removed alternative; they stay unwired.
// `native_router_top10` (the router's native member) is already wired in qsa_vk.cpp; `native_moe_combine` is
// the DEFAULT combine because `native_moe_combine_enabled()` answers true (native_caps_vk.cpp).
//
// THE FOUR MORE (this batch) - the MoE / GR / PLE tail that still had no definition, in call-site order:
//
//   7. `shared_expert_scratch_bytes`  layer.cpp:347   `host`  the MoE workspace size (moe_buffers_init)
//   8. `fused_gr_supported`           layer.cpp:1189  `host`  the GR geometry predicate (short-circuited off,
//                                                             but layer.cpp references it, so it must LINK)
//   9. `moe_combine`                  layer.cpp:464   kernel  -> moe_combine_f32.spv (the LEGACY combine)
//  10. `ngram_rows`                   layer.cpp:1293  `host`  the PLE hash (src/kernels/ngram.cpp on CUDA)
//
// `shared_expert` ITSELF IS NOT WIRED AND CANNOT BE, and that is a FINDING rather than an omission: its
// CANONICAL (non-native) path (`shared_expert.cu:242-361`) dispatches `s_gemv_q8_0_split` / `s_gemv_q8k_split`
// for the K-quant gate/up and the legacy Q8_0-activation down projection, and NEITHER has a shader in this
// tree.  Wiring it would need those two shaders PORTED - a shader job, not a wrapper job.  Only its
// free workspace SIZE is answerable here, so that is what this TU answers.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each symbol below is the thin wrapper already declared in
// include/strata/kernels/{gr,ple,ngram,router_top10,native_moe}.hpp; this TU answers the `strata::kernels::`
// symbol the wrapper calls.  On a CUDA/HIP build src/kernels/cuda/*.cu answer them; on this build THIS file
// does - each body resolves the engine's raw device pointers to arena views (`arena_resolve`), takes the
// pipeline the device layer caches for the shader's own signature, and dispatches.
//
// ============================================================================================================
// THE `host` ROWS THIS TU ANSWERS (not a bare bind)
// ============================================================================================================
//
//   * `gr_workspace_init` / `gr_workspace_bytes(s)` - the `gr_read` WORKSPACE table.  `block_buffers_init`
//     (`layer.cpp:1140`) calls it; on a CUDA build `gr.cu` owns it.  Transcribed from gr.cu:300-330: ONE
//     table sizes and assigns the five 16-byte-aligned regions (the CUDA's own worst bug sized them in a
//     different order from the assignments).
//   * `ple_block_scratch_bytes` - the PLE workspace SIZE (`layer.cpp:1203/1316`).  Transcribed from
//     `ple.cu:219-228`; the wrapper carves the same regions out of it.
//   * `ple_native_bf16_enabled` / `ple_native_postops_enabled` (+ setters) - the PLE branch policy, answered
//     the way `native_gdn_enabled()` is: the backend reports what it implements.  It implements the CANONICAL
//     (non-native) PLE projections and post-ops, so both answer FALSE.
//   * `gr_set_native_mmvf` / `gr_set_fp32_activations` - the GR branch policy: the port implements the DEFAULT
//     (BF16-activation, legacy) GR form, so both setters are no-ops (the `native_*_set_enabled` precedent).
#if !defined(STRATA_ENABLE_VULKAN)
#error "ple_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/gr.hpp"             // gr_read / gr_write / GrShapes / GrWorkspace / gr_workspace_init
#include "strata/kernels/ple.hpp"            // ple_block / ple_history_advance / ple_block_scratch_bytes
#include "strata/kernels/ngram.hpp"          // NG_N_EMBD / NG_HC / NG_HC_DIM / NG_HIST / PLE_CONV_KERNEL / NGRAM_SIZE
#include "strata/kernels/router_top10.hpp"   // router_top10
#include "strata/kernels/native_moe.hpp"     // native_moe_combine
#include "strata/kernels/shared_expert.hpp"  // shared_expert_scratch_bytes / moe_combine (the MoE block)
#include "strata/kernels/fused_gr.hpp"       // fused_gr_supported (the GR geometry predicate)
#include "strata/kernels/quantize_act.hpp"   // quantize_q8_0 (the PLE key's activation image)
#include "strata/kernels/bf16_gemv.hpp"      // bf16_gemv / bf16_gemv_fp32_mmvf (the PLE projections)
#include "strata/kernels/s2_gemv_q8.hpp"     // s2_gemv_q8 (the PLE key projection)
#include "strata/kernels/elementwise.hpp"    // f32_to_bf16_bulk (the value projection's activation copy)
#include "strata/vulkan/vk_backend.hpp"      // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                      // the arena + pointer->buffer resolution

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::vulkan {

using namespace strata::kernels;   // GrShapes/GrWorkspace/PleWeights/PleOut/NG_* are the engine's, qualify-free

static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

// A Vulkan descriptor cannot be null.  `native_moe_combine` always binds `shared` (binding 2) even with
// has_shared=0; the sentinel lives with the stream (a per-dispatch allocation would exhaust the arena).
static Buf& dummy_buf(Stream& s) {
    if (s.dummy.buffer == VK_NULL_HANDLE) {
        s.dummy = s.ctx->alloc(64);
        const int32_t zero[16] = {0};
        s.ctx->write(s.dummy, zero, sizeof(zero));
    }
    return s.dummy;
}

// A device-to-device float copy through copy.spv (2 bindings, push {int n}).  The port's `ple_gnorm` shader
// normalises IN PLACE, so the query norm (`gnorm(hidden) -> d_query`, a two-destination call in the CUDA) and
// every PLE export are a copy followed by the in-place kernel - exactly what the port's `case_ple` does.
static void copy_floats(Stream& s, const Buf& src, uint64_t src_off, const Buf& dst, uint64_t dst_off,
                        uint64_t count) {
    if (count == 0) return;
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/copy.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) count};
    Buf sv = src, dv = dst;   // offsets are folded into the view by arena_resolve's callers below
    (void) src_off; (void) dst_off;
    s.ctx->dispatch(p, {&sv, &dv}, &pc, sizeof(pc), groups_for(count));
}

// ============================================================================================================
// THE GR SHARED STAGES
// ============================================================================================================

// `gr_write` -> gr_write.spv (R ro, BO ro, INJ ro, OUT rw; push {int n_embd; int hc}; one surplus group so a
// missing `i < n` guard is visible).  In place is the engine's shape (R_out == R) and is elementwise-safe.
static void gr_write_impl(Stream& s, const float* R, const float* block_out, const float* inject, int64_t n_embd,
                          int64_t hc, float* R_out) {
    if (n_embd <= 0 || hc <= 0) return;
    const uint64_t n = (uint64_t) hc * (uint64_t) n_embd;
    if (!R || !block_out || !inject || !R_out) refuse("gr_write", "a required pointer is null");
    Buf rv{}, bv{}, iv{}, ov{};
    if (!arena_resolve(s, R, n * 4, rv) || !arena_resolve(s, block_out, (uint64_t) n_embd * 4, bv) ||
        !arena_resolve(s, inject, (uint64_t) hc * 4, iv) || !arena_resolve(s, R_out, n * 4, ov))
        refuse("gr_write", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gr_write.spv", 4, 8);
    struct Push { int32_t n_embd; int32_t hc; } pc{};
    pc.n_embd = (int32_t) n_embd;
    pc.hc = (int32_t) hc;
    s.ctx->dispatch(pipe, {&rv, &bv, &iv, &ov}, &pc, sizeof(pc), groups_for(n) + 1u);
}

// `gr_read` -> gr_norm + gr_down + gr_gate + gr_mean + gr_inject (the UNFUSED five-stage chain).
//   gr_norm   (R ro, WN ro, XN rw, XQ rw; push {n_embd, hc, eps}; one workgroup per stream)
//   gr_down   (XQ ro, WD ro, LQ rw; push {hc_dim, hc_lr, hc}; one workgroup per bottleneck row)
//   gr_gate   (LQ ro, WU ro, XN ro, GT rw; push {hc_dim, hc_lr}; one workgroup per stream element)
//   gr_mean   (GT ro, MX rw; push {n_embd, hc})   mixed = MEAN over streams
//   gr_inject (XQ ro, WI ro, INJ rw; push {hc_dim, hc})   inject = bf16(xn) @ w_inject.T
// `w_inject == nullptr` is the FINAL mixer (`lm_head_mix`): the inject stage is SKIPPED and the caller's buffer
// is left untouched (gr.cu:387-389).  The workspace is caller-owned; its `bytes` is CHECKED (gr.cu:352).
static void gr_read_impl(Stream& s, const float* R, const float* w_norm, const uint16_t* w_down,
                         const uint16_t* w_up, const uint16_t* w_inject, float eps, int64_t n_embd, int64_t hc,
                         int64_t hc_lr, const GrWorkspace& ws, float* mixed, float* inject) {
    const int64_t hc_dim = hc * n_embd;
    if (n_embd <= 0 || hc <= 0 || hc_lr <= 0 || (hc_lr % 2) != 0 || (hc_dim % 2) != 0)
        refuse("gr_read", "the GR geometry is outside this shader's contract (even hc_lr, even hc*n_embd)");
    if (!R || !w_norm || !w_down || !w_up || !mixed || !ws.xn || !ws.xq || !ws.lq || !ws.gated)
        refuse("gr_read", "a required pointer is null");
    GrShapes shapes{};
    shapes.n_embd = n_embd; shapes.hc = hc; shapes.hc_lr = hc_lr;
    if (ws.bytes < gr_workspace_bytes(shapes))
        refuse("gr_read", "the GrWorkspace is under-sized (see gr_workspace_init)");
    Buf rv{}, nv{}, xnv{}, xqv{}, dv{}, lv{}, uv{}, gv{}, mv{}, iv{}, jv{};
    if (!arena_resolve(s, R, (uint64_t) hc_dim * 4, rv) || !arena_resolve(s, w_norm, (uint64_t) hc_dim * 4, nv) ||
        !arena_resolve(s, ws.xn, (uint64_t) hc_dim * 4, xnv) || !arena_resolve(s, ws.xq, (uint64_t) hc_dim * 4, xqv) ||
        !arena_resolve(s, w_down, (uint64_t) hc_lr * (uint64_t) (hc_dim / 2) * 4, dv) ||
        !arena_resolve(s, ws.lq, (uint64_t) hc_lr * 4, lv) ||
        !arena_resolve(s, w_up, (uint64_t) hc_dim * (uint64_t) (hc_lr / 2) * 4, uv) ||
        !arena_resolve(s, ws.gated, (uint64_t) hc_dim * 4, gv) || !arena_resolve(s, mixed, (uint64_t) n_embd * 4, mv))
        refuse("gr_read", "a pointer is not inside this stream's arena");
    if (w_inject && inject) {
        if (!arena_resolve(s, w_inject, (uint64_t) hc * (uint64_t) (hc_dim / 2) * 4, iv) ||
            !arena_resolve(s, inject, (uint64_t) hc * 4, jv))
            refuse("gr_read", "an inject pointer is not inside this stream's arena");
    } else {
        iv = jv = dummy_buf(s);   // bound but never dispatched
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gr_norm.spv", 4, 12);
        struct { int32_t n_embd, hc; float eps; } pc{(int32_t) n_embd, (int32_t) hc, eps};
        s.ctx->dispatch(p, {&rv, &nv, &xnv, &xqv}, &pc, sizeof(pc), (uint32_t) hc);
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gr_down.spv", 3, 12);
        struct { int32_t hc_dim, hc_lr, hc; } pc{(int32_t) hc_dim, (int32_t) hc_lr, (int32_t) hc};
        s.ctx->dispatch(p, {&xqv, &dv, &lv}, &pc, sizeof(pc), (uint32_t) hc_lr);
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gr_gate.spv", 4, 8);
        struct { int32_t hc_dim, hc_lr; } pc{(int32_t) hc_dim, (int32_t) hc_lr};
        s.ctx->dispatch(p, {&lv, &uv, &xnv, &gv}, &pc, sizeof(pc), (uint32_t) hc_dim);
    }
    {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gr_mean.spv", 2, 8);
        struct { int32_t n_embd, hc; } pc{(int32_t) n_embd, (int32_t) hc};
        s.ctx->dispatch(p, {&gv, &mv}, &pc, sizeof(pc), groups_for((uint64_t) n_embd));
    }
    if (w_inject && inject) {
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gr_inject.spv", 3, 8);
        struct { int32_t hc_dim, hc; } pc{(int32_t) hc_dim, (int32_t) hc};
        s.ctx->dispatch(p, {&xqv, &iv, &jv}, &pc, sizeof(pc), (uint32_t) hc);
    }
}

// ============================================================================================================
// THE PLE STAGE (the canonical non-native path; layer 1 only)
// ============================================================================================================

// `ple_block` -> quantize_q8_0 + s2_gemv_q8 (key) ; f32_to_bf16_bulk + bf16_gemv (value) ; gnorm, gate, bcast,
// gnorm, conv, add3.  THE CANONICAL PATH (`ple_native_bf16_enabled() == false`, ple.cu:313-329).  THE KEY
// PROJECTION IS PART OF THIS SYMBOL - `ple_block_projected` is the separate entry for a caller that has
// already projected and stays a `todo` row.  `out.normalized` is the CONV INPUT (`d_norm`), `out.key` the
// NORMALISED key (`d_key`), exactly as ple.cu:352-361 exports them.
static void ple_block_impl(Stream& s, const float* emb, const float* hidden, const float* hist_rows,
                           const PleWeights& w, PleOut& out, void* scratch) {
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    if (!emb || !hidden || !hist_rows || !out.result || !scratch)
        refuse("ple_block", "a required pointer is null");
    if (w.key_native_data != nullptr)
        refuse("ple_block", "the native key projection has no shader in this tree (answer ple_native_bf16 false)");
    if (w.key_bf16 == nullptr && (w.key_codes == nullptr || w.key_scales == nullptr))
        refuse("ple_block", "the canonical key projection needs key_codes/key_scales");
    if (w.value_bf16 == nullptr || w.norm_key == nullptr || w.norm_query == nullptr || w.norm_conv == nullptr ||
        w.conv1d_f16 == nullptr)
        refuse("ple_block", "a PLE weight pointer is null");

    // ---- the scratch region table (ple.cu:258, :293-303) ----
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    uint8_t* base = (uint8_t*) scratch;
    float* d_key = (float*) base;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));

    const auto res = [&](const void* p, uint64_t bytes) -> Buf {
        Buf b{};
        if (!arena_resolve(s, p, bytes, b)) refuse("ple_block", "a pointer is not inside this stream's arena");
        return b;
    };
    Buf ev = res(emb, (uint64_t) n_embd * 4), hv = res(hidden, (uint64_t) hc_dim * 4),
        hhv = res(hist_rows, (uint64_t) NG_HIST * (uint64_t) hc_dim * 4), addv = res(scratch, ple_block_scratch_bytes());
    Buf sk = res(d_key, (uint64_t) hc_dim * 4), sq = res(d_query, (uint64_t) hc_dim * 4),
        sn = res(d_norm, (uint64_t) hc_dim * 4), sg = res(d_gated, (uint64_t) hc_dim * 4),
        sc = res(d_conv, (uint64_t) hc_dim * 4), sv = res(d_value, (uint64_t) n_embd * 4),
        sgt = res(d_gate, (uint64_t) hc * 4), sact = res(d_act, q8_bytes),
        em16v = res(d_emb16, (uint64_t) n_embd * 2);
    Buf wk = res(w.norm_key, (uint64_t) hc_dim * 4), wqv = res(w.norm_query, (uint64_t) hc_dim * 4),
        wcv = res(w.norm_conv, (uint64_t) hc_dim * 4), kcv = res(w.conv1d_f16, (uint64_t) PLE_CONV_KERNEL * (uint64_t) hc_dim * 2),
        vbv = res(w.value_bf16, (uint64_t) n_embd * (uint64_t) n_embd * 2);
    (void) ev; (void) addv; (void) sact; (void) em16v; (void) vbv;

    // ---- key = gnorm(ple_key @ emb) ----
    if (w.key_bf16 != nullptr) {
        (void) res(w.key_bf16, (uint64_t) n_embd * (uint64_t) hc_dim * 2);
        strata::kernels::bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, &s);
    } else {
        (void) res(w.key_codes, (uint64_t) hc_dim * (uint64_t) (n_embd / 4));
        (void) res(w.key_scales, (uint64_t) hc_dim * (uint64_t) (n_embd / 64) * 4);
        strata::kernels::quantize_q8_0(emb, d_act, n_embd, &s);
        strata::kernels::s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, &s);
    }
    {   // gnorm(key) IN PLACE on d_key
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/ple_gnorm.spv", 2, 12);
        struct { int32_t rows, cols; float eps; } pc{hc, n_embd, NG_RMS_EPS};
        s.ctx->dispatch(p, {&sk, &wk}, &pc, sizeof(pc), (uint32_t) hc);
    }
    {   // query = gnorm(hidden -> d_query): the shader is in place, so copy hidden first (the CUDA writes a
        // distinct destination; `hidden` is needed again by add3, so normalising it in place would be wrong)
        Buf hv2 = res(hidden, (uint64_t) hc_dim * 4);
        copy_floats(s, hv2, 0, sq, 0, (uint64_t) hc_dim);
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/ple_gnorm.spv", 2, 12);
        struct { int32_t rows, cols; float eps; } pc{hc, n_embd, NG_RMS_EPS};
        s.ctx->dispatch(p, {&sq, &wqv}, &pc, sizeof(pc), (uint32_t) hc);
    }
    // ---- value = bf16_gemv(bf16(emb), value_bf16) ----
    strata::kernels::f32_to_bf16_bulk(emb, d_emb16, n_embd, &s);
    strata::kernels::bf16_gemv(d_emb16, w.value_bf16, d_value, n_embd, n_embd, &s);

    {   // gate = f(key, query) ; gated = value * gate ; normalized = gnorm(gated) ; conv ; result
        VkPipeline pg = s.ctx->pipeline(s.spv_dir + "/ple_gate.spv", 3, 12);
        struct { int32_t streams, n; float inv_sqrt_n; } pcg{hc, n_embd, 1.0f / std::sqrt((float) n_embd)};
        s.ctx->dispatch(pg, {&sk, &sq, &sgt}, &pcg, sizeof(pcg), (uint32_t) hc);
        VkPipeline pb = s.ctx->pipeline(s.spv_dir + "/ple_bcast.spv", 3, 8);
        struct { int32_t n, hc; } pcb{n_embd, hc};
        s.ctx->dispatch(pb, {&sv, &sgt, &sg}, &pcb, sizeof(pcb), groups_for((uint64_t) hc_dim));
        // gated -> normalized.  THE PORT'S gnorm IS IN PLACE, so `gated` must be COPIED into d_norm first: the
        // CUDA writes a distinct destination (`gnorm_kernel(d_gated, w, d_norm)`), and `gated` is needed again by
        // add3 and by the export.  Writing the norm back onto d_gated would clobber it - the exact composition
        // trap the port's `case_ple` records ("a composition test has to catch its intermediates on the way past").
        copy_floats(s, sg, 0, sn, 0, (uint64_t) hc_dim);
        VkPipeline pn = s.ctx->pipeline(s.spv_dir + "/ple_gnorm.spv", 2, 12);
        struct { int32_t rows, cols; float eps; } pcn{hc, n_embd, NG_RMS_EPS};
        s.ctx->dispatch(pn, {&sn, &wcv}, &pcn, sizeof(pcn), (uint32_t) hc);   // normalized = gnorm(gated), in place
        VkPipeline pc2 = s.ctx->pipeline(s.spv_dir + "/ple_conv.spv", 4, 16);
        struct { int32_t hc_dim, kern, dil, nhist; } pcc{hc_dim, PLE_CONV_KERNEL, NGRAM_SIZE, NG_HIST};
        s.ctx->dispatch(pc2, {&hhv, &sn, &kcv, &sc}, &pcc, sizeof(pcc), groups_for((uint64_t) hc_dim));
        // result = hidden + gated + conv.  `out.result` may alias `hidden` (layer.cpp:1206 sets result = bb.R);
        // add3 is elementwise, so the in-place read is well-defined.
        Buf rv = res(out.result, (uint64_t) hc_dim * 4);
        VkPipeline pa = s.ctx->pipeline(s.spv_dir + "/add3.spv", 4, 4);
        struct { int32_t n; } pca{hc_dim};
        s.ctx->dispatch(pa, {&hv, &sg, &sc, &rv}, &pca, sizeof(pca), groups_for((uint64_t) hc_dim));
    }
    // ---- the exports the caller asked for (a device copy each; `normalized` IS the conv input d_norm) ----
    const auto export_to = [&](float* dst, uint64_t count, const Buf& src) {
        if (dst == nullptr) return;
        Buf d = res(dst, count * 4);
        copy_floats(s, src, 0, d, 0, count);
    };
    export_to(out.value, (uint64_t) n_embd, sv);
    export_to(out.key, (uint64_t) hc_dim, sk);
    export_to(out.gate, (uint64_t) hc, sgt);
    export_to(out.gated, (uint64_t) hc_dim, sg);
    export_to(out.normalized, (uint64_t) hc_dim, sn);
    export_to(out.conv, (uint64_t) hc_dim, sc);
}

// `ple_history_advance` -> ple_history_advance.spv (H rw, N ro; push {int channels; int nhist}).  ONE THREAD
// PER CHANNEL, row-fastest `hist[row + nhist*channel]`; in place and bit-exact (a copy, not arithmetic).
static void ple_history_advance_impl(Stream& s, float* hist, const float* normalized) {
    const int channels = NG_HC_DIM, nhist = NG_HIST;
    if (!hist || !normalized) refuse("ple_history_advance", "a required pointer is null");
    Buf hv{}, nv{};
    if (!arena_resolve(s, hist, (uint64_t) channels * (uint64_t) nhist * 4, hv) ||
        !arena_resolve(s, normalized, (uint64_t) channels * 4, nv))
        refuse("ple_history_advance", "a pointer is not inside this stream's arena");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/ple_history_advance.spv", 2, 8);
    struct { int32_t channels, nhist; } pc{channels, nhist};
    s.ctx->dispatch(p, {&hv, &nv}, &pc, sizeof(pc), groups_for((uint64_t) channels));
}

// ============================================================================================================
// THE MoE ROUTING ROWS
// ============================================================================================================

// `router_top10` -> router_top10_f32.spv (LOGITS ro, IDS rw, WEIGHTS rw; push {n_tokens, n_expert, k}; one
// workgroup per token).  THE PORTABLE (f32) variant: the engine's router computes its exponentials and its
// sum in DOUBLE and the target has no shaderFloat64, so the faithful member cannot run here.  The honest
// deviation is MEASURED by the gate's `case_router` (ids vs the double rule on realistic rows and near-ties).
static void router_top10_impl(Stream& s, const float* logits, int n_tokens, int n_expert, int k, int32_t* ids,
                              float* weights) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (!logits || !ids || !weights) refuse("router_top10", "a required pointer is null");
    Buf lv{}, iv{}, wv{};
    if (!arena_resolve(s, logits, (uint64_t) n_tokens * (uint64_t) n_expert * 4, lv) ||
        !arena_resolve(s, ids, (uint64_t) n_tokens * (uint64_t) k * 4, iv) ||
        !arena_resolve(s, weights, (uint64_t) n_tokens * (uint64_t) k * 4, wv))
        refuse("router_top10", "a pointer is not inside this stream's arena");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/router_top10_f32.spv", 3, 12);
    struct { int32_t n_tokens, n_expert, k; } pc{n_tokens, n_expert, k};
    s.ctx->dispatch(p, {&lv, &iv, &wv}, &pc, sizeof(pc), (uint32_t) n_tokens);
}

// `native_moe_combine` -> native_moe_combine.spv (PARTS ro, W ro, S ro, Y rw; push {n_embd, k, has_shared}).
// The NATIVE pure-f32 expression: the first term is a PRODUCT, later terms a separate multiply and add, the
// shared row added PLAIN.  `shared == nullptr` binds the sentinel with has_shared=0.
static void native_moe_combine_impl(Stream& s, const float* parts, const float* weights, const float* shared,
                                    float* output, int64_t n_embd, int64_t k) {
    if (n_embd <= 0 || k <= 0) return;
    if (!parts || !weights || !output) refuse("native_moe_combine", "a required pointer is null");
    if (k > 15) refuse("native_moe_combine", "the pinned contract is k in [1, 15]");
    Buf pv{}, wv{}, sv{}, ov{};
    if (!arena_resolve(s, parts, (uint64_t) k * (uint64_t) n_embd * 4, pv) ||
        !arena_resolve(s, weights, (uint64_t) k * 4, wv) || !arena_resolve(s, output, (uint64_t) n_embd * 4, ov))
        refuse("native_moe_combine", "a pointer is not inside this stream's arena");
    if (shared != nullptr) {
        if (!arena_resolve(s, shared, (uint64_t) n_embd * 4, sv))
            refuse("native_moe_combine", "the shared pointer is not inside this stream's arena");
    } else {
        sv = dummy_buf(s);
    }
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/native_moe_combine.spv", 4, 12);
    struct { int32_t n_embd, k, has_shared; } pc{(int32_t) n_embd, (int32_t) k, shared ? 1 : 0};
    s.ctx->dispatch(p, {&pv, &wv, &sv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) n_embd));
}

// `moe_combine` -> moe_combine_f32.spv (PARTS ro, W ro, S ro, Y rw; push {n_embd, k, has_shared}).  THE
// LEGACY combine - the `native_moe_combine_enabled() == false` branch of `moe_combine_parts`
// (`layer.cpp:464`).  The port answers that flag TRUE, so this member is OFF the selected path, but the
// symbol is reached by the layer body's own source and must LINK.  f32, Kahan/fma accumulation
// (`moe_combine_f32.comp`), one thread per output element; `shared == nullptr` binds the sentinel with
// has_shared=0 (a Vulkan descriptor cannot be null).  The shared row is ADDED PLAIN - the port's rule.
static void moe_combine_impl(Stream& s, const float* parts, const float* weights, const float* shared,
                             float* y, int64_t n_embd, int64_t k) {
    if (n_embd <= 0 || k <= 0) return;
    if (!parts || !weights || !y) refuse("moe_combine", "a required pointer is null");
    if (k > 64) refuse("moe_combine", "k > 64 (the engine refuses rather than truncating)");
    Buf pv{}, wv{}, sv{}, yv{};
    if (!arena_resolve(s, parts, (uint64_t) k * (uint64_t) n_embd * 4, pv) ||
        !arena_resolve(s, weights, (uint64_t) k * 4, wv) || !arena_resolve(s, y, (uint64_t) n_embd * 4, yv))
        refuse("moe_combine", "a pointer is not inside this stream's arena");
    if (shared != nullptr) {
        if (!arena_resolve(s, shared, (uint64_t) n_embd * 4, sv))
            refuse("moe_combine", "the shared pointer is not inside this stream's arena");
    } else {
        sv = dummy_buf(s);
    }
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/moe_combine_f32.spv", 4, 12);
    struct { int32_t n_embd, k, has_shared; } pc{(int32_t) n_embd, (int32_t) k, shared ? 1 : 0};
    s.ctx->dispatch(p, {&pv, &wv, &sv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n_embd));
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/*.hpp declare --------------------------
namespace strata::kernels {

// every body refuses when the opaque stream is not a live Vulkan stream.
static strata::vulkan::Stream* need_stream(const char* who, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return s;
}

// gr.hpp: `void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& s,
//     float* R_out, void* stream);`  (layer.cpp:1261, :1329, :1195)
void gr_write(const float* R, const float* block_out, const float* inject, const GrShapes& sh, float* R_out,
              void* stream) {
    strata::vulkan::gr_write_impl(*need_stream("gr_write", stream), R, block_out, inject, sh.n_embd, sh.hc, R_out);
}

// gr.hpp: `void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
//     const uint16_t* w_inject, float eps, const GrShapes& s, const GrWorkspace& ws, float* mixed,
//     float* inject, void* stream);`  (layer.cpp:1255, :1278, lm_head_mix:1103)
void gr_read(const float* R, const float* w_norm, const uint16_t* w_down, const uint16_t* w_up,
             const uint16_t* w_inject, float eps, const GrShapes& sh, const GrWorkspace& ws, float* mixed,
             float* inject, void* stream) {
    strata::vulkan::gr_read_impl(*need_stream("gr_read", stream), R, w_norm, w_down, w_up, w_inject, eps, sh.n_embd,
                                 sh.hc, sh.hc_lr, ws, mixed, inject);
}

// gr.hpp: `size_t gr_workspace_init(const GrShapes& s, void* base, GrWorkspace& out);`  (a `host` row:
//     `block_buffers_init`, layer.cpp:1140).  Transcribed from gr.cu:300-330 - ONE table sizes and assigns.
size_t gr_workspace_init(const GrShapes& sh, void* base, GrWorkspace& out) {
    const size_t hc_dim = (size_t) sh.hc * (size_t) sh.n_embd;
    // **THE TWO ACTIVATION REGIONS ARE SIZED AS f32 HERE, AND THAT IS A PORT DEVIATION, STATED PLAINLY.**  The
    // CUDA's `xq`/`lq` are uint16 (`gr.cu`'s `gr_down_kernel<uint16_t>`); THIS port's `gr_norm.comp` binding 3 is
    // `float v[]` and it stores `bf16_round(x)` AS AN F32 ("bf16(xn) as f32"), and `gr_down`/`gr_gate`/`gr_inject`
    // read it the same way.  So the region must be 4 bytes per element or the shader writes past it into `lq`.
    // `gr_workspace_bytes` is the AUTHORITY for the allocation (`block_buffers_init` uses it) and `gr_read` reads
    // the same pointers, so the layout is internally consistent; only a comparison against the CUDA's byte
    // layout would see a difference, and the VALUES are identical.
    const size_t sz[5] = {
        hc_dim * sizeof(float),                    // 0: xn
        hc_dim * sizeof(float),                    // 1: xq  (bf16(xn) HELD AS F32 in this port)
        (size_t) sh.hc_lr * sizeof(float),         // 2: lq  (bf16(lo) HELD AS F32 in this port)
        hc_dim * sizeof(float),                    // 3: gated
        (size_t) sh.hc_lr * sizeof(float),         // 4: lo
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t) 15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char* p = (unsigned char*) base;
        void* ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float*) ptr[0];
        out.xq = (uint16_t*) ptr[1];
        out.lq = (uint16_t*) ptr[2];
        out.gated = (float*) ptr[3];
        out.lo = (float*) ptr[4];
    }
    return bytes;
}

// ple.hpp: `void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w,
//     PleOut& out, void* scratch, void* stream);`  (layer.cpp:1208)
void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w, PleOut& out,
               void* scratch, void* stream) {
    strata::vulkan::ple_block_impl(*need_stream("ple_block", stream), emb, hidden, hist_rows, w, out, scratch);
}

// ple.hpp: `void ple_history_advance(float* hist, const float* normalized, void* stream);`  (layer.cpp:1222)
void ple_history_advance(float* hist, const float* normalized, void* stream) {
    strata::vulkan::ple_history_advance_impl(*need_stream("ple_history_advance", stream), hist, normalized);
}

// ple.hpp: `uint64_t ple_block_scratch_bytes();`  (a `host` row: layer.cpp:1203/1316).  Transcribed from
//     ple.cu:219-228.
uint64_t ple_block_scratch_bytes() {
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return (uint64_t) (((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256);
}

// ple.hpp: the PLE BRANCH POLICY.  The backend reports what it implements (the `native_gdn_enabled()` rule):
// the CANONICAL (non-native) projections and post-ops, so both answer FALSE.
void ple_set_native_bf16(bool) {}
bool ple_native_bf16_enabled() { return false; }
void ple_set_native_postops(bool) {}
bool ple_native_postops_enabled() { return false; }

// gr.hpp: the GR BRANCH POLICY.  The port implements the DEFAULT (BF16-activation, legacy) GR form.
void gr_set_native_mmvf(bool) {}
void gr_set_fp32_activations(bool) {}

// router_top10.hpp: `void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids,
//     float* weights, void* stream);`  (layer.cpp:373, :415)
void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    strata::vulkan::router_top10_impl(*need_stream("router_top10", stream), logits, n_tokens, n_expert, k, ids,
                                      weights);
}

// native_moe.hpp: `void native_moe_combine(const float* parts, const float* weights, const float* shared,
//     float* output, int64_t n_embd, int64_t k, void* stream);`  (layer.cpp:463)
void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output,
                        int64_t n_embd, int64_t k, void* stream) {
    strata::vulkan::native_moe_combine_impl(*need_stream("native_moe_combine", stream), parts, weights, shared,
                                            output, n_embd, k);
}

// shared_expert.hpp: `uint64_t shared_expert_scratch_bytes(int64_t n_ff);`  (a `host` row: `moe_buffers_init`,
//     `layer.cpp:347`).  Transcribed from `shared_expert.cu:234`: gate(n_ff f32) | up(n_ff f32) |
//     q8_0(n_ff/32*34) | q8k(n_ff/256*292) | g(1 f32), each 16-byte aligned, plus 32 bytes of tail.
//     **THE `qk` REGION IS SIZED ON `n_ff/256*292` AND IS ZERO FOR THE ARTIFACT'S DOWN WIDTH (640 < 256*2),
//     WHICH IS NOT A BUG** - the header argues Q8_K is structurally impossible for 640; the CUDA sizes it
//     the same way.  The rival reading (size it on Q8_0 for every projection) changes the answer for the
//     gate/up widths, which the case pins.
uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~(uint64_t) 15;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~(uint64_t) 15;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~(uint64_t) 15;
    return a * 2 + q0 + qk + 32;
}

// shared_expert.hpp: `void moe_combine(const float* parts, const float* weights, const float* shared,
//     float* y, int64_t n_embd, int64_t k, void* stream);`  (layer.cpp:464, the LEGACY combine; the port's
//     `native_moe_combine_enabled()` answers true so the engine takes the native member, but the source row
//     still has to link).
void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    strata::vulkan::moe_combine_impl(*need_stream("moe_combine", stream), parts, weights, shared, y, n_embd, k);
}

// fused_gr.hpp: `bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr);`  (layer.cpp:1188/1328,
//     the `fused = g_fused_gr && fused_gr_supported(...)` test in `block_layer_pre`/`block_layer_post`).
//
// **A REACHABILITY DEFECT FOUND WHILE WIRING `shared_expert` (this batch), AND FIXED AT ITS CAUSE.**  The
// previous batch left this returning the CUDA's GEOMETRY predicate (`n_embd==2560 && hc==4 && hc_lr==320`) on
// the reasoning that "a backend cannot return false without lying about the geometry", and recorded that
// `fused_gr_read` "leaves the forward path" because `g_fused_gr` "is forced false by the GR contract".
// **THAT CLAIM WAS FALSE OF THE CODE.**  `g_fused_gr` (layer.cpp:42) DEFAULTS false, but generate.cpp:2284
// calls `layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && dump_layers.empty() &&
// dump_halves.empty() && !o.stage_timing)`, and the shipped `--native` launch sets `o.gr_native_mmvf = true`
// (generate.cpp:1804) with `no_fused_gr` false (its default) - so on the shipped run `g_fused_gr` IS TRUE, and
// with this predicate TRUE the fused branch (`fused_gr_read`, layer.cpp:1253/1276) IS the one the layer
// reaches, while the ported `gr_read` is the NON-selected branch.  **This is the exact shape of the
// `qsa_decode_attn_step` defect: a class label that was true of a PLAN and false of the CODE.**
//
// THE FIX, and why it is this predicate rather than a shader port: the branch is selected by the BACKEND's own
// answer here, and this backend has NO fused_gr shader.  Answering "the fused read is supported" while having no
// kernel for it is the lie; answering FALSE is the capability truth, and it is the SAME discipline as
// `native_mmvq_supported` (matvec_vk.cpp: true only for the six types this tree has shaders for).  With FALSE
// the engine takes the PORTED unfused read (`gr_read`) - the branch the port's whole GR contract was built
// against - and `fused_gr_read` becomes genuinely unreachable under every shipped configuration, which is what
// its LOUD REFUSAL (refusals_vk.cpp) names.  A future batch that ports `fused_gr_down`/`fused_gr_up` flips this
// to `return n_embd == 2560 && hc == 4 && hc_lr == 320;` and retires the refusal.
//
// (The GEOMETRY predicate itself is preserved as a distinct observable: `case_fused_gr_supported_entry` asserts
// that the CUDA geometry rule would answer TRUE at (2560,4,320) and FALSE elsewhere, and that the backend's
// capability answer is FALSE everywhere - so the two readings are told apart, not conflated.)
bool fused_gr_supported(int64_t n_embd, int64_t hc, int64_t hc_lr) {
    (void) n_embd; (void) hc; (void) hc_lr;
    return false;   // no fused_gr shader in this tree: the backend reports what it implements
}

// ngram.hpp: `void ngram_rows(const int32_t* tokens, const int32_t* prev, int n_tokens, const PleConsts& c,
//     uint32_t* out);`  (a `host` row: `ple_issue_token`, `layer.cpp:1293`).  THE PLE HASH - pure host, no
//     shader.  It USED to be transcribed here verbatim because a Vulkan build compiled no `src/` TU; the
//     PleTable wiring below now links the engine's OWN `src/kernels/ngram.cpp`, so the hash is the engine's
//     definition, not a second copy, and this TU no longer defines it (a duplicate would be a link error).

}  // namespace strata::kernels
