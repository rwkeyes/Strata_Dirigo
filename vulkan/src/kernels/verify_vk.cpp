// vulkan/src/kernels/verify_vk.cpp - THE P6 VERIFY WINDOW's kernels (and the MTP drafter's embedding branch).
//
// WHY THIS FILE EXISTS.  `Verifier::init` now SUCCEEDS (`verify.cpp:336`'s `fused_gr_supported` disjunct is
// true because the fused read is ported), so the window's BODY runs and reaches each of these symbols in turn.
// The refusals they used to be carried text saying "NOT REACHED by the shipped configuration" - **that text was
// FALSE of the code**: the shipped `--native` launch (NO `--no-pool`, `--expert-profile`, `--expert-cache`) DOES
// reach the verify window, and the window is a NATIVE PACK'S ONLY DECODE PATH (`generate.cpp:7578-7579` breaks
// the token loop).  This file replaces those refusals with real bodies.  The one refusal left in this file's
// family (`gdn_step_norm_multi`'s commit half was NOT left; see below) - nothing.
//
// WHICH SYMBOLS, AND IN THE ORDER `verify.cpp` REACHES THEM
// --------------------------------------------------------
//   record_window -> embedding:  `broadcast_streams`                verify.cpp:592/:606
//   pre(layer)   -> the GDN conv: `gdn_conv_l2_multi`                verify.cpp:726/:730
//                -> the GDN a/b : `gdn_ab_multi`                     verify.cpp:732
//                -> the recurrence: `gdn_step_norm_multi`            verify.cpp:743/:748
//                -> the router:  `native_router_top10_multi`         verify.cpp:920
//                -> the shared expert: `shared_expert_multi`         verify.cpp:980
//   post(layer)  -> the combine:  `native_moe_combine_multi`         verify.cpp:1083
//   capture_commit -> the conv commit: `gdn_conv_commit`             verify.cpp:1293/:1844
//   (mtp.cpp:497) -> the drafter's embedding: `add_streams_broadcast`
//
// THE SHAPING (the brief's standing rule, and the header's own contract): a `_multi` variant was a LOOP over
// the ALREADY-GATED single-token kernel, so each contract below was true BY CONSTRUCTION rather than by a
// shared-weight kernel that had to reproduce the single-token summation order.  `verify_kernels.hpp` says so
// in as many words:
//   * `gdn_conv_l2_multi`   "Bitwise `fused_gdn_conv_l2` per token."
//   * `gdn_ab_multi`        "Bitwise `fused_gdn_ab` per column."
//   * `gdn_step_norm_multi` "Bitwise `fused_gdn_step_norm`." (ONE exception - see below)
//   * `native_router_top10_multi` "each row exactly as the single call" (`native_router.hpp`)
//   * `native_moe_combine_multi`  "each as the single call" (`native_moe.hpp`)
//   * `shared_expert_multi`       "every token is bitwise `shared_expert` on that token" (`shared_expert.hpp`)
// IT IS NOW ONE DISPATCH PER ROUND, NOT A HOST LOOP.  The loop shape cost a recorded dispatch per token per
// stage, and the window's cost is a dispatch COUNT (~4,792 per round at ~46-54 us).  Each `_multi` issues ONE
// dispatch whose grid carries the token (a Y dimension, or the module's own column dimension) and whose
// per-token arithmetic is the single kernel's expression, so every contract above still holds by construction
// - `ports/vulkan/PERFORMANCE-B70-2026-10-06.md` measures the round.  The CUDA's one-weight-read-for-all-tokens
// is still a COST optimisation this port does not carry; the column walk inside a workgroup is the workaround.
//
// THE ONE EXCEPTION, AND WHY IT IS NOT A LOOP: `gdn_step_norm_multi` reads its loop bound `n = n_keep ? *n_keep
// : T` FROM DEVICE MEMORY (`verify_kernels.cu:178`), and its commit half is captured ONCE
// (`verify.cpp`'s `capture_commit`) and replayed with a live, per-window `n_keep`.  A host-side loop bound would
// bake one window's accepted count into the recording; a device->host read inside a capture is what this port
// forbids (see `moe_hit_grouped_s2_dev`).  So the loop is INSIDE the shader
// (`gdn_step_norm_multi.comp`), whose token body is the SAME enumeration as the ported single-token
// `fused_gdn_step_norm.spv` - per-token bitwise by construction, proven by the gate case.
//
// TWO THINGS THE MULTIS ADD THAT THE SINGLE-TONKERNELS DO NOT, both from the header, both checked:
//   * `gdn_conv_l2_multi` must NOT write the caller's `history` ("history is NOT written"), where the single
//     token kernel SLIDES it in place.  The loop therefore runs on a working copy, SEEDED with the window's own
//     starting history: `gdn_conv_tail` (`t_begin` host constant) when the group starts at a nonzero token, a
//     plain copy when it starts at 0.  The seeding rule is the CUDA multi's own window:
//     `win[j] = (t+j < 3) ? hist[c*3+t+j] : qkv[(t+j-3)*C+c]`.
//   * `gdn_step_norm_multi`'s verify half must NOT write the state ("the state is left untouched"): the token
//     loop runs on the stream's working scratch, and the real state is only READ (token 0).  The commit half
//     runs in place - the state is its output.
# if !defined(STRATA_ENABLE_VULKAN)
#error "verify_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/verify_kernels.hpp"   // broadcast_streams / add_streams_broadcast / gdn_conv_l2_multi /
                                               //   gdn_conv_commit / gdn_ab_multi / gdn_step_norm_multi
#include "strata/kernels/fused_gdn.hpp"        // fused_gdn_conv_l2 / fused_gdn_ab / fused_gdn_step_norm (the loops)
#include "strata/kernels/native_router.hpp"    // native_router_top10 (the loop)
#include "strata/kernels/native_moe.hpp"       // native_moe_combine (the loop)
#include "strata/kernels/shared_expert.hpp"    // shared_expert_multi + NativeSharedWeights
#include "strata/kernels/native_mmvq.hpp"      // native_quantize_q8_1 / native_mmvq / native_q8_1_bytes / _supported

#include "strata/vulkan/vk_backend.hpp"        // Stream, stream_of
#include "vk_arena.hpp"                        // arena_alloc / arena_resolve / Buf
#include "vk_multi.hpp"                        // the one-dispatch-per-round forms (fused_gdn_conv_l2_n, ...)

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

static Stream& need_stream(const char* who, void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return *s;
}

// THE WORKING SCRATCH (see the Stream field).  Carved lazily from the arena - `arena_alloc` is a pure bump
// (vk_arena.cpp:117-129) with no API call, which is what makes it legal inside a capture; a per-call bump would
// exhaust the arena, so each SIZE is placed once and reused, and the list is bounded (a runaway allocation is a
// loud refusal, never a silent arena exhaustion).
static void* need_scratch(Stream& s, uint64_t bytes) {
    for (int i = 0; i < s.vscratch_n; ++i)
        if (s.vscratch_bytes[i] >= bytes) return s.vscratch[i];
    if (s.vscratch_n >= (int) (sizeof(s.vscratch) / sizeof(s.vscratch[0])))
        refuse("verify window scratch",
               "the stream's working-scratch regions are all smaller than this call needs (bounded list)");
    void* p = arena_alloc(s, bytes);
    s.vscratch[s.vscratch_n] = p;
    s.vscratch_bytes[s.vscratch_n] = bytes;
    ++s.vscratch_n;
    return p;
}

// A Vulkan descriptor cannot be null; a buffer bound but never read reuses an already-resolved view.
static Buf& dummy_buf(Stream& s) {
    if (s.dummy.buffer == VK_NULL_HANDLE) {
        s.dummy = s.ctx->alloc(64);
        const int32_t zero[16] = {0};
        s.ctx->write(s.dummy, zero, sizeof(zero));
    }
    return s.dummy;
}

// THE DESCRIPTOR-OFFSET ALIGNMENT (kept as a note, because it is the reason the old per-token loop shape is
// gone).  The single-token wrappers take raw pointers, so a loop handed them `base + t*stride`, and a
// storage-buffer descriptor offset must be a multiple of the device's `minStorageBufferOffsetAlignment`
// (vk_compute.cpp:701-712).  On the Arc that limit is 4 bytes, so every stride bound; on llvmpipe it is 16,
// and `h_v*4 = 12` (h_v = 3) or `k*4 = 40` (k = 10) did NOT - which is why the per-token form carried a NAMED
// refusal and the gate SKIPPED such an arm.  THE ROUND FORM HAS NO PER-TOKEN DESCRIPTOR AT ALL: the token is a
// grid dimension and a push constant, so the whole class of arm disappears with the loop.
// (The old `token_view_bindable` helper was removed with the last per-token loop; nothing binds `base+t*stride`
// any more.)

// ---- the shared expert's three small stages, the SAME .spv the single-token `shared_expert` dispatches --------
// (`shared_expert_vk.cpp`'s swiglu_f32 / scalar_gate_f32 / scale_rows; the multi repeats them per token because
// it has no `scratch` argument of its own - see the note in the `shared_expert_multi` body).
static void swiglu_f32(Stream& s, const float* gate, const float* up, float* out, int64_t n) {
    Buf gv{}, uv{}, ov{};
    if (!arena_resolve(s, gate, (uint64_t) n * 4, gv) || !arena_resolve(s, up, (uint64_t) n * 4, uv) ||
        !arena_resolve(s, out, (uint64_t) n * 4, ov))
        refuse("shared_expert_multi", "the swiglu operand is not inside this stream's arena");
    struct { int32_t n; } pc{(int32_t) n};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/swiglu_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&gv, &uv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// Both shaders are ONE WORKGROUP PER TOKEN (`gl_WorkGroupID.x` = the token, `n_tokens` in the push constant), so
// the whole window is ONE dispatch each - and that is what keeps `shared_expert_multi` free of a per-token
// `g + t` view (a 4-BYTE descriptor offset, unbindable on a device whose `minStorageBufferOffsetAlignment` is
// 16).  Same shaders, same arithmetic, one dispatch instead of n_tok.
static void scalar_gate_f32(Stream& s, const uint16_t* x_bf16, const uint16_t* w_bf16, float* g, int64_t n_tok,
                            int64_t n_embd) {
    Buf xv{}, wv{}, ov{};
    if (!arena_resolve(s, x_bf16, (uint64_t) n_tok * (uint64_t) n_embd * 2, xv) ||
        !arena_resolve(s, w_bf16, (uint64_t) n_embd * 2, wv) ||
        !arena_resolve(s, g, (uint64_t) n_tok * 4, ov))
        refuse("shared_expert_multi", "the scalar-gate operand is not inside this stream's arena");
    struct { int32_t n_tokens; int32_t n_embd; } pc{(int32_t) n_tok, (int32_t) n_embd};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/scalar_gate_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&xv, &wv, &ov}, &pc, sizeof(pc), (uint32_t) n_tok);
}

static void scale_rows(Stream& s, float* out, const float* g, int64_t n_tok, int64_t n_embd) {
    Buf ov{}, gv{};
    if (!arena_resolve(s, out, (uint64_t) n_tok * (uint64_t) n_embd * 4, ov) ||
        !arena_resolve(s, g, (uint64_t) n_tok * 4, gv))
        refuse("shared_expert_multi", "the per-row scale operand is not inside this stream's arena");
    struct { int32_t n_tokens; int32_t n; } pc{(int32_t) n_tok, (int32_t) n_embd};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/scale_rows.spv", 2, sizeof(pc));
    s.ctx->dispatch(p, {&ov, &gv}, &pc, sizeof(pc), (uint32_t) n_tok);
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/*.hpp declare ---------------------------
namespace strata::kernels {

// ---- `broadcast_streams` / `add_streams_broadcast` -> bcast_streams.comp (ONE shader, a `mode` push constant) --
// verify_kernels.hpp: `void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream);`
//     (verify.cpp:592/:606 - the window's embedding broadcast, and it is the FIRST symbol the window's body
//     reaches).  `R[t][c][:] = x[t][:]` for every one of the `hc` streams.
void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    if (n_embd <= 0 || hc <= 0 || n_tok <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("broadcast_streams", stream);
    strata::vulkan::Buf xv{}, rv{};
    if (!arena_resolve(s, x, (uint64_t) n_tok * (uint64_t) n_embd * 4, xv) ||
        !arena_resolve(s, R, (uint64_t) n_tok * (uint64_t) hc * (uint64_t) n_embd * 4, rv))
        strata::vulkan::refuse("broadcast_streams", "a pointer is not inside this stream's arena");
    struct { int32_t n, hc, n_tok, mode; } pc{(int32_t) n_embd, hc, n_tok, 0};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/bcast_streams.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&xv, &xv, &rv}, &pc, sizeof(pc), strata::vulkan::groups_for((uint64_t) n_embd * (uint64_t) hc),
                   (uint32_t) n_tok);
}

// verify_kernels.hpp: `void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc,
//     int n_tok, void* stream);`  (mtp.cpp:497 - the MTP DRAFTER's embedding branch; NOT the verify window's,
//     which is `broadcast_streams` above.  This port's shipped launch does not pass `--mtp`, so it is the
//     drafter's symbol - ported because it shares the shader and the arithmetic, and because a reader who sees
//     the two names side by side in the map should find both real).
void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    if (n_embd <= 0 || hc <= 0 || n_tok <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("add_streams_broadcast", stream);
    strata::vulkan::Buf hv{}, ev{}, rv{};
    if (!arena_resolve(s, h, (uint64_t) n_tok * (uint64_t) hc * (uint64_t) n_embd * 4, hv) ||
        !arena_resolve(s, e, (uint64_t) n_tok * (uint64_t) n_embd * 4, ev) ||
        !arena_resolve(s, R, (uint64_t) n_tok * (uint64_t) hc * (uint64_t) n_embd * 4, rv))
        strata::vulkan::refuse("add_streams_broadcast", "a pointer is not inside this stream's arena");
    struct { int32_t n, hc, n_tok, mode; } pc{(int32_t) n_embd, hc, n_tok, 1};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/bcast_streams.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&hv, &ev, &rv}, &pc, sizeof(pc), strata::vulkan::groups_for((uint64_t) n_embd * (uint64_t) hc),
                   (uint32_t) n_tok);
}

// ---- `gdn_conv_commit` / the multi's history seed -> gdn_conv_tail.comp -------------------------------------
// verify_kernels.hpp: `void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep,
//     void* stream);`  (verify.cpp:1293/:1844, the commit graph: history <- the last three of
//     [history | qkv_0 .. qkv_{*n_keep - 1}]).  The count is a DEVICE int32.
void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    if (history == nullptr || qkv == nullptr || n_keep == nullptr || channels <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("gdn_conv_commit", stream);
    strata::vulkan::Buf hv{}, qv{}, kv{}, dv{};
    if (!arena_resolve(s, history, (uint64_t) channels * 3 * 4, hv) ||
        !arena_resolve(s, qkv, (uint64_t) channels * 4, qv) || !arena_resolve(s, n_keep, 4, kv) ||
        !arena_resolve(s, history, (uint64_t) channels * 3 * 4, dv))
        strata::vulkan::refuse("gdn_conv_commit", "a pointer is not inside this stream's arena");
    struct { int32_t channels, n, use_dev; } pc{channels, 0, 1};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gdn_conv_tail.spv", 4, sizeof(pc));
    s.ctx->dispatch(p, {&hv, &qv, &kv, &dv}, &pc, sizeof(pc), strata::vulkan::groups_for((uint64_t) channels));
}

// verify_kernels.hpp: `void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w,
//     float* h, int channels, int qk_heads, float eps, int n_tok, void* stream, int t_begin = 0);`
//     (verify.cpp:726/:730).  Contract: "For token t of T: conv over [history(3) | qkv_0 .. qkv_t] -> SiLU ->
//     L2 norm of the q/k heads -> h[t].  `history` is NOT written.  Bitwise `fused_gdn_conv_l2` per token."
// So: a WORKING COPY of the history, seeded so the loop's first token sees the window the CUDA multi would,
// then `n_tok` calls of the ALREADY-GATED single-token `fused_gdn_conv_l2`.  The single-token kernel SLIDES the
// history it is given, which is exactly what makes the running window correct for token t+1 - and why the copy
// (not the caller's array) is what it slides.
void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (history == nullptr || qkv == nullptr || conv_w == nullptr || h == nullptr) return;
    if (channels <= 0 || n_tok < 1 || t_begin < 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("gdn_conv_l2_multi", stream);
    // ONE DISPATCH FOR THE WHOLE ROUND (the token is the grid's Y dimension).  `history` is NOT written
    // (`write_hist = 0`) and the running window is read from `history`/`qkv` directly as stream[t_begin+t ..
    // t_begin+t+3] - the CUDA multi's own window formula - instead of from a slid working copy, so no token
    // depends on another and no scratch is needed.  The per-token arithmetic is the single kernel's.
    strata::vulkan::fused_gdn_conv_l2_n(s, history, qkv, conv_w, h, channels, qk_heads, eps, t_begin, n_tok,
                                        /*write_hist=*/0);
}

// verify_kernels.hpp: `void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta,
//     const float* dt, const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, int n_tok,
//     void* stream);`  (verify.cpp:732).  Contract: "Bitwise `fused_gdn_ab` per column."  A plain loop - the
//     single-token kernel touches no state (the column is `x + t*n_embd`, the outputs `gate/beta + t*h_v`).
void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt,
                  const float* ssm_a, float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd <= 0 || h_v <= 0 || n_tok < 1) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("gdn_ab_multi", stream);
    // ONE DISPATCH FOR THE WHOLE ROUND (the token is the grid's Y dimension); the weights are token-invariant.
    strata::vulkan::fused_gdn_ab_n(s, x, w_alpha, w_beta, dt, ssm_a, gate, beta, n_embd, h_v, n_tok);
}

// verify_kernels.hpp: `void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate,
//     const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
//     const int32_t* n_keep, void* stream, int t_out_begin = 0);`  (verify.cpp:743/:748 verify, :1294 commit).
//     The loop bound is DEVICE data, so the loop is in the shader (`gdn_step_norm_multi.comp`, whose token body
//     is the single-token shader's own enumeration) - see this file's header note.
void gdn_step_norm_multi(float* state, const float* h, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    const int S = 128;   // the ported single-token kernel's own fixed S (the closing-norm tree needs it)
    if (state == nullptr || h == nullptr || gate == nullptr || beta == nullptr || z == nullptr || gamma == nullptr ||
        y == nullptr || h_k <= 0 || h_v <= 0 || n_tok < 1)
        return;
    if (h_v % h_k != 0)
        strata::vulkan::refuse("gdn_step_norm_multi", "h_v is not a multiple of h_k (the single-token contract)");
    if (conv_channels != 2 * S * h_k + S * h_v)
        strata::vulkan::refuse("gdn_step_norm_multi", "conv_channels is not 2*S*h_k + S*h_v (the q|k|v layout)");
    strata::vulkan::Stream& s = strata::vulkan::need_stream("gdn_step_norm_multi", stream);
    const uint64_t state_floats = (uint64_t) S * (uint64_t) h_v * (uint64_t) S;
    strata::vulkan::Buf st{}, sc{}, hb{}, gb{}, bb{}, zb{}, gmb{}, yb{}, kb{};
    if (!arena_resolve(s, state, state_floats * 4, st) ||
        !arena_resolve(s, h, (uint64_t) n_tok * (uint64_t) conv_channels * 4, hb) ||
        !arena_resolve(s, gate, (uint64_t) n_tok * (uint64_t) h_v * 4, gb) ||
        !arena_resolve(s, beta, (uint64_t) n_tok * (uint64_t) h_v * 4, bb) ||
        !arena_resolve(s, z, (uint64_t) n_tok * (uint64_t) h_v * (uint64_t) S * 4, zb) ||
        !arena_resolve(s, gamma, (uint64_t) S * 4, gmb) ||
        !arena_resolve(s, y, (uint64_t) n_tok * (uint64_t) h_v * (uint64_t) S * 4, yb))
        strata::vulkan::refuse("gdn_step_norm_multi", "a pointer is not inside this stream's arena");
    // the verify half runs on the stream's working state scratch; the commit half runs in place
    const int use_scratch = (n_keep == nullptr) ? 1 : 0;
    if (use_scratch) {
        float* work = reinterpret_cast<float*>(strata::vulkan::need_scratch(s, state_floats * 4));
        if (!arena_resolve(s, work, state_floats * 4, sc))
            strata::vulkan::refuse("gdn_step_norm_multi", "the working state is not inside this stream's arena");
    } else {
        sc = st;
    }
    if (n_keep != nullptr) {
        if (!arena_resolve(s, n_keep, 4, kb))
            strata::vulkan::refuse("gdn_step_norm_multi", "n_keep is not inside this stream's arena");
    } else {
        kb = strata::vulkan::dummy_buf(s);
    }
    struct { int32_t S, h_k, h_v; float eps; int32_t n_tok, t_out_begin, has_nkeep, use_scratch; } pc{
        S, h_k, h_v, eps, n_tok, t_out_begin, n_keep ? 1 : 0, use_scratch};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gdn_step_norm_multi.spv", 9, sizeof(pc));
    s.ctx->dispatch(p, {&st, &sc, &hb, &gb, &bb, &zb, &gmb, &yb, &kb}, &pc, sizeof(pc),
                   strata::vulkan::groups_for((uint64_t) h_v * (uint64_t) S));
}

// ---- `native_router_top10_multi` -> a LOOP over the gated `native_router_top10` -----------------------------
// native_router.hpp: `void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok,
//     void* stream);`  (verify.cpp:920 - the pre-router of every layer of the window).  Contract: "n_tok rows at
//     once (logits [n,512], ids/weights [n,10]); each row exactly as the single call."  512 experts and 10
//     outputs are the single call's own pinned geometry (`native_router.cu:88`), so the row stride is fixed.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (logits == nullptr || ids == nullptr || weights == nullptr || n_tok < 1) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("native_router_top10_multi", stream);
    // ONE DISPATCH FOR THE WHOLE ROUND: the shader's own token dimension is the grid x.  (The per-token
    // 40-byte ids/weights descriptor-offset refusal that used to guard this loop is gone with the loop.)
    strata::vulkan::router_top10_n(s, logits, ids, weights, n_tok, /*n_expert=*/512);
}

// ---- `wait_flag_ge` -> THE HANDSHAKE SEAM (deliverable A): a HOST boundary, never a spin ---------------------
// verify_kernels.hpp / verify_kernels.cu:496: the CUDA is `while (*flag < value) strata_spin_pause();` - a
// ONE-THREAD SPIN on host-mapped memory that the engine's `post` issues between the pool's plan/answer writes and
// the GPU ops that consume them (verify.cpp:1042/:1049/:1066).  This backend forbids a spinning kernel (a Vulkan
// spin cannot be preempted inside the 640 ms Battlemage GuC budget), and the flag is HOST memory the engine's own
// host loop (`Verifier::run`) raises AFTER the graph launch, so the wait is carried HOST-SIDE:
//
//   * WHAT THE FLAG IS: a u32 handshake word (`m_flagA_`/:1042, `m_flagB_`/:1049, `m_flag_`/:1066 - the mapped
//     twins of `h_flagA_`/`h_flagB_`/`h_flag_`), read by the GPU and written by the HOST at `verify.cpp:1506`
//     (`*h_flagA_ = want`), `raise_flag(h_flagB_, want)` and `*flag = want`, where `want = (l-lb_)*G+grp+1`.  The
//     host's writes do not happen until well after the launch: the device cannot wait for them.
//   * THE CARRY: under capture (the only way the window is recorded) this records a HOST BOUNDARY - the recording
//     is CUT here, and the driver submits the next segment only once `*flag >= value`, polling on the HOST thread
//     between SPLIT SUBMISSIONS.  The engine's own `cudaStreamQuery(cs_)` inside its `while (*seq < want)` spin
//     loop, and its closing `cudaStreamSynchronize(cs_)`, are what drive the segments out (the shim's launch is
//     asynchronous for a graph that recorded a boundary; a boundary-less step is unchanged).
//   * OUTSIDE capture it is the boundary's enforcement (as `doorbell_wait` is): the host must have answered
//     BEFORE the consumer is submitted.  An un-answered handoff is a LOUD REFUSAL, never a hang.
void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    if (flag == nullptr) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("wait_flag_ge", stream);
    if (s.ctx != nullptr && s.ctx->capturing()) {
        s.ctx->capture_boundary(flag, value);
        return;
    }
    const uint32_t cur = *(const volatile uint32_t*) flag;   // mapped, coherent host memory: a plain read
    if (cur < value) {
        std::fprintf(stderr,
                     "strata::kernels::wait_flag_ge: the host has not answered (flag %u < %u).  This backend never "
                     "asks the device to wait: the host writes the answer, THEN the consumer is submitted.  "
                     "Refusing rather than submitting a waiting kernel\n",
                     cur, value);
        std::exit(2);
    }
}

// ---- `copy_indexed` -> copy_indexed.comp (verify_kernels.cu:263) ------------------------------------------
// verify_kernels.hpp: `void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index,
//     int64_t n, void* stream);`  (verify.cpp:1311, the commit graph's PLE-history copy; also :1867).
//     The CUDA: `const int idx = *index; if (idx < 0) return; dst[i] = src[idx*stride + i]` for i in [0, n).
// `index` is DEVICE int32 (the engine's committed count, `commit_ + 1`), so the row it selects cannot be a
// descriptor offset resolved on the host: the whole source row table is bound and `idx*stride` is computed IN
// the shader.  The source is ARENA memory (`hist_snap_`), and the view covers the engine's own window maximum
// (`kVerifyMaxT` rows), so the shader can never index past a bound that was declared.
void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    if (n <= 0) return;
    if (dst == nullptr || src == nullptr || index == nullptr || stride <= 0) {
        std::fprintf(stderr, "strata::vulkan::copy_indexed: a null argument or a non-positive stride - refusing\n");
        std::exit(2);
    }
    strata::vulkan::Stream& s = strata::vulkan::need_stream("copy_indexed", stream);
    strata::vulkan::Buf dv{}, sv{}, iv{};
    if (!arena_resolve(s, dst, (uint64_t) n * 4, dv) ||
        !arena_resolve(s, src, (uint64_t) stride * (uint64_t) kVerifyMaxT * 4, sv) ||
        !arena_resolve(s, index, 4, iv))
        strata::vulkan::refuse("copy_indexed",
                               "a pointer is not inside this stream's arena, or the source row table does not "
                               "cover the window maximum (kVerifyMaxT rows)");
    struct { int32_t n, stride; } pc{(int32_t) n, (int32_t) stride};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/copy_indexed.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&dv, &sv, &iv}, &pc, sizeof(pc), strata::vulkan::groups_for((uint64_t) n));
}

// ---- `native_moe_combine_multi` -> a LOOP over the gated `native_moe_combine` ------------------------------
// native_moe.hpp: `void native_moe_combine_multi(const float* parts, const float* weights, const float* shared,
//     float* output, int64_t n_embd, int64_t k, int n_tok, void* stream);`  (verify.cpp:1083 - the window's
//     combine).  Contract: "n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each
//     as the single call."
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (n_embd <= 0 || k <= 0 || n_tok < 1) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("native_moe_combine_multi", stream);
    // ONE DISPATCH FOR THE WHOLE ROUND (the token is the flat grid's high digits).  (The per-token k*4-byte
    // weights descriptor-offset refusal that used to guard this loop is gone with the loop.)
    strata::vulkan::native_moe_combine_n(s, parts, weights, shared, output, n_embd, k, n_tok);
}

// ---- `shared_expert_multi` -> a LOOP over the shared expert's NATIVE path, per token ------------------------
// shared_expert.hpp: `void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16,
//     const NativeSharedWeights& nw, const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out,
//     int64_t n_embd, int64_t n_ff, void* stream);`  (verify.cpp:980 - the MoE block's shared expert, for the
//     whole window).  Contract: "the shared expert for n_tok <= 8 tokens ... with all three projections native:
//     multi-column MMVQ, so the weights are read once; every token is bitwise `shared_expert` on that token."
// The port carries the per-token contract (the one-weight-read is a COST property, stated as not carried) and
// repeats the single-token native path's five steps in the single-token ORDER: quantize x -> gate/up MMVQ ->
// SwiGLU on the gate vs up -> quantize the intermediate -> down MMVQ -> the BF16 scalar gate -> the row scale.
// The single-token `shared_expert` carves `gate`/`up`/`q8`/`g` from a `scratch` argument the multi does NOT
// have (the CUDA multi writes into the caller's `gate`/`up` and its own `nw.q8_1`), so the same .spv stages are
// driven here against the caller's buffers - and the gate case holds the two entry points to BITWISE equality
// on the same fixture, which is what keeps the two renderings honest.
void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out,
                         int64_t n_embd, int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8) return;
    if (n_embd <= 0 || n_ff <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("shared_expert_multi", stream);
    if (nw.q8_1 == nullptr || nw.gate_data == nullptr || nw.up_data == nullptr || nw.down_data == nullptr)
        strata::vulkan::refuse("shared_expert_multi",
                               "the multi contract requires all three projections native and a q8_1 scratch");
    if (!native_mmvq_supported(nw.gate_type) || !native_mmvq_supported(nw.up_type) ||
        !native_mmvq_supported(nw.down_type))
        strata::vulkan::refuse("shared_expert_multi", "a native projection's type has no shader on this backend");
    // ONE DISPATCH PER STAGE FOR THE WHOLE ROUND, not one per token.  The projections and the quantiser
    // already carry a COLUMN dimension (`native_mmvq`'s `ncols`, `native_quantize_q8_1`'s `ncols`), and that
    // dimension's layout is the single-column layout replicated - `iq1m_mmvq.comp`: `arow = c*arow_bytes`,
    // `y.v[c*n_out + row]`; `quantize_q8_1.comp`: `x.v[c*n_in + i]`, block `c*(n_in/32) + i/32` - so token t's
    // column is BITWISE the single-column call, which is exactly the multi's own contract ("every token is
    // bitwise `shared_expert` on that token").  The SwiGLU is flat in its element index and `gate`/`up`/`out`
    // are contiguous per token, so one call covers the window.  No per-token descriptor offset is needed, so
    // the descriptor-alignment refusal this loop used to carry is gone with the loop.
    const void* act = (nw.x_q8_1 != nullptr) ? nw.x_q8_1 : static_cast<const void*>(nw.q8_1);
    if (nw.x_q8_1 == nullptr) native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, act, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, act, up, (int) n_embd, (int) n_ff, n_tok, stream);
    strata::vulkan::swiglu_f32(s, gate, up, gate, (int64_t) n_tok * (int64_t) n_ff);
    // the down activation goes into the SAME scratch after both gate/up projections have read it; the
    // dispatches are ordered, so the overwrite is defined.
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    // the BF16 scalar gate and the row scale for the WHOLE window, one dispatch each (see the helpers' note)
    strata::vulkan::scalar_gate_f32(s, x_bf16, gate_inp_bf16, g, n_tok, n_embd);
    strata::vulkan::scale_rows(s, out, g, n_tok, n_embd);
}

// ---- `fetch_blobs` / `rebase_ptrs` - the P6 verify window's PCIe STAGING (verify.cpp:1053/:1054) -----------
//
// WHAT THEY DO (verify_kernels.cu:271-284, verify_kernels.hpp:72-74):
//   * `fetch_blobs(src, n, dst, blob_bytes, cap, stream)` gathers `*n` DEVICE-POINTED blobs of `blob_bytes`
//     bytes each - `src[k]` is a device address in the pinned complement (`expert_source.cpp:2083`
//     `P.ptr2[q] = d.src->device_alias(...)`) - into the VRAM staging region at `dst + k*blob_bytes`, as
//     coalesced 16-byte loads.  `per = blob_bytes/16` uint4s per blob; the CUDA launches a flat grid over
//     `*n * per` and early-returns if `cap <= 0`.
//   * `rebase_ptrs(ptr, n, base, bytes, stream)` then rewrites the DEVICE pointer table so the following
//     grouped-expert launch (`verify.cpp:1059`) reads the staged copies: `ptr[k] = base + k*bytes` for `k < *n`.
//
// WHICH FORM THIS RUN NEEDS - MEASURED FROM THE COUNTS, NOT ASSUMED.  The branch is taken under the DEFAULT
// `--pcie-mode auto` (`generate.cpp:471` default `"auto"`; `:5168` `set_pcie_mode(... : 2)` -> `pcie_mode == 2`).
// The count that gates the work is `P.counts[2] = fetches` (`expert_source.cpp:2086`), and with this box's PCIe
// probe (0.1 GB/s -> `pcie_frac 0.00`):
//     `pcie_num   = (int)(0.00*256 + 0.5) = 0`                       (generate.cpp:7850)
//     `pcie_ok    = (pcie_num > 0) && ... = FALSE`                   (expert_source.cpp:2034)
//     `m          = pcie_ok ? (nmiss*pcie_num)>>8 : 0 = 0`           (expert_source.cpp:2035)
//     `miss_rank >= nmiss - m`  =>  `miss_rank >= nmiss`, never true for miss_rank in [0, nmiss)
//   so `fetches = 0` and **`counts[2] == 0` on EVERY group** - the CUDA pair is an EMPTY no-op
//   (`fetch_blobs_kernel` is launched with `total = 0`; `rebase_ptrs_kernel` writes nothing because `k < 0` is
//   false for all 128 threads).  The measured run agrees: `experts streamed 375 (0 by DMA)`.
//
// SO BOTH ARE CARRIED AS THEIR GENERAL, DEVICE-SIDE FORMS - which is STRICTLY SAFER than a host-side empty
// shortcut, because `*n` is DEVICE data read AT SUBMIT: with `*n == 0` every invocation returns before any load
// or store (a faithful empty no-op), and with `*n > 0` the gather and the rewrite actually run.  A host-side
// shortcut that recorded nothing would be right for this configuration and a SILENT wrong answer the moment
// `pcie_frac` moved.  NO KERNEL WAITS OR SPINS.
//
// `rebase_ptrs`'s stored address is written as a lo/hi uint32 PAIR (glslang has no 64-bit buffer index; the
// `ptr_to_off.spv` technique) with the 64-bit ADD done in the shader, so the stored synthetic pointer is
// byte-for-byte what the CUDA stores.
static constexpr uint64_t kBlobWinBytes = (1ull << 32) - (64ull << 20);   // 4 GiB window the source view advances by
static constexpr uint64_t kBlobFetchCap = 64;   // the engine's own hard cap on a layer's PCIe fetches (expert_source.cpp:2054)

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;                                       // the CUDA's own early return (verify_kernels.cu:364)
    if (blob_bytes <= 0 || blob_bytes % 16 != 0) {              // the CUDA exits(1) on a non-multiple of 16 (:365)
        std::fprintf(stderr, "strata::vulkan::fetch_blobs: blob_bytes %lld is not a positive multiple of 16 "
                             "(the CUDA's own precondition) - refusing\n", (long long) blob_bytes);
        std::exit(2);
    }
    const uint64_t per = (uint64_t) blob_bytes / 16;            // uint4 chunks per blob
    if (per == 0 || per > 0xFFFFFFFFull) {
        std::fprintf(stderr, "strata::vulkan::fetch_blobs: blob_bytes/16 = %llu does not fit the shader's 32-bit per\n",
                     (unsigned long long) per);
        std::exit(2);
    }
    strata::vulkan::Stream& s = strata::vulkan::need_stream("fetch_blobs", stream);
    strata::vulkan::Buf b_ptr{}, b_n{}, b_dst{};
    if (!strata::vulkan::arena_resolve(s, src, (uint64_t) cap * 8, b_ptr) ||
        !strata::vulkan::arena_resolve(s, n, 4, b_n) ||
        !strata::vulkan::arena_resolve(s, dst, (uint64_t) cap * (uint64_t) blob_bytes, b_dst))
        strata::vulkan::refuse("fetch_blobs",
                               "the pointer table, the count or the staging destination is not inside this stream's arena");
    const uint64_t base = strata::vulkan::Stream::kArenaBase;
    struct { uint32_t base_lo, base_hi, win_bytes, win_id, per; } pc{
        (uint32_t) (base & 0xFFFFFFFFu), (uint32_t) (base >> 32), (uint32_t) kBlobWinBytes, 0u, (uint32_t) per};
    const uint32_t nwin = (uint32_t) ((s.arena_bytes + kBlobWinBytes - 1) / kBlobWinBytes);
    for (uint32_t w = 0; w < nwin; ++w) {
        const uint64_t wbase = (uint64_t) w * kBlobWinBytes;
        if (wbase >= s.arena_bytes) break;
        strata::vulkan::Buf wview = strata::vulkan::view(s.arena, wbase);
        pc.win_id = w;
        VkPipeline p = s.ctx->pipeline(s.spv_dir + "/fetch_blobs.spv", 4, sizeof(pc));
        // RECORDS under capture (the count is re-read at every replay), submits+waits otherwise.
        s.ctx->dispatch(p, {&b_ptr, &b_n, &wview, &b_dst}, &pc, sizeof(pc), (uint32_t) cap);
    }
}

// ---- `resident_plan` - the P6 verify window's ALL-RESIDENT per-group plan (verify.cpp:938) ------------------
//
// verify_kernels.hpp: `void resident_plan(ids, n_entries, k, res_layer, n_expert, cache_base, slot_off, blob,
//     plan, capx, skip, ring, stream);`  Contract: "One group's plan, built on the device when every routed
//     expert of its n*k entries is resident: the host pool's layout (counts | start | dst | tok | pad | ptr |
//     ptr2 | start2, `capx` entries) and order (distinct experts in routing order, their entries ascending),
//     no PCIe groups.  *skip = ring when it did, else 0."
//
// THE ONE DANGEROUS LINE in this whole port is `ptr[grp] = cache_base + slot_off[slot]` (verify_kernels.cu:529):
// a 64-bit ADDRESS built by adding a base to a per-slot OFFSET.  Every base+offset defect this port has shipped
// was of exactly this shape (a `view` that SET the offset instead of adding the base; an element-size mismatch;
// a wrong window), so this wrapper treats it as an address-arithmetic seam and the gate PROVES THE POINTER
// VALUES, not just the counts.  The shader reads/writes the 64-bit values as lo/hi uint32 pairs (glslang 15.1
// has no 64-bit buffer index; the `ptr_to_off.spv` technique) and does the ADD in int64.
//
// THE PLAN IS BOUND WHOLE (offset 0) and indexed as a uint array, so the ptr region's own byte offset (which is
// only 8-byte aligned for the engine's capx) is never a DESCRIPTOR OFFSET - it cannot trip
// `minStorageBufferOffsetAlignment`.
//
// `slot_off_d_` (cudaMalloc'd by Verifier::init, `hits.n_slots * 8` bytes) arrives with NO length argument, so
// it is resolved against the LIVE tail of the arena: the shader reads `slot_off[slot]` for the slots the engine's
// `res` table names, all of which lie inside that allocation (the CUDA reads the same, unguarded).
void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    const int kResidentPlanMax = 128;                       // the CUDA's own block size (verify_kernels.cu:504)
    if (n_entries < 0 || k <= 0 || n_expert <= 0 || capx <= 0)
        strata::vulkan::refuse("resident_plan", "a non-positive geometry (k, n_expert or capx)");
    if (n_entries > kResidentPlanMax)
        strata::vulkan::refuse("resident_plan",
                               "n_entries exceeds the CUDA's one-block bound (kResidentPlanMax = 128)");
    if (ids == nullptr || res_layer == nullptr || plan == nullptr)
        strata::vulkan::refuse("resident_plan", "a null ids/res/plan pointer");
    strata::vulkan::Stream& s = strata::vulkan::need_stream("resident_plan", stream);
    strata::vulkan::Buf b_ids{}, b_res{}, b_so{}, b_plan{}, b_skip{};
    if (!arena_resolve(s, ids, (uint64_t) n_entries * 4, b_ids) ||
        !arena_resolve(s, res_layer, (uint64_t) n_expert * 4, b_res))
        strata::vulkan::refuse("resident_plan", "ids or res_layer is not inside this stream's arena");
    // the plan region (verify.cpp:385-387): counts(4) | start(capx+1) | dst(capx) | tok(capx) | pad |
    // ptr(capx u64) | ptr2(capx u64) | start2(capx+1).  This is the engine's own per-group size, recomputed
    // here so the bound checked is the bound the writer uses.
    const uint64_t i32 = (uint64_t) (4 + (capx + 1) + 2 * capx);
    const uint64_t ptr_off = (i32 + 1) & ~(uint64_t) 1;
    const uint64_t plan_i32 = ptr_off + 4 * (uint64_t) capx + ((uint64_t) capx + 1) + 1;
    if (!arena_resolve(s, plan, plan_i32 * 4, b_plan))
        strata::vulkan::refuse("resident_plan", "the plan buffer does not cover the engine's own layout");
    if (slot_off != nullptr) {
        const uintptr_t a = reinterpret_cast<uintptr_t>(slot_off);
        if (a < strata::vulkan::Stream::kArenaBase)
            strata::vulkan::refuse("resident_plan", "slot_off is not inside this stream's arena");
        const uint64_t off = (uint64_t) (a - strata::vulkan::Stream::kArenaBase);
        if (off > s.bump || !arena_resolve(s, slot_off, s.bump - off, b_so))
            strata::vulkan::refuse("resident_plan", "slot_off is not inside this stream's arena");
    } else {
        b_so = strata::vulkan::dummy_buf(s);
    }
    if (skip != nullptr) {
        if (!arena_resolve(s, skip, 4, b_skip))
            strata::vulkan::refuse("resident_plan", "the skip word is not inside this stream's arena");
    } else {
        b_skip = strata::vulkan::dummy_buf(s);
    }
    const uint64_t base = (uint64_t) (uintptr_t) cache_base;
    const uint64_t bb = (uint64_t) blob;
    struct {
        uint32_t base_lo, base_hi, blob_lo, blob_hi;
        int32_t n, k, n_expert, capx, has_slot_off, has_skip;
        uint32_t ring;
    } pc{(uint32_t) (base & 0xFFFFFFFFu), (uint32_t) (base >> 32),
         (uint32_t) (bb & 0xFFFFFFFFu), (uint32_t) (bb >> 32),
         n_entries, k, n_expert, (int32_t) capx, slot_off ? 1 : 0, skip ? 1 : 0, ring};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/resident_plan.spv", 5, sizeof(pc));
    s.ctx->dispatch(p, {&b_ids, &b_res, &b_so, &b_plan, &b_skip}, &pc, sizeof(pc), 1u);
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    if (blob_bytes <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("rebase_ptrs", stream);
    strata::vulkan::Buf b_ptr{}, b_n{};
    // The table has `capx = max_t_*K` entries; the engine's `fetches` is capped at 64, so a 64-entry bound
    // covers every write.  A shorter window sizes the table below that, so fall back to a 16-entry bound.
    if (!strata::vulkan::arena_resolve(s, ptr, kBlobFetchCap * 8, b_ptr)) {
        if (!strata::vulkan::arena_resolve(s, ptr, 16 * 8, b_ptr) ||
            !strata::vulkan::arena_resolve(s, n, 4, b_n))
            strata::vulkan::refuse("rebase_ptrs", "the pointer table or the count is not inside this stream's arena");
    } else if (!strata::vulkan::arena_resolve(s, n, 4, b_n)) {
        strata::vulkan::refuse("rebase_ptrs", "the count is not inside this stream's arena");
    }
    const uint64_t b = (uint64_t) (uintptr_t) base;             // the staging region's synthetic device address
    const uint64_t bytes = (uint64_t) blob_bytes;
    struct { uint32_t base_lo, base_hi, bytes_lo, bytes_hi; } pc{
        (uint32_t) (b & 0xFFFFFFFFu), (uint32_t) (b >> 32),
        (uint32_t) (bytes & 0xFFFFFFFFu), (uint32_t) (bytes >> 32)};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/rebase_ptrs.spv", 2, sizeof(pc));
    s.ctx->dispatch(p, {&b_ptr, &b_n}, &pc, sizeof(pc), 1u);    // one 256-thread group covers k < *n <= 64
}

}  // namespace strata::kernels
