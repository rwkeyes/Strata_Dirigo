// vulkan/src/kernels/qsa_vk.cpp - the Vulkan backend's ATTENTION / QSA / rope entry points (the non-GDN half).
//
// ============================================================================================================
// WHICH EIGHT, AND WHY THESE EIGHT (derived from the engine's own body, not from the plan's list)
// ============================================================================================================
//
// `qsa_layer` (`src/core/layer.cpp:876`) is the NON-GDN layer body - the analogue of `gdn_layer` for the 12 QSA
// layers (`qsa_interval = 4`, so layers 3,7,...,47), reached from `block_layer_pre` (`layer.cpp:1259`:
// `if (qsa) qsa_layer(...) else gdn_layer(...)`).  This TU wires the first eight entry points `qsa_layer`'s own
// body reaches that this tree can actually DISPATCH, in the order its call sites appear (the plan's §3 list
// names the rows in prose and is NOT an order).  THE ORDER IS THE READING: within the two native/legacy stages
// the branches are alternatives, so "the order the body reaches them" is the order the calls appear.
//
//   1. `native_qsa_rms_norm_weighted`  layer.cpp:879   `normalize_rotate`, the NATIVE norm (native_qsa_enabled)
//   2. `native_rope_apply`             layer.cpp:881   `normalize_rotate`, the NATIVE rope (native_rope_enabled)
//   3. `rope_neox_apply`               layer.cpp:882   `normalize_rotate`, the LEGACY rope (the else branch)
//   4. `qsa_block_scores`              layer.cpp:970   the g_fast_select block scores (qsa_select.hpp)
//   5. `qsa_block_topk`                layer.cpp:971   the g_fast_select weighted top-k
//   6. `native_qsa_gate_apply`         layer.cpp:1010  the gate, NATIVE member (native_qsa_enabled)
//   7. `qsa_gate_apply_f32`            layer.cpp:1011  the gate, LEGACY member (the else branch)
//   8. `native_router_top10`           layer.cpp:370   moe_route's router (native_router_enabled; "the expert
//                                                       routing"), the NEXT stage the non-GDN body reaches -
//                                                       `block_layer_pre` run4 calls `moe_route` after run1's
//                                                       `qsa_layer`.
//
// WHY THESE AND NOT THE WHOLE GROUP (the group is 36).  PLE and GR (`ple_block`, `ple_history_advance`,
// `gr_read`, `gr_write`, `fused_gr_*`) sit in `block_layer_pre`'s shared stages and run for EVERY layer, GDN or
// QSA - they are not the non-GDN body and are their own increment.  The stage-3 attention entry
// `qsa_decode_attn_step` (layer.cpp:980) has NO shader in this tree and its contract (`qsa_decode_attn.hpp`)
// reads the KV POOLS through the page table - a DIFFERENT kernel from the map's claimed `attn_decode_short`,
// whose own header says it is `native_flash_attn_short_step` (a gathered [capacity,2,256] f16 WINDOW).  It is
// wired here as the NINTH entry point (below), driving the new `qsa_decode_attn.spv`.  Its sibling
// `native_flash_attn_short_step` (:995) remains UNWIRED (a diagnostic-only branch, `native_flash_attn_short`
// default false); `qsa_index_step`, `topk_512_step`, `qsa_attend_step` and `native_qsa_indexer_append` are
// PORT-MAP `todo` (no shader either).
//
// THE TENTH AND ELEVENTH (this batch): `qsa_step_fill` (:908, PURE HOST - the four derived per-token counts)
// and `indexer_key_append` (:948, the LEGACY indexer member, `indexer_key_append.spv`).  Both are reached by
// `qsa_layer`'s own body, in that order; both were STILL UNDEFINED in this tree before this batch (the
// class-A triage recorded `indexer_key_append` "LANDED" against a shader THIS tree had but a wrapper it did
// not).  Neither is a no-shader row: one needs no shader (host), one has had one.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each symbol below is the thin wrapper already declared in
// `include/strata/kernels/{native_qsa,native_rope,rope,qsa,qsa_select,native_router}.hpp`; this TU answers the
// `strata::kernels::` symbol the wrapper calls.  On a CUDA/HIP build `src/kernels/cuda/*.cu` answer them; on
// this build THIS file does.  Each body resolves the engine's raw device pointers to arena views
// (`arena_resolve`), takes the pipeline the device layer caches for the shader's own signature, and dispatches -
// the same shape as `gdn_vk.cpp` / `matvec_vk.cpp` - and the shader each drives is the one the port's numeric
// gate has already gated:
//
//     native_qsa_rms_norm_weighted -> native_qsa_rms_norm_weighted.spv (case_native_qsa_rms_norm_weighted)
//     native_rope_apply            -> native_rope_apply.spv            (case_native_rope_apply)
//     rope_neox_apply              -> rope_neox.spv                    (case_rope)
//     qsa_block_scores             -> qsa_block_scores.spv             (case_qsa_select)
//     qsa_block_topk               -> qsa_block_topk.spv               (case_qsa_select)
//     native_qsa_gate_apply        -> native_qsa_gate_apply.spv        (case_native_qsa_gate_apply)
//     qsa_gate_apply_f32           -> qsa_gate_apply_f32.spv           (case_qsa_gate_apply_f32)
//     native_router_top10          -> native_router_top10.spv          (case_native_router_top10)
//
// The gate's `case_*_entry` cases re-run each with the ENGINE WRAPPER and compare BITWISE to that same shader
// path AND against the case's explicit oracle, so this file's claim is not "it compiles" but "the wrapper's
// answer equals the ported shader's answer".
//
// `rope_scaling()` is a `host` row: the rope CONSTANTS.  `rope_scaling.hpp` declares the process config,
// `rope_scaling_set` once at startup and `rope_scaling()` at every analytic rope launch; on a CUDA build
// `src/kernels/cuda/rope_scaling.cu` owns the storage, and a Vulkan build compiles no such file.  The one host
// row this batch answers is therefore the storage plus its setter, so `layer.cpp:881`'s `rope_scaling()` (and
// the engine's startup `rope_scaling_set`) link against THIS backend's own config.
#if !defined(STRATA_ENABLE_VULKAN)
#error "qsa_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/native_qsa.hpp"     // native_qsa_rms_norm_weighted / native_qsa_gate_apply
#include "strata/kernels/native_rope.hpp"    // native_rope_apply
#include "strata/kernels/rope.hpp"           // rope_neox_apply
#include "strata/kernels/mrope.hpp"          // rope_table_for (the STRATA_ROPE_TABLE guard in native_rope_apply)
#include "strata/kernels/rope_scaling.hpp"   // RopeScaling / rope_scaling / rope_scaling_set
#include "strata/kernels/qsa.hpp"            // QsaShapes / qsa_step_fill / kStepCount
#include "strata/kernels/qsa_select.hpp"     // qsa_block_scores / qsa_block_topk
#include "strata/kernels/qsa_decode_attn.hpp"  // qsa_decode_attn_step / qsa_decode_attn_scratch_floats
#include "strata/kernels/native_qsa_indexer.hpp"  // native_qsa_indexer_append / _batch
#include "strata/kernels/native_router.hpp"  // native_router_top10
#include "strata/vulkan/vk_backend.hpp"      // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                      // the arena + pointer->buffer resolution
#include "vk_multi.hpp"                      // the verify window's one-dispatch-per-round forms

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace strata::vulkan {

// Every shader here is `local_size_x = 256` (checked against the host's constant by the gate's census).
static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// The port's rule: refuse, never degrade.  A pointer outside the arena, or a shape the engine's own wrapper
// contract rejects, is a loud exit rather than a silently wrong binding.
[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

// A Vulkan descriptor cannot be null: the two rope shaders ALWAYS bind an mrope table (binding 3 / 5) even when
// `mrope == 0` and its contents are never read.  A per-dispatch allocation would exhaust the arena (it never
// decreases), so the sentinel lives with the stream - the `iq_grids`/`cvec_tables` precedent.  Sixteen ints is
// more than the shader can index with mrope 0.
static Buf& dummy_buf(Stream& s) {
    if (s.dummy.buffer == VK_NULL_HANDLE) {
        s.dummy = s.ctx->alloc(64);
        const int32_t zero[16] = {0};
        s.ctx->write(s.dummy, zero, sizeof(zero));
    }
    return s.dummy;
}

// ============================================================================================================
// THE EIGHT (each a bind-and-dispatch over the shader its gate case already proved)
// ============================================================================================================

// ---- 1. `native_qsa_rms_norm_weighted` -> native_qsa_rms_norm_weighted.spv (IN ro, G ro, OUT rw; push {int
//        rows; int cols; float eps}; one workgroup per row).  The engine wrapper's argument order is
//        (n_cols, n_rows); the shader's push is (rows, cols) - kept straight here.  In-place (output == input) is
//        the engine's own call shape and is exact: each thread reads and writes its own element (the source's
//        comment).  No `cols == 128` restriction; the real artifact is 2560/256.
void native_qsa_rms_norm_weighted(Stream& s, const float* input, const float* gamma, float* output, int n_cols,
                                  int n_rows, float epsilon) {
    if (n_cols <= 0 || n_rows <= 0) return;
    const uint64_t n = (uint64_t) n_cols * (uint64_t) n_rows;
    Buf iv{}, gv{}, ov{};
    if (!arena_resolve(s, input, n * 4, iv) || !arena_resolve(s, gamma, (uint64_t) n_cols * 4, gv) ||
        !arena_resolve(s, output, n * 4, ov))
        refuse("native_qsa_rms_norm_weighted", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_qsa_rms_norm_weighted.spv", 3, 12);
    struct Push {
        int32_t rows;
        int32_t cols;
        float eps;
    } pc{};
    pc.rows = n_rows;
    pc.cols = n_cols;
    pc.eps = epsilon;
    s.ctx->dispatch(pipe, {&iv, &gv, &ov}, &pc, sizeof(pc), (uint32_t) n_rows);
}

// ---- 2. `native_rope_apply` -> native_rope_apply.spv (X ro, OUT rw, POS ro, MROPETAB ro; push {int rows;
//        head_dim; n_rot; mrope; float theta_scale; freq_scale; corr_low; corr_high; ext_factor; mscale}).
//        The NATIVE body computes the angle ON DEVICE in f32; `theta_scale = powf(freq_base, -2/n_rot)` is the
//        ONE value computed on the host (the source does the same), and the scaling constants come from
//        `RopeScaling::kernel_args` - `mscale` is the RAW `attn_factor`, NOT `RopeScaling::mscale()` (ggml's
//        kernels apply the log term inside the helper when `ext_factor != 0`, exactly as the shader does).
//        Grid (ceil((head_dim/2)/256), rows): the pairs are x, the row is y (the source's mapping).
void native_rope_apply(Stream& s, const float* x, float* out, int rows, int head_dim, int n_rot,
                       const strata::kernels::RopeScaling& scaling, const int* positions) {
    if (rows <= 0 || head_dim <= 0 || n_rot <= 0) return;
    // STRATA_ROPE_TABLE=1 with a table built for THIS scaling: the engine's <true> rope path would rotate by the
    // table's exact float64 angles.  This backend has no table-reading rope shader (native_rope_apply.spv
    // computes the angle on device; see rope_vk.cpp), so the analytic arithmetic below would SILENTLY differ from
    // the engine by ~0.0014 rad at 32K.  Refuse the one configuration the port cannot honour rather than diverge;
    // the default (no table) is bit-for-bit the engine's <false> branch.
    if (strata::kernels::rope_table_for(scaling).cos != nullptr)
        refuse("native_rope_apply", "STRATA_ROPE_TABLE=1 with a matching table: this backend has no table-reading "
                                    "rope shader (the analytic path would silently differ from the engine's)");
    const uint64_t n = (uint64_t) rows * (uint64_t) head_dim;
    Buf xv{}, ov{}, pv{}, mv{};
    if (!arena_resolve(s, x, n * 4, xv) || !arena_resolve(s, out, n * 4, ov) ||
        !arena_resolve(s, positions, (uint64_t) rows * 4, pv))
        refuse("native_rope_apply", "a pointer is not inside this stream's arena");
    mv = dummy_buf(s);
    const strata::kernels::RopeKernelArgs ka = scaling.kernel_args(n_rot);
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_rope_apply.spv", 4, 40);
    struct Push {
        int32_t rows;
        int32_t head_dim;
        int32_t n_rot;
        int32_t mrope;
        float theta_scale;
        float freq_scale;
        float corr_low;
        float corr_high;
        float ext_factor;
        float mscale;
    } pc{};
    pc.rows = rows;
    pc.head_dim = head_dim;
    pc.n_rot = n_rot;
    pc.mrope = 0;   // text: the positions ARE rotary positions (the engine's decode call)
    pc.theta_scale = std::pow((float) scaling.freq_base, -2.0f / (float) n_rot);
    pc.freq_scale = ka.freq_scale;
    pc.corr_low = ka.corr_low;
    pc.corr_high = ka.corr_high;
    pc.ext_factor = ka.ext_factor;
    pc.mscale = ka.attn_factor;
    const uint32_t gx = (uint32_t) ((head_dim / 2 + (int) kLocalSize - 1) / (int) kLocalSize);
    s.ctx->dispatch(pipe, {&xv, &ov, &pv, &mv}, &pc, sizeof(pc), gx, (uint32_t) rows);
}

// ---- 3. `rope_neox_apply` -> rope_neox.spv (X ro, OUT rw, COS ro, SIN ro, POS ro, MROPETAB ro; push {int
//        rows; head_dim; n_rot; mrope}).  One thread per ROW.  The cos/sin table is host-built and its LENGTH is
//        not an argument - `arena_resolve` uses its `bytes` only as the in-arena range check (the descriptor
//        binds VK_WHOLE_SIZE from the offset), so one pair row is the honest minimum to check against.  The
//        engine's decode call passes no mrope table (mrope 0), so the sentinel is bound.
void rope_neox_apply(Stream& s, const float* x, float* out, int64_t rows, int head_dim, int n_rot,
                     const float* cos_tab, const float* sin_tab, const int* pos) {
    if (rows <= 0 || head_dim <= 0 || n_rot <= 0) return;
    const uint64_t n = (uint64_t) rows * (uint64_t) head_dim;
    const uint64_t one_pair = (uint64_t) (n_rot / 2) * 4;   // the table's minimum: one row of pairs
    Buf xv{}, ov{}, cv{}, sv{}, pv{}, mv{};
    if (!arena_resolve(s, x, n * 4, xv) || !arena_resolve(s, out, n * 4, ov) ||
        !arena_resolve(s, cos_tab, one_pair, cv) || !arena_resolve(s, sin_tab, one_pair, sv) ||
        !arena_resolve(s, pos, (uint64_t) rows * 4, pv))
        refuse("rope_neox_apply", "a pointer is not inside this stream's arena");
    mv = dummy_buf(s);
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/rope_neox.spv", 6, 16);
    struct Push {
        int32_t rows;
        int32_t head_dim;
        int32_t n_rot;
        int32_t mrope;
    } pc{};
    pc.rows = (int32_t) rows;
    pc.head_dim = head_dim;
    pc.n_rot = n_rot;
    pc.mrope = 0;
    s.ctx->dispatch(pipe, {&xv, &ov, &cv, &sv, &pv, &mv}, &pc, sizeof(pc), groups_for((uint64_t) rows));
}

// ---- 4. `qsa_block_scores` -> qsa_block_scores.spv (POOLED ro, DEAD ro, QIDX ro, STEPS ro, SCORES rw; push
//        {int max_blocks}).  Grid (max_blocks, nq): one workgroup per (query, block).  `active_blocks > 0`
//        launches only that many x-groups (the header's perf-review contract; -1 means the capacity).  The
//        shader hardcodes R=4 / IDX_DIM=128 / IDX_HEADS=4, so a shape that disagrees is a loud refusal rather
//        than a read past the pooled rows.
void qsa_block_scores(Stream& s, const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                      int64_t nq, int64_t max_blocks, const strata::kernels::QsaShapes& sh, float* scores, int64_t active_blocks) {
    if (nq <= 0 || max_blocks <= 0) return;
    if (sh.idx_block != 4 || sh.idx_dim != 128 || sh.idx_n_head != 4)
        refuse("qsa_block_scores", "the shader fixes idx_block=4, idx_dim=128, idx_n_head=4");
    Buf pv{}, dv{}, qv{}, sv{}, ov{};
    if (!arena_resolve(s, pooled, (uint64_t) max_blocks * 128 * 4, pv) ||
        !arena_resolve(s, dead, 128 * 4, dv) ||
        !arena_resolve(s, q_idx, (uint64_t) nq * 4 * 128 * 4, qv) ||
        !arena_resolve(s, steps, (uint64_t) nq * 4 * 4, sv) ||
        !arena_resolve(s, scores, (uint64_t) nq * (uint64_t) max_blocks * 4, ov))
        refuse("qsa_block_scores", "a pointer is not inside this stream's arena");
    const uint32_t gx = (active_blocks > 0) ? (uint32_t) active_blocks : (uint32_t) max_blocks;
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/qsa_block_scores.spv", 5, 4);
    struct Push {
        int32_t max_blocks;
    } pc{};
    pc.max_blocks = (int32_t) max_blocks;
    s.ctx->dispatch(pipe, {&pv, &dv, &qv, &sv, &ov}, &pc, sizeof(pc), gx, (uint32_t) nq);
}

// ---- 5. `qsa_block_topk` -> qsa_block_topk.spv (SCORES ro, STEPS ro, IDS rw; push {int max_blocks; int cap}).
//        One workgroup per query; the ids are emitted ASCENDING.  `active_blocks` is the header's bound on the
//        largest n_bid+1 of the call; the port's shader is the reference kernel and sizes nothing by it, so it
//        is accepted and ignored (its grid is sized by the query count, exactly as the case drives it).
void qsa_block_topk(Stream& s, const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
                    int64_t cap, const strata::kernels::QsaShapes& sh, int32_t* ids, int64_t active_blocks) {
    (void) active_blocks;
    if (nq <= 0 || max_blocks <= 0 || cap <= 0) return;
    if (sh.idx_block != 4)
        refuse("qsa_block_topk", "the shader fixes R = idx_block = 4");
    Buf sv{}, tv{}, ov{};
    if (!arena_resolve(s, scores, (uint64_t) nq * (uint64_t) max_blocks * 4, sv) ||
        !arena_resolve(s, steps, (uint64_t) nq * 4 * 4, tv) ||
        !arena_resolve(s, ids, (uint64_t) nq * (uint64_t) cap * 4, ov))
        refuse("qsa_block_topk", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/qsa_block_topk.spv", 3, 8);
    struct Push {
        int32_t max_blocks;
        int32_t cap;
    } pc{};
    pc.max_blocks = (int32_t) max_blocks;
    pc.cap = (int32_t) cap;
    s.ctx->dispatch(pipe, {&sv, &tv, &ov}, &pc, sizeof(pc), (uint32_t) nq);
}

// ---- 6. `native_qsa_gate_apply` -> native_qsa_gate_apply.spv (ATTN ro, QFULL ro, OUT rw; push {int n_head;
//        head_dim}; one thread per (head, dim)).  `out = attn * sigmoid(second-half gate)`, all f32.  One
//        surplus group is dispatched so a missing element guard is visible (the shader has none - the CUDA's
//        256-thread grid is exact, and the case's fixture uses the same surplus).
void native_qsa_gate_apply(Stream& s, const float* attn, const float* q_full, float* output, int n_head,
                           int head_dim) {
    if (n_head <= 0 || head_dim <= 0) return;
    const uint64_t n = (uint64_t) n_head * (uint64_t) head_dim;
    Buf av{}, qv{}, ov{};
    if (!arena_resolve(s, attn, n * 4, av) || !arena_resolve(s, q_full, n * 2 * 4, qv) ||
        !arena_resolve(s, output, n * 4, ov))
        refuse("native_qsa_gate_apply", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_qsa_gate_apply.spv", 3, 8);
    struct Push {
        int32_t n_head;
        int32_t head_dim;
    } pc{};
    pc.n_head = n_head;
    pc.head_dim = head_dim;
    s.ctx->dispatch(pipe, {&av, &qv, &ov}, &pc, sizeof(pc), groups_for(n) + 1u);
}

// ---- 7. `qsa_gate_apply_f32` -> qsa_gate_apply_f32.spv (same layout as #6).  The LEGACY sibling: the same RULE,
//        the same f32 arithmetic on this target (no shaderFloat64).  `head_dim` comes from the QsaShapes the
//        engine passes; the shader's push is the same {n_head, head_dim} pair.
void qsa_gate_apply_f32(Stream& s, const float* attn, const float* q_full, const strata::kernels::QsaShapes& sh, float* output) {
    const int n_head = (int) sh.n_head, head_dim = (int) sh.head_dim;
    if (n_head <= 0 || head_dim <= 0) return;
    const uint64_t n = (uint64_t) n_head * (uint64_t) head_dim;
    Buf av{}, qv{}, ov{};
    if (!arena_resolve(s, attn, n * 4, av) || !arena_resolve(s, q_full, n * 2 * 4, qv) ||
        !arena_resolve(s, output, n * 4, ov))
        refuse("qsa_gate_apply_f32", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/qsa_gate_apply_f32.spv", 3, 8);
    struct Push {
        int32_t n_head;
        int32_t head_dim;
    } pc{};
    pc.n_head = n_head;
    pc.head_dim = head_dim;
    s.ctx->dispatch(pipe, {&av, &qv, &ov}, &pc, sizeof(pc), groups_for(n) + 1u);
}

// ---- 8. `native_router_top10` -> native_router_top10.spv (LOGITS ro, IDS rw, WEIGHTS rw; push {n_tokens,
//        n_expert}).  ONE token per call (the engine's moe_route calls it per token).  Top-10, softmax, ggml's
//        2^-14 lower clamp - all fixed by the shader; the expert width comes from the push constant (512 here,
//        the canonical member's own geometry, but the shader is width-general - see `router_top10_n`).
//        `n_tokens = 1` and one workgroup, exactly as the case's single-token arm drives it.
void native_router_top10(Stream& s, const float* logits, int32_t* ids, float* weights) {
    router_top10_n(s, logits, ids, weights, /*n_tok=*/1, /*n_expert=*/512);
}

// THE MULTI.  `native_router_top10.spv` ALREADY carries a token dimension - `gl_WorkGroupID.x` is the token
// and the push constant is `{n_tokens, n_expert}` - so the window's whole group is ONE dispatch of n_tok
// workgroups instead of n_tok dispatches of one.  Each workgroup's arithmetic (including its shared `nr_*`
// arrays and every barrier) is per-token and untouched, so this is bitwise the single call per token.
//
// `n_expert` is passed in: the shader's ACTIVE expert width is a push constant, so the canonical 512 form and a
// narrower (e.g. 256-expert) model run the SAME shader at their own width.  The shared `rs_*`/`nr_*` arrays are
// sized for the 512 CAP, so the caller must keep n_expert <= 512 (the wrapper's `router_top10_uses_native`
// predicate owns that check).
void router_top10_n(Stream& s, const float* logits, int32_t* ids, float* weights, int64_t n_tok, int n_expert) {
    if (n_tok < 1 || n_expert < 1 || n_expert > 512) return;
    Buf lv{}, iv{}, wv{};
    if (!arena_resolve(s, logits, (uint64_t) n_tok * (uint64_t) n_expert * 4, lv) ||
        !arena_resolve(s, ids, (uint64_t) n_tok * 10ull * 4, iv) ||
        !arena_resolve(s, weights, (uint64_t) n_tok * 10ull * 4, wv))
        refuse("native_router_top10", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_router_top10.spv", 3, 8);
    struct Push {
        int32_t n_tokens, n_expert;
    } pc{};
    pc.n_tokens = (int32_t) n_tok;
    pc.n_expert = n_expert;
    s.ctx->dispatch(pipe, {&lv, &iv, &wv}, &pc, sizeof(pc), (uint32_t) n_tok);
}

// ---- 9. `qsa_decode_attn_step` -> qsa_decode_attn.spv.  THE DEFAULT QSA DECODE ATTENTION (layer.cpp:980): the
//        KV POOLS read DIRECTLY through the page table and the selection ids - no gather copy - with an online
//        softmax, one workgroup per query head.  Storage mode 0 = f16 pools (the shipped `--kv fp16` default),
//        mode 1 = int8 codes + fp16 scale per 64 values (`kv_q8.hpp`).  The q4_0 pool and the K8V4 hybrid are
//        REFUSED rather than dispatched against storage this shader does not implement - the port's loud-refusal
//        rule; a wrong pool read is a plausible token, not a fault.
//
//        The engine's `scratch` (the CUDA's chunk partials) is REQUIRED by the contract but NOT read here: this
//        re-derivation keeps the softmax state in registers, so the scratch layout is the CUDA's alone.  The
//        wrapper still refuses a null scratch (the engine's own contract refuses one) so a wiring bug is loud.
void qsa_decode_attn(Stream& s, const float* q, const strata::kernels::QsaAttnPools& pools, const int32_t* ids,
                     const int32_t* step, int64_t cap, const strata::kernels::QsaShapes& sh, float* scratch,
                     float* attn) {
    const int64_t nh = sh.n_head, kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (nh <= 0 || kh <= 0 || hd <= 0 || ps <= 0 || cap <= 0) return;
    if (nh % kh != 0) refuse("qsa_decode_attn_step", "n_head is not a multiple of n_head_kv");
    if (!pools.page_table || !ids || !step || !q || !attn || !scratch)
        refuse("qsa_decode_attn_step", "a required pointer is null");
    // The storage mode is chosen the way the CUDA chooses it (which pointers are non-null).
    const int mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                    : (pools.k_q != nullptr ? 1 : 0));
    if (mode == 2) refuse("qsa_decode_attn_step", "the Q4_0 pool storage is not implemented in this shader");
    if (mode == 3) refuse("qsa_decode_attn_step", "the K8V4 hybrid pool storage is not implemented in this shader");
    if (mode == 1 && hd % 64 != 0) refuse("qsa_decode_attn_step", "head_dim is not a multiple of 64 (int8 scale group)");
    Buf qv{}, tv{}, iv{}, sv{}, ov{}, kpv{}, vpv{}, kqv{}, vqv{}, ksv{}, vsv{};
    if (!arena_resolve(s, q, (uint64_t) nh * (uint64_t) hd * 4, qv) ||
        !arena_resolve(s, pools.page_table, 4, tv) ||    // the table's extent is unknown here; 4 B is the port's check
        !arena_resolve(s, ids, (uint64_t) cap * 4, iv) ||
        !arena_resolve(s, step, 20, sv) ||               // kStepCount * 4
        !arena_resolve(s, attn, (uint64_t) nh * (uint64_t) hd * 4, ov))
        refuse("qsa_decode_attn_step", "a pointer is not inside this stream's arena");
    if (mode == 0) {
        if (!pools.k_pool || !pools.v_pool) refuse("qsa_decode_attn_step", "the f16 pools are missing");
        if (!arena_resolve(s, pools.k_pool, (uint64_t) kh * (uint64_t) hd * 2, kpv) ||
            !arena_resolve(s, pools.v_pool, (uint64_t) kh * (uint64_t) hd * 2, vpv))
            refuse("qsa_decode_attn_step", "a pool pointer is not inside this stream's arena");
        kqv = vqv = ksv = vsv = tv;   // unused lanes bind the page table (never read in mode 0) - no null descriptor
    } else {
        if (!pools.k_q || !pools.v_q || !pools.k_scale || !pools.v_scale)
            refuse("qsa_decode_attn_step", "the int8 pools are incomplete");
        if (!arena_resolve(s, pools.k_q, (uint64_t) kh * (uint64_t) hd, kqv) ||
            !arena_resolve(s, pools.v_q, (uint64_t) kh * (uint64_t) hd, vqv) ||
            !arena_resolve(s, pools.k_scale, (uint64_t) kh * (uint64_t) (hd / 64) * 2, ksv) ||
            !arena_resolve(s, pools.v_scale, (uint64_t) kh * (uint64_t) (hd / 64) * 2, vsv))
            refuse("qsa_decode_attn_step", "an int8 pool pointer is not inside this stream's arena");
        kpv = vpv = tv;   // unused lanes bind the page table (never read in mode 1)
    }
    struct Push {
        int32_t n_head;
        int32_t kv_heads;
        int32_t head_dim;
        int32_t page_size;
        int32_t mode;
    } pc{};
    pc.n_head = (int32_t) nh;
    pc.kv_heads = (int32_t) kh;
    pc.head_dim = (int32_t) hd;
    pc.page_size = (int32_t) ps;
    pc.mode = mode;
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/qsa_decode_attn.spv", 11, sizeof(pc));
    s.ctx->dispatch(pipe, {&qv, &kpv, &vpv, &kqv, &vqv, &ksv, &vsv, &tv, &iv, &sv, &ov}, &pc, sizeof(pc),
                    (uint32_t) nh);
}

// `indexer_key_append` -> indexer_key_append.spv (RAW ro, POS ro, WK ro, TAIL rw, DEAD rw, POOLED rw,
// BPOS rw, COS ro, SIN ro; push {idx_dim, r, n_rot, pos_base, eps}).  ONE CELL per dispatch: the tail is a
// ring and the spare pooled row MOVES (`n_bid = pos/r`), so a caller drives one call per token exactly as
// `qsa_layer` does (`layer.cpp:948`).  The engine's own `pos_dev` is a DEVICE pointer, so the wrapper reads
// nothing back - the shader derives the slot from device memory, which is what keeps this capturable.
//
// THE POOLED/COS/SIN EXTENTS ARE THE CALLER'S, NOT THE WRAPPER'S.  The row the shader writes is `pos/r`, a
// DEVICE quantity the host cannot see, and the cos/sin table is `st.cos_tab` over `st.max_cells` - so the
// live-range check here names the smallest extent that is certainly inside the region (`idx_dim` for the
// pooled row, `n_rot/2` for one rotation row).  That is the same contract the CUDA has: the caller sized
// the state for `max_cells` and the arena is one buffer, so the check is a pointer-in-arena test, not a
// size assertion.  A wrap or a missing completion rotation would be an out-of-range write the caller sized
// for; the gate's `case_indexer_key_append_entry` pins the values and the completion rule instead.
static void indexer_key_append_impl(Stream& s, const float* raw, const int32_t* pos_dev, int32_t pos_base,
                                    const float* w_k_norm, float eps, const strata::kernels::QsaIndexerBuffers& b,
                                    const strata::kernels::QsaShapes& sh, const float* cos_tab, const float* sin_tab) {
    const int64_t idx_dim = sh.idx_dim, r = sh.idx_block, n_rot = sh.n_rot;
    if (idx_dim <= 0 || r <= 0 || n_rot <= 0) return;
    if (!raw || !pos_dev || !w_k_norm || !b.tail || !b.dead || !b.pooled || !b.block_pos || !cos_tab || !sin_tab)
        refuse("indexer_key_append", "a required pointer is null");
    Buf rawv{}, posv{}, wv{}, tailv{}, deadv{}, poolv{}, bpv{}, ctv{}, stv{};
    if (!arena_resolve(s, raw, (uint64_t) idx_dim * 4, rawv) ||
        !arena_resolve(s, pos_dev, 4, posv) ||
        !arena_resolve(s, w_k_norm, (uint64_t) idx_dim * 4, wv) ||
        !arena_resolve(s, b.tail, (uint64_t) (r - 1) * idx_dim * 4, tailv) ||
        !arena_resolve(s, b.dead, (uint64_t) idx_dim * 4, deadv) ||
        !arena_resolve(s, b.pooled, (uint64_t) idx_dim * 4, poolv) ||
        !arena_resolve(s, b.block_pos, 4, bpv) ||
        !arena_resolve(s, cos_tab, (uint64_t) (n_rot / 2) * 4, ctv) ||
        !arena_resolve(s, sin_tab, (uint64_t) (n_rot / 2) * 4, stv))
        refuse("indexer_key_append", "a pointer is not inside this stream's arena");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/indexer_key_append.spv", 9, 20);
    struct { int32_t idx_dim, r, n_rot, pos_base; float eps; } pc{(int32_t) idx_dim, (int32_t) r,
                                                                 (int32_t) n_rot, pos_base, eps};
    s.ctx->dispatch(p, {&rawv, &posv, &wv, &tailv, &deadv, &poolv, &bpv, &ctv, &stv}, &pc, sizeof(pc), 1u);
}

// The same pointer rule the prefill TU uses (`resolve_dev`): a native pack holds some of these tables in MAPPED
// PINNED HOST memory, and a Vulkan shader binds the region's device-visible buffer, not the host address.
bool resolve_dev(const Stream& s, const void* p, uint64_t bytes, Buf& out) {
    if (p == nullptr) return false;
    return arena_resolve(s, p, bytes, out) || mapped_resolve(p, bytes, out);
}

// ---- native_qsa_indexer_append -> pf_indexer_native.spv.  ONE cell per dispatch, ONE workgroup of 256 with
// the first idx_dim (128) lanes active.  This is the NATIVE sibling of indexer_key_append_impl, a DIFFERENT
// arithmetic (see pf_indexer_native.comp): F16-rounded raw keys, an F32 barrier-tree reduce, and the rotation
// angle computed ON DEVICE.  `pos_dev` is a DEVICE pointer (the CUDA reads `*pos_dev`), so the wrapper reads
// nothing back - the shader derives the slot and the block from device memory, which keeps it capturable.
void native_qsa_indexer_append_dev(Stream& s, const float* raw, const int32_t* pos_dev, int32_t pos_base,
                                   const float* gamma, float eps, const strata::kernels::QsaIndexerBuffers& b,
                                   const strata::kernels::QsaShapes& sh, int64_t max_cells,
                                   const strata::kernels::RopeScaling& scaling) {
    const int64_t D = sh.idx_dim, r = sh.idx_block, n_rot = sh.n_rot;
    if (D != 128 || r != 4 || n_rot != 64)
        refuse("native_qsa_indexer_append", "the native indexer's geometry is fixed (idx_dim 128, idx_block 4, n_rot 64)");
    if (max_cells < 1) refuse("native_qsa_indexer_append", "max_cells < 1");
    if (!raw || !pos_dev || !gamma || !b.tail || !b.dead || !b.pooled || !b.block_pos)
        refuse("native_qsa_indexer_append", "a required pointer is null");
    // STRATA_ROPE_TABLE=1 with a matching table: the engine's <true> append rotates by the table's exact float64
    // angles; this shader computes the angle analytically.  Refuse the one configuration the port cannot honour
    // rather than silently diverge - the same guard native_rope_apply applies.
    if (strata::kernels::rope_table_for(scaling).cos != nullptr)
        refuse("native_qsa_indexer_append", "STRATA_ROPE_TABLE=1 with a matching table: this backend computes the "
                                            "indexer angle analytically, not from the float64 table");
    Buf rawv{}, posv{}, gv{}, tailv{}, deadv{}, poolv{}, bpv{};
    const uint64_t pooled_bytes = (uint64_t) (max_cells / r + 1) * (uint64_t) D * 4;
    if (!resolve_dev(s, raw, (uint64_t) D * 4, rawv) || !resolve_dev(s, pos_dev, 4, posv) ||
        !resolve_dev(s, gamma, (uint64_t) D * 4, gv) ||
        !resolve_dev(s, b.tail, (uint64_t) (r - 1) * D * 4, tailv) ||
        !resolve_dev(s, b.dead, (uint64_t) D * 4, deadv) || !resolve_dev(s, b.pooled, pooled_bytes, poolv) ||
        !resolve_dev(s, b.block_pos, 4, bpv))
        refuse("native_qsa_indexer_append", "a pointer is neither in this stream's arena nor a live mapped region");
    const strata::kernels::RopeKernelArgs ka = scaling.kernel_args((int) n_rot);
    struct Push {
        int32_t idx_dim, r, n_rot, pos_base;
        float eps, theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale;
    } pc{};
    pc.idx_dim = (int32_t) D;
    pc.r = (int32_t) r;
    pc.n_rot = (int32_t) n_rot;
    pc.pos_base = pos_base;
    pc.eps = eps;
    pc.theta_scale = std::pow((float) scaling.freq_base, -2.0f / (float) n_rot);
    pc.freq_scale = ka.freq_scale;
    pc.corr_low = ka.corr_low;
    pc.corr_high = ka.corr_high;
    pc.ext_factor = ka.ext_factor;
    pc.mscale = ka.attn_factor;
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/pf_indexer_native.spv", 7, sizeof(pc));
    s.ctx->dispatch(p, {&rawv, &posv, &gv, &tailv, &deadv, &poolv, &bpv}, &pc, sizeof(pc), 1u);
}

// THE CHUNK'S POSITION ROW, placed once per stream (`arena_alloc` never decreases, so a per-call bump would
// grow the arena - the `iq_grids`/`vscratch` precedent).  `native_qsa_indexer_append_batch` writes every cell
// of the chunk into it with ONE `stream_write`, then binds cell t's own 4 bytes as POS for that cell's
// dispatch.  Grown on demand, never shrunk.
struct IdxPosScratch { int32_t* p = nullptr; uint64_t n = 0; };
std::unordered_map<Stream*, IdxPosScratch> g_idx_pos;
static int32_t* indexer_pos_scratch(Stream& s, uint64_t n) {
    IdxPosScratch& c = g_idx_pos[&s];
    if (c.n < n) { c.p = arena_alloc<int32_t>(s, n); c.n = n; }
    return c.p;
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/*.hpp declare --------------------------
namespace strata::kernels {

// helper: every body refuses when the opaque stream is not a live Vulkan stream.
static strata::vulkan::Stream* need_stream(const char* who, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return s;
}

// native_qsa.hpp: `void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output,
//     int n_cols, int n_rows, float epsilon, void* stream);`  (layer.cpp:879)
void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                                  float epsilon, void* stream) {
    strata::vulkan::native_qsa_rms_norm_weighted(*need_stream("native_qsa_rms_norm_weighted", stream), input, gamma,
                                                 output, n_cols, n_rows, epsilon);
}

// native_rope.hpp: `void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot,
//     const strata::kernels::RopeScaling& scaling, const int* positions, void* stream);`  (layer.cpp:881)
void native_rope_apply(const float* x, float* out, int rows, int head_dim, int n_rot, const strata::kernels::RopeScaling& scaling,
                       const int* positions, void* stream) {
    strata::vulkan::native_rope_apply(*need_stream("native_rope_apply", stream), x, out, rows, head_dim, n_rot,
                                      scaling, positions);
}

// rope.hpp: `void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot,
//     const float* cos_tab, const float* sin_tab, const int* pos, void* stream);`  (layer.cpp:882)
void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    strata::vulkan::rope_neox_apply(*need_stream("rope_neox_apply", stream), x, out, rows, head_dim, n_rot, cos_tab,
                                    sin_tab, pos);
}

// qsa_select.hpp: `void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx,
//     const int32_t* steps, int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
//     int64_t active_blocks);`  (layer.cpp:970)
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks) {
    strata::vulkan::qsa_block_scores(*need_stream("qsa_block_scores", stream), pooled, dead, q_idx, steps, nq,
                                     max_blocks, s, scores, active_blocks);
}

// qsa_select.hpp: `void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks,
//     int64_t cap, const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks);`  (layer.cpp:971)
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream, int64_t active_blocks) {
    strata::vulkan::qsa_block_topk(*need_stream("qsa_block_topk", stream), scores, steps, nq, max_blocks, cap, s,
                                   ids, active_blocks);
}

// native_qsa.hpp: `void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head,
//     int head_dim, void* stream);`  (layer.cpp:1010)
void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim,
                           void* stream) {
    strata::vulkan::native_qsa_gate_apply(*need_stream("native_qsa_gate_apply", stream), attn, q_full, output,
                                          n_head, head_dim);
}

// qsa.hpp: `void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out,
//     void* stream);`  (layer.cpp:1011)
void qsa_gate_apply_f32(const float* attn, const float* q_full, const QsaShapes& s, float* out, void* stream) {
    strata::vulkan::qsa_gate_apply_f32(*need_stream("qsa_gate_apply_f32", stream), attn, q_full, s, out);
}

// native_router.hpp: `void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);`
//     (moe_route, layer.cpp - the expert routing)
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    strata::vulkan::native_router_top10(*need_stream("native_router_top10", stream), logits, ids, weights);
}

// qsa_decode_attn.hpp: `void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids,
//     const int32_t* step, int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream);`
//     (layer.cpp:980 - THE DEFAULT decode attention, the fast-attn branch of the shipped configuration).
void qsa_decode_attn_step(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* step,
                          int64_t cap, const QsaShapes& s, float* scratch, float* attn, void* stream) {
    strata::vulkan::qsa_decode_attn(*need_stream("qsa_decode_attn_step", stream), q, pools, ids, step, cap, s,
                                    scratch, attn);
}

// qsa_decode_attn.hpp: `uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s);`  (a `host` row:
//     `layer.cpp:771/774` ask for the scratch SIZE, not for a dispatch).  A pure function of `cap` and the head
//     count, transcribed from `qsa_decode_attn.cu` (`CHUNK = 64`, `HD = 256`): chunks * n_head * (HD + 2) + 64.
uint64_t qsa_decode_attn_scratch_floats(int64_t cap, const QsaShapes& s) {
    const int64_t chunks = (cap + 63) / 64;
    return (uint64_t) chunks * (uint64_t) s.n_head * (uint64_t) (256 + 2) + 64;
}

// qsa.hpp: `void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s);`  (layer.cpp:908).  IT IS
// A PURE HOST ROW, NOT A DEVICE OP - `host_step` is host memory and there is no shader for it in this tree,
// and none is needed: it writes the four DERIVED per-token counts from one position, which is the point of
// the buffer (`n_kv`, `n_bid` and `width` must not be computed twice, differently).  Transcribed from
// `qsa.cu:699`.  The parent tier's no-shader list named this symbol; that list is WRONG for it and this is
// the correction - the one thing it shares with a no-shader row is that it used to be undefined here.
void qsa_step_fill(int32_t* host_step, int64_t pos, const QsaShapes& s) {
    if (host_step == nullptr) return;
    if (pos < 0) {
        std::fprintf(stderr, "qsa_step_fill: pos < 0 - refusing\n");
        std::exit(2);
    }
    const int64_t n_kv = pos + 1;
    host_step[kStepPos] = (int32_t) pos;
    host_step[kStepNKv] = (int32_t) n_kv;
    host_step[kStepNBid] = (int32_t) (n_kv / s.idx_block);
    host_step[kStepWidth] = (int32_t) qsa_selection_width(n_kv, s);
}

// qsa.hpp: `void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base,
//     const float* w_k_norm, float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
//     const float* sin_tab, void* stream);`  (layer.cpp:948).  THE LEGACY INDEXER MEMBER the
//     `native_qsa_indexer_enabled() == false` contract selects; the indexer runs on the MAIN forward path of
//     all 12 QSA layers every token (the append precedes the selection).  PORT-MAP was `kernel` already but
//     the wrapper had never been written - the class-A triage TABLE recorded it "LANDED" while THIS tree had
//     no definition for it at all, so the link still showed it undefined.  That is the batch's second
//     classification finding (the first: `qsa_step_fill` is host, not no-shader).
void indexer_key_append(const float* raw, const int32_t* pos_dev, int32_t pos_base, const float* w_k_norm,
                        float eps, const QsaIndexerBuffers& b, const QsaShapes& s, const float* cos_tab,
                        const float* sin_tab, void* stream) {
    strata::vulkan::indexer_key_append_impl(*need_stream("indexer_key_append", stream), raw, pos_dev, pos_base,
                                            w_k_norm, eps, b, s, cos_tab, sin_tab);
}

// native_qsa_indexer.hpp: `void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device,
//     int32_t pos_base, const float* gamma, float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s,
//     int64_t max_cells, const RopeScaling& scaling, void* stream);`  (layer.cpp:945, verify.cpp:819/1305/1859).
//     The NATIVE indexer member the `native_qsa_indexer_enabled()` contract selects; the flag is now TRUE
//     (native_caps_vk.cpp), so this is on the MAIN forward path of all 12 QSA layers every token.
void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s,
                               int64_t max_cells, const RopeScaling& scaling, void* stream) {
    strata::vulkan::native_qsa_indexer_append_dev(*need_stream("native_qsa_indexer_append", stream), raw,
                                                  relative_pos_device, pos_base, gamma, epsilon, b, s, max_cells,
                                                  scaling);
}

// native_qsa_indexer.hpp: the BATCHED append (prefill.cpp:2101).  The header's own contract: "leaving the
// buffers exactly as n calls of the single append in order would" - so this is a LOOP over the single append,
// one cell per dispatch, positions written to a device int per token.  The batch is documented HOST-side
// positions ("not for a captured graph"), so the host-staged position is honest here; the SINGLE append (the
// decode/verify path) reads the engine's real device `pos_dev` and is capturable.
//
// **THE POSITION STAGE IS NOT A PER-CELL STAGE (2026-10-06).**  The loop exists because the shader appends ONE
// cell per dispatch (`POS_.v[0]`, one workgroup).  What sat INSIDE it as well was the position UPLOAD: one
// `stream_write` of 4 bytes per cell.  `Ctx::write` flushes the live batch by contract (`vk_compute.cpp:1344`,
// the host-visible rule), so a 198-cell chunk issued **198 single-dispatch submits**.  The chunk's positions are
// `p0 .. p0+n-1` - all known before the loop - so the upload does not depend on the loop's reason at all.
// `prefill::rope` (prefill_vk.cpp:842) and `prefill::kv_append` (:909) already build their whole position array
// and write it ONCE outside their dispatch loops; this wrapper was the only one that did not.  FIX: one
// `stream_write` of the n-int row, then cell `t` binds its own 4 bytes (`pos + t`) as POS - the same value the
// per-cell write left in the same place, so the dispatch's INPUT IS BIT-IDENTICAL BY CONSTRUCTION.  The
// dispatches then batch (`kLiveBatchMax`) instead of each paying a submit+fence.  The per-cell form is kept and
// is taken when the device's `minStorageBufferOffsetAlignment` exceeds 4 (a 4-byte step is not bindable there -
// llvmpipe measures 16) and for `n == 1`; `STRATA_VK_INDEXER_POS_ONCE=0` forces it back for an A/B in one
// binary (read per call, the `STRATA_PF_GDN_REC_FUSED` precedent).  MEASURED in the engine: flush site
// `stream_write <- native_qsa_indexer_append_batch` is n=2,364 flushes of exactly ONE dispatch each - the
// largest flush COUNT of the whole prefill (2,364 of 3,021).
void native_qsa_indexer_append_batch(const float* raw, int64_t n, int64_t p0, int32_t pos_base, const float* gamma,
                                     float epsilon, const QsaIndexerBuffers& b, const QsaShapes& s, int64_t max_cells,
                                     const RopeScaling& scaling, void* stream) {
    if (n <= 0) return;
    strata::vulkan::Stream* st = need_stream("native_qsa_indexer_append_batch", stream);
    if (p0 < 0 || p0 + n > max_cells)
        strata::vulkan::refuse("native_qsa_indexer_append_batch", "the batch [p0, p0+n) does not fit max_cells");
    // Read PER CALL, not once per process, so one binary can A/B it and the gate can drive both paths.
    const char* once_env = std::getenv("STRATA_VK_INDEXER_POS_ONCE");
    const bool pos_once = once_env == nullptr || once_env[0] != '0';
    // A 4-byte descriptor step binds only where the device's own alignment limit allows it; `arena_alloc`
    // carves on a 256-byte boundary, so the condition reduces to `align | 4`.  Elsewhere: the per-cell form.
    const uint32_t align = (st->ctx != nullptr) ? st->ctx->info().min_storage_offset_align : 0;
    if (pos_once && n > 1 && align != 0 && align <= 4) {
        int32_t* pos = strata::vulkan::indexer_pos_scratch(*st, (uint64_t) n);
        std::vector<int32_t> hp((size_t) n);
        for (int64_t t = 0; t < n; ++t) hp[(size_t) t] = (int32_t) (p0 + t);
        strata::vulkan::stream_write(*st, pos, hp.data(), (uint64_t) n * 4);     // ONE upload, ONE flush
        for (int64_t t = 0; t < n; ++t)
            strata::vulkan::native_qsa_indexer_append_dev(*st, raw + t * s.idx_dim, pos + t, pos_base, gamma,
                                                          epsilon, b, s, max_cells, scaling);
        return;
    }
    int32_t* pos_dev = strata::vulkan::arena_alloc<int32_t>(*st, 1);
    for (int64_t t = 0; t < n; ++t) {
        const int32_t pos = (int32_t) (p0 + t);
        strata::vulkan::stream_write(*st, pos_dev, &pos, 4);
        strata::vulkan::native_qsa_indexer_append_dev(*st, raw + t * s.idx_dim, pos_dev, pos_base, gamma, epsilon,
                                                      b, s, max_cells, scaling);
    }
}

// qsa_decode_attn.hpp: the BATCHED decode attention (layer.cpp:996? - the `qsa_decode_attn_batch` fallback the
//     prompt path takes when `qsa_prompt_attn_batch` returns false, prefill.cpp:2226; the P6 verifier and the
//     MTP drafter use it too).  Plan v0.3 P5: "n_q queries at once, each with its own selection".  Served one
//     query at a time by the ALREADY-GATED decode kernel `qsa_decode_attn_step` (case_qsa_decode_attn) - the
//     batch is a LOOP, not a second attention kernel, which is exactly the header's own description
//     ("`qsa_decode_attn_batch` serves a prompt one query at a time with the decode kernel").
void qsa_decode_attn_batch(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                           int64_t cap, const QsaShapes& s, float* scratch, float* out, int64_t n_q, void* stream) {
    if (n_q <= 0) return;
    strata::vulkan::Stream* st = need_stream("qsa_decode_attn_batch", stream);
    const int64_t zv = s.n_head * s.head_dim;   // floats per query across heads
    for (int64_t i = 0; i < n_q; ++i)
        strata::vulkan::qsa_decode_attn(*st, q + i * zv, pools, ids + i * cap, steps + i * kStepCount, cap, s,
                                        scratch, out + i * zv);
}

// ---- the `host` row: the rope CONSTANTS (rope_scaling.hpp) -------------------------------------------------
// On a CUDA build `src/kernels/cuda/rope_scaling.cu` owns this storage; a Vulkan build compiles no such file.
// Plain storage, one writer at startup (the header's contract), read at every analytic rope launch.  The
// default-constructed value is the identity - exactly the engine's unscaled decode.
static RopeScaling g_rope_scaling;
void rope_scaling_set(const strata::kernels::RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

}  // namespace strata::kernels
