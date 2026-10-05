// vulkan/src/kernels/matvec_vk.cpp - the Vulkan backend's MATVEC / GEMV / KV entry points.
//
// ============================================================================================================
// WHICH EIGHT, AND WHY THESE EIGHT (derived from the layer body's own call sites, not the plan's list)
// ============================================================================================================
//
// The group is `plan/BACKEND-INTEGRATION.md` §3's I3 (the bar reads its matvec/GEMV/KV rows as 21).  The
// plan's list is NOT an order, so this file fixes the order by what `src/core/layer.cpp` and its sibling
// files actually reach FIRST, in decode-path stage order:
//
//   #1 `quantize_q8_K`          layer.cpp:236     gdn_layer stage 1 - the K-quant activation image
//   #2 `quantize_q8_0`          layer.cpp:237     gdn_layer stage 1 - the Q8_0 activation image
//   #3 `native_quantize_q8_1`   layer.cpp:150     gdn_layer stage 2 - the native qkv projection's activation
//   #4 `native_mmvq`            layer.cpp:151     gdn_layer stage 2 - the native projection ITSELF (COMPOSITE:
//                                                 six `*_mmvq` shaders, chosen by the weight's ggml type)
//   #5 `kv_append_q4_step`      layer.cpp:935     qsa_layer - the Q4_0 KV cache APPEND
//   #6 `kv_gather_q4_step`      layer.cpp:985     qsa_layer - the Q4_0 KV cache GATHER
//   #7 `quantize_q8_0_scaled`   session.cpp:868   the MoE routed-expert activation (every layer)
//   #8 `native_q5_k_f32`        native_head.cpp:78 the head's native Q5_K matvec (the native sibling of #4)
//
// The other two of the increment's ten are NOT reached by the single-GPU decode path and are therefore
// DEFERRED, with the reason: `quantize_q8_1_rows` is the PEER-expert pool's activation (`peer_experts.cpp:230`,
// `remote_experts.cpp:307` - the peer tier), and `s_gemv_split_async` appears only in the standalone driver
// mains `overlap_main.cpp`/`concurrent_main.cpp`, never in `src/core/`.  (Both still have shaders and rows; they
// are simply later in reach order than these eight.)
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each symbol below is the thin wrapper already declared in
// `include/strata/kernels/{quantize_act,native_mmvq,iq_kernels,kv_q4}.hpp`; this TU answers the
// `strata::kernels::` symbol the wrapper calls.  A body resolves the engine's raw device pointers to arena
// views (vk_arena.hpp), takes the pipeline the device layer caches for the shader's own signature, and
// dispatches - and the shader each drives is the one the port's numeric gate has already gated:
//
//     quantize_q8_K        -> quantize_q8_K.spv        (case_quantize_q8_K)
//     quantize_q8_0        -> quantize_q8_0.spv        (case_quantize_q8_0)
//     quantize_q8_0_scaled -> quantize_q8_0_scaled.spv (case_quantize_q8_0)
//     native_quantize_q8_1 -> quantize_q8_1.spv        (case_quantize_q8_1)
//     native_mmvq          -> iq1m/iq2s/iq3s/iq3xxs/iq4nl/iq4xs_mmvq.spv (case_iq2s_mmvq + the other mmvq cases)
//     native_q5_k_f32      -> native_q5_k_f32.spv      (case_native_q5_k_f32), preceded by quantize_q8_1.spv
//     kv_append_q4_step    -> kv_q4_append.spv         (case_kv_q4_rot)
//     kv_gather_q4_step    -> kv_q4_gather.spv         (case_kv_q4_rot)
//
// The gate's `case_*_entry` cases re-run each with the ENGINE WRAPPER and compare BITWISE to that same shader
// path AND against the case's explicit oracle, so this file's claim is not "it compiles" but "the wrapper's
// answer equals the ported shader's answer".
//
// ============================================================================================================
// THE `host` ROWS THIS FILE ANSWERS, AND WHY (they are not a bare bind - the task anticipates these)
// ============================================================================================================
// The matvec/KV family carries REAL host state that a Vulkan build must answer itself, because a CUDA/HIP build
// gets it from the .cu files this tree does not compile:
//
//   * `iq_row_bytes(type, n)` - the per-format ROW STRIDE the six `*_mmvq` shaders take as `row_bytes`, and the
//     weight-buffer size the wrapper range-checks.  A quantisation CONSTANT, per ggml type.
//   * `native_mmvq_supported(type)` - the CAPABILITY CHECK that gates the composite.  This is the "*_supported
//     whose flag gates more than one symbol" this batch had to keep honest: `native_mmvq` dispatches by ggml
//     type, and the port ships shaders for exactly SIX types (IQ1_M/IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS), so the
//     check answers TRUE for those six and FALSE for every other type - answering TRUE for a type with no
//     shader would route the engine (`native_dense.cpp:53/166`, `native_head.cpp:34`) at an unported symbol.
//     The gate case `case_native_capabilities` is unaffected (it checks the six `*_enabled()` getters, not this
//     per-type query); the honesty is enforced HERE and by `case_native_mmvq_entry`, which drives two arms and
//     requires the composite to REFUSE an unsupported type.
//   * `native_mmvq_weight_bytes(type, n_in, n_out)` - a size: `iq_row_bytes(type, n_in) * n_out`.
//   * `native_q8_1_bytes(n_in, ncols)` - the q8_1 ACTIVATION scratch size (36 B per 32 values per column).
//     `native_q5_k_f32` sizes its internal quantise scratch with it.
#if !defined(STRATA_ENABLE_VULKAN)
#error "matvec_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/iq_kernels.hpp"   // quantize_q8_1_rows, iq_row_bytes
#include "strata/kernels/kv_q4.hpp"        // kv_append_q4_step, kv_gather_q4_step, QsaShapes
#include "strata/kernels/native_mmvq.hpp"  // native_quantize_q8_1, native_mmvq, native_q5_k_f32, *_supported/bytes
#include "strata/kernels/quantize_act.hpp" // quantize_q8_0, quantize_q8_0_scaled, quantize_q8_K

#include "iq_grids_vk.hpp"                 // the I-quant grid tables (a verbatim copy of the port's generated
                                           //   harness header - I1 adopted vk_compute.* the same way)
#include "strata/vulkan/vk_backend.hpp"    // the backend's seam: Stream, stream_of
#include "vk_arena.hpp"                    // the arena + pointer->buffer resolution

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::vulkan {

// Every shader in this file is `local_size_x = 256` (checked against the host's constant by the gate's census).
static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// The port's rule: refuse, never degrade.
[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

// The engine's opaque stream: a live Vulkan Stream, or a loud refusal (the port's rule).
static Stream& stream_for(const char* who, void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return *s;
}

// ---- #1 `quantize_q8_K` -> quantize_q8_K.spv (X ro, BLOCKS rw; push {int n_blocks}; one thread per block).
//        `n` is a multiple of 256 (292 B/256).  The shader writes the block's own bytes and nothing else.
void quantize_q8_K(Stream& s, const float* x, uint8_t* blocks, int64_t n) {
    if (n <= 0) return;
    if (n % 256 != 0) refuse("quantize_q8_K", "n is not a multiple of 256 (the Q8_K block is 256 values)");
    const int64_t nb = n / 256;
    Buf xv{}, bv{};
    if (!arena_resolve(s, x, (uint64_t) n * 4, xv) || !arena_resolve(s, blocks, (uint64_t) nb * 292, bv))
        refuse("quantize_q8_K", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/quantize_q8_K.spv", 2, 4);
    struct { int32_t n_blocks; } pc{(int32_t) nb};
    s.ctx->dispatch(pipe, {&xv, &bv}, &pc, sizeof(pc), groups_for((uint64_t) nb));
}

// ---- #2 `quantize_q8_0` -> quantize_q8_0.spv (X ro, BLOCKS rw; push {int n_blocks}; one thread per block).
//        `n` is a multiple of 32 (34 B/32).  ggml's bytes: the scale is an fp16 `d = amax/127`.
void quantize_q8_0(Stream& s, const float* x, uint8_t* blocks, int64_t n) {
    if (n <= 0) return;
    if (n % 32 != 0) refuse("quantize_q8_0", "n is not a multiple of 32 (the Q8_0 block is 32 values)");
    const int64_t nb = n / 32;
    Buf xv{}, bv{};
    if (!arena_resolve(s, x, (uint64_t) n * 4, xv) || !arena_resolve(s, blocks, (uint64_t) nb * 34, bv))
        refuse("quantize_q8_0", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/quantize_q8_0.spv", 2, 4);
    struct { int32_t n_blocks; } pc{(int32_t) nb};
    s.ctx->dispatch(pipe, {&xv, &bv}, &pc, sizeof(pc), groups_for((uint64_t) nb));
}

// ---- #7 `quantize_q8_0_scaled` -> quantize_q8_0_scaled.spv (X ro, BLOCKS rw, SCALES rw; push {int n_blocks}).
//        The same block bytes with the CPU's fp32 SCALE written beside them (`R4.2h`), so the hit path can use
//        the CPU's multiplier rather than the stored fp16 `d`.
void quantize_q8_0_scaled(Stream& s, const float* x, uint8_t* blocks, float* scales, int64_t n) {
    if (n <= 0) return;
    if (n % 32 != 0) refuse("quantize_q8_0_scaled", "n is not a multiple of 32 (the Q8_0 block is 32 values)");
    const int64_t nb = n / 32;
    Buf xv{}, bv{}, sv{};
    if (!arena_resolve(s, x, (uint64_t) n * 4, xv) || !arena_resolve(s, blocks, (uint64_t) nb * 34, bv) ||
        !arena_resolve(s, scales, (uint64_t) nb * 4, sv))
        refuse("quantize_q8_0_scaled", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/quantize_q8_0_scaled.spv", 3, 4);
    struct { int32_t n_blocks; } pc{(int32_t) nb};
    s.ctx->dispatch(pipe, {&xv, &bv, &sv}, &pc, sizeof(pc), groups_for((uint64_t) nb));
}

// ---- #3 `native_quantize_q8_1` -> quantize_q8_1.spv (X ro, Y rw; push {int n_in; int ncols}); the q8_1
//        activation block (36 B: fp16 d, fp16 sum, int8 qs[32]) the native MMVQ contract reads.  One
//        workgroup per 256 values, one column per `ncols`.
void native_quantize_q8_1(Stream& s, const float* x, void* x_q8_1, int64_t n_in, int64_t ncols) {
    if (n_in <= 0 || ncols <= 0) return;
    if (n_in % 32 != 0) refuse("native_quantize_q8_1", "n_in is not a multiple of 32 (the q8_1 block is 32 values)");
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n_in * (uint64_t) ncols * 4, xv) ||
        !arena_resolve(s, x_q8_1, strata::kernels::native_q8_1_bytes((int) n_in, (int) ncols), yv))
        refuse("native_quantize_q8_1", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/quantize_q8_1.spv", 2, 8);
    struct { int32_t n_in; int32_t ncols; } pc{(int32_t) n_in, (int32_t) ncols};
    const uint32_t per_col = (uint32_t) ((n_in + kLocalSize - 1) / kLocalSize);
    s.ctx->dispatch(pipe, {&xv, &yv}, &pc, sizeof(pc), per_col * (uint32_t) ncols);
}

// ---- `quantize_q8_1_rows` (the peer-expert pool's activation; the SAME shader as #3) -> quantize_q8_1.spv.
//        `x` is `n_rows` rows of `n_cols` floats; `y` is n_rows columns of n_cols/32 blocks of 36 B.  Wired as
//        a thin bind-and-dispatch so the peer tier's symbol is answered; not part of this batch's eight cases.
void quantize_q8_1_rows(Stream& s, const float* x, int64_t n_rows, int64_t n_cols, void* y) {
    if (n_rows <= 0 || n_cols <= 0) return;
    if (n_cols % 32 != 0) refuse("quantize_q8_1_rows", "n_cols is not a multiple of 32 (the q8_1 block is 32 values)");
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n_rows * (uint64_t) n_cols * 4, xv) ||
        !arena_resolve(s, y, strata::kernels::native_q8_1_bytes((int) n_cols, (int) n_rows), yv))
        refuse("quantize_q8_1_rows", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/quantize_q8_1.spv", 2, 8);
    struct { int32_t n_in; int32_t ncols; } pc{(int32_t) n_cols, (int32_t) n_rows};
    const uint32_t per_col = (uint32_t) ((n_cols + kLocalSize - 1) / kLocalSize);
    s.ctx->dispatch(pipe, {&xv, &yv}, &pc, sizeof(pc), per_col * (uint32_t) n_rows);
}

// ---- #4 `native_mmvq` -> SIX shaders, chosen by ggml type.  THE COMPOSITE.  Every arm is ONE WORKGROUP PER
//        OUTPUT ROW, `push {int n_in; int n_out; int row_bytes; int ncols}`, grid = n_out.  The IQ arms that
//        need a lookup grid take it as a fourth storage buffer; the port places the grids in the stream
//        (`Stream::iq_grids`), lazily, so a per-layer call does not grow the arena.
static const uint32_t* grid_for(Stream& s, int ggml_type, Buf& out) {
    // (type, spv name, grid pointer, grid size in uint32, grid slot) - the port's six IQ formats.
    switch (ggml_type) {
    case 29:  // IQ1_M
        if (s.iq_grids.iq1s.buffer == VK_NULL_HANDLE) { s.iq_grids.iq1s = s.ctx->alloc(sizeof(strata::vkport::kIq1sGrid));
            s.ctx->write(s.iq_grids.iq1s, strata::vkport::kIq1sGrid, sizeof(strata::vkport::kIq1sGrid)); }
        out = s.iq_grids.iq1s; return strata::vkport::kIq1sGrid;
    case 22:  // IQ2_S
        if (s.iq_grids.iq2s.buffer == VK_NULL_HANDLE) { s.iq_grids.iq2s = s.ctx->alloc(sizeof(strata::vkport::kIq2sGrid));
            s.ctx->write(s.iq_grids.iq2s, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid)); }
        out = s.iq_grids.iq2s; return strata::vkport::kIq2sGrid;
    case 21:  // IQ3_S
        if (s.iq_grids.iq3s.buffer == VK_NULL_HANDLE) { s.iq_grids.iq3s = s.ctx->alloc(sizeof(strata::vkport::kIq3sGrid));
            s.ctx->write(s.iq_grids.iq3s, strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid)); }
        out = s.iq_grids.iq3s; return strata::vkport::kIq3sGrid;
    case 18:  // IQ3_XXS
        if (s.iq_grids.iq3xxs.buffer == VK_NULL_HANDLE) { s.iq_grids.iq3xxs = s.ctx->alloc(sizeof(strata::vkport::kIq3xxsGrid));
            s.ctx->write(s.iq_grids.iq3xxs, strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid)); }
        out = s.iq_grids.iq3xxs; return strata::vkport::kIq3xxsGrid;
    default: return nullptr;   // IQ4_NL / IQ4_XS carry their table in the shader
    }
}

void native_mmvq(Stream& s, int ggml_type, const void* weights, const void* x_q8_1, float* y, int64_t n_in,
                 int64_t n_out, int64_t ncols) {
    if (n_in <= 0 || n_out <= 0 || ncols <= 0) return;
    if (n_in % 32 != 0) refuse("native_mmvq", "n_in is not a multiple of 32 (the q8_1 activation block)");
    const char* spv = nullptr;
    switch (ggml_type) {
    case 29: spv = "iq1m_mmvq.spv"; break;
    case 22: spv = "iq2s_mmvq.spv"; break;
    case 21: spv = "iq3s_mmvq.spv"; break;
    case 18: spv = "iq3xxs_mmvq.spv"; break;
    case 20: spv = "iq4nl_mmvq.spv"; break;
    case 23: spv = "iq4xs_mmvq.spv"; break;
    default: refuse("native_mmvq", "this backend has no shader for this ggml type (IQ1_M/IQ2_S/IQ3_S/IQ3_XXS/IQ4_NL/IQ4_XS only)");
    }
    const uint64_t row_bytes = strata::kernels::iq_row_bytes(ggml_type, n_in);
    const uint64_t wbytes = strata::kernels::native_mmvq_weight_bytes(ggml_type, (int) n_in, (int) n_out);
    const uint64_t abytes = strata::kernels::native_q8_1_bytes((int) n_in, (int) ncols);
    Buf wv{}, av{}, gv{}, yv{};
    if (!arena_resolve(s, weights, wbytes, wv) || !arena_resolve(s, x_q8_1, abytes, av) ||
        !arena_resolve(s, y, (uint64_t) n_out * (uint64_t) ncols * 4, yv))
        refuse("native_mmvq", "a pointer is not inside this stream's arena");
    struct { int32_t n_in; int32_t n_out; int32_t row_bytes; int32_t ncols; } pc{
        (int32_t) n_in, (int32_t) n_out, (int32_t) row_bytes, (int32_t) ncols};
    const uint32_t* grid = grid_for(s, ggml_type, gv);
    VkPipeline pipe;
    if (grid != nullptr) {
        pipe = s.ctx->pipeline(s.spv_dir + "/" + spv, 4, sizeof(pc));
        s.ctx->dispatch(pipe, {&wv, &av, &gv, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
    } else {
        pipe = s.ctx->pipeline(s.spv_dir + "/" + spv, 3, sizeof(pc));
        s.ctx->dispatch(pipe, {&wv, &av, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
    }
}

// ---- #8 `native_q5_k_f32` -> native_quantize_q8_1 (quantize_q8_1.spv) THEN native_q5_k_f32.spv.  The engine's
//        own composition, not a new shader: Q5_K's per-part scale AND min are PACKED in the 12-byte `scales`,
//        and the packed dot lives in `native_q5_k_f32.spv`.  `scratch_q8_1` is the caller's q8_1 buffer, sized
//        by native_q8_1_bytes(n_in, ncols).
void native_q5_k_f32(Stream& s, const void* weights, const float* x, void* scratch_q8_1, float* y, int64_t n_in,
                     int64_t n_out, int64_t ncols) {
    if (n_in <= 0 || n_out <= 0 || ncols <= 0) return;
    if (n_in % 256 != 0) refuse("native_q5_k_f32", "n_in is not a multiple of 256 (the Q5_K block is 256 values)");
    // The activation half: the q8_1 image the Q5_K dot reads.
    native_quantize_q8_1(s, x, scratch_q8_1, n_in, ncols);
    const uint64_t row_bytes = strata::kernels::iq_row_bytes(13 /*Q5_K*/, n_in);
    Buf wv{}, av{}, yv{};
    if (!arena_resolve(s, weights, row_bytes * (uint64_t) n_out, wv) ||
        !arena_resolve(s, scratch_q8_1, strata::kernels::native_q8_1_bytes((int) n_in, (int) ncols), av) ||
        !arena_resolve(s, y, (uint64_t) n_out * (uint64_t) ncols * 4, yv))
        refuse("native_q5_k_f32", "a pointer is not inside this stream's arena");
    struct { int32_t n_in; int32_t n_out; int32_t row_bytes; int32_t ncols; } pc{
        (int32_t) n_in, (int32_t) n_out, (int32_t) row_bytes, (int32_t) ncols};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_q5_k_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(pipe, {&wv, &av, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
}

// ---- #5 `kv_append_q4_step` -> kv_q4_append.spv (KQ4 rw, VQ4 rw, TAB ro, STEP ro, KC ro, VC ro; push
//        {int kv_heads; int head_dim; int page_size; int host_layout}; grid = 2 * kv_heads * head_dim/32).  K
//        and V share the grid.  `step` is a device int32 block whose [0] is the position.  The POOL's extent is
//        a session property the signature does not carry (the CUDA kernel addresses one row and needs no
//        total); the resolve below range-checks the pool BASE against one cell's row, which is the largest
//        region the call itself names.  `host_layout != 0` selects the KV-streaming identity row (the
//        host mirror); a negative page is the shader's "no write".
void kv_append_q4_step(Stream& s, uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const strata::kernels::QsaShapes& sh,
                       const strata::kernels::KvHostPools* host) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0) return;
    if (hd % 32 != 0) refuse("kv_append_q4_step", "head_dim is not a multiple of 32 (the Q4_0 group)");
    const uint64_t bytes_per_head = strata::kernels::kv_q4_bytes_per_head((int) hd);
    Buf kv{}, vv{}, tv{}, sv{}, kcv{}, vcv{};
    if (!arena_resolve(s, k_q4, bytes_per_head * (uint64_t) kh, kv) ||
        !arena_resolve(s, v_q4, bytes_per_head * (uint64_t) kh, vv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, step, 20, sv) ||
        !arena_resolve(s, kcur, (uint64_t) kh * (uint64_t) hd * 4, kcv) ||
        !arena_resolve(s, vcur, (uint64_t) kh * (uint64_t) hd * 4, vcv))
        refuse("kv_append_q4_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; int32_t host_layout; } pc{
        (int32_t) kh, (int32_t) hd, (int32_t) ps, host != nullptr ? 1 : 0};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_q4_append.spv", 6, sizeof(pc));
    const uint64_t n_groups = 2 * (uint64_t) kh * (uint64_t) (hd / 32);
    s.ctx->dispatch(pipe, {&kv, &vv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), groups_for(n_groups));
}

// ---- #6 `kv_gather_q4_step` -> kv_q4_gather.spv (KQ4 ro, VQ4 ro, TAB ro, IDS ro, STEP ro, KS rw, VS rw; push
//        {int kv_heads; int head_dim; int page_size}; grid = max_ids * kv_heads * head_dim/32, "the grid is the
//        CAPACITY, not the live count" - the shader exits on the surplus).  No inverse rotation: the pool is
//        H-rotated and stays rotated (kv_q4_gather.comp says why).
void kv_gather_q4_step(Stream& s, const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table,
                       const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const strata::kernels::QsaShapes& sh, uint16_t* k_scratch, uint16_t* v_scratch) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0 || max_ids <= 0) return;
    if (hd % 32 != 0) refuse("kv_gather_q4_step", "head_dim is not a multiple of 32 (the Q4_0 group)");
    const uint64_t bytes_per_head = strata::kernels::kv_q4_bytes_per_head((int) hd);
    const uint64_t scratch = (uint64_t) max_ids * (uint64_t) kh * (uint64_t) hd * 2;
    Buf kv{}, vv{}, tv{}, iv{}, sv{}, ksv{}, vsv{};
    if (!arena_resolve(s, k_q4, bytes_per_head * (uint64_t) kh, kv) ||
        !arena_resolve(s, v_q4, bytes_per_head * (uint64_t) kh, vv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, ids, (uint64_t) max_ids * 4, iv) ||
        !arena_resolve(s, step, 20, sv) || !arena_resolve(s, k_scratch, scratch, ksv) ||
        !arena_resolve(s, v_scratch, scratch, vsv))
        refuse("kv_gather_q4_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; } pc{(int32_t) kh, (int32_t) hd, (int32_t) ps};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_q4_gather.spv", 7, sizeof(pc));
    const uint64_t n_groups = (uint64_t) max_ids * (uint64_t) kh * (uint64_t) (hd / 32);
    s.ctx->dispatch(pipe, {&kv, &vv, &tv, &iv, &sv, &ksv, &vsv}, &pc, sizeof(pc), groups_for(n_groups));
}

}  // namespace strata::vulkan

// ============================================================================================================
// THE HOST ROWS (see the header note: quantisation constants, capability check, shape sizing)
// ============================================================================================================
namespace strata::kernels {

// `iq_row_bytes(type, n)` - bytes of ONE ROW of `n` values.  Per-format block sizes are ggml's (the same
// constants the six `*_mmvq` shaders carry); a type this backend has no shader for is a loud refusal, because
// a wrong row stride is a silently wrong answer.  Only the types this file can dispatch are spelled out; the
// table also covers the Q8_0/K-quant formats the engine's other TUs size with it.
size_t iq_row_bytes(int ggml_type, int64_t n) noexcept {
    const auto per256 = [&](int64_t block) { return (size_t) ((n / 256) * block); };
    const auto per32  = [&](int64_t block) { return (size_t) ((n / 32) * block); };
    switch (ggml_type) {
    case 29: return per256(56);    // IQ1_M
    case 16: return per256(66);    // IQ2_XXS
    case 17: return per256(74);    // IQ2_XS
    case 22: return per256(82);    // IQ2_S
    case 18: return per256(98);    // IQ3_XXS
    case 21: return per256(110);   // IQ3_S
    case 11: return per256(110);   // Q3_K
    case 12: return per256(144);   // Q4_K
    case 13: return per256(176);   // Q5_K
    case 14: return per256(210);   // Q6_K
    case 23: return per256(136);   // IQ4_XS
    case 20: return per32(18);     // IQ4_NL
    case 42: return (size_t) ((n / 64) * 18);   // Q2_0
    case 2:  return per32(18);     // Q4_0
    case 6:  return per32(22);     // Q5_0
    case 7:  return per32(22);     // Q5_1
    case 8:  return per32(34);     // Q8_0
    case 30: return (size_t) (n * 2);            // BF16
    default:
        std::fprintf(stderr, "strata::kernels::iq_row_bytes: no row layout for ggml type %d; refusing\n", ggml_type);
        std::exit(2);
    }
}

// `native_mmvq_supported(type)` - THE CAPABILITY CHECK THAT GATES THE COMPOSITE.  TRUE for exactly the six
// types whose `*_mmvq` shader this backend ships; FALSE for every other type, so the engine's native loader
// leaves a tensor of an unported type on the (ported) non-native branch rather than dispatching a missing
// kernel.  This is the "capability answer equals every gated symbol has a shader" rule, applied per type.
bool native_mmvq_supported(int ggml_type) noexcept {
    switch (ggml_type) {
    case 29: case 22: case 21: case 18: case 20: case 23: return true;
    default: return false;
    }
}

// `native_mmvq_weight_bytes(type, n_in, n_out)` - the weight-buffer size the wrapper range-checks.
std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    return iq_row_bytes(ggml_type, n_in) * (std::size_t) n_out;
}

// `native_q8_1_bytes(n_in, ncols)` - the q8_1 activation scratch: 36 B per 32 values, per column.
std::size_t native_q8_1_bytes(int n_in, int ncols) {
    if (n_in <= 0 || ncols <= 0) return 0;
    return (std::size_t) (n_in / 32) * 36u * (std::size_t) ncols;
}

// ---- the engine's entry points: the symbols include/strata/kernels/*.hpp declare --------------------------
void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    strata::vulkan::quantize_q8_K(strata::vulkan::stream_for("quantize_q8_K", stream), x, blocks, n);
}
void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    strata::vulkan::quantize_q8_0(strata::vulkan::stream_for("quantize_q8_0", stream), x, blocks, n);
}
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    strata::vulkan::quantize_q8_0_scaled(strata::vulkan::stream_for("quantize_q8_0_scaled", stream), x, blocks,
                                         scales, n);
}
void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    if (n_in <= 0 || ncols <= 0) return;
    strata::vulkan::native_quantize_q8_1(strata::vulkan::stream_for("native_quantize_q8_1", stream), x, x_q8_1,
                                         n_in, ncols);
}
void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    if (n_rows <= 0 || n_cols <= 0) return;
    strata::vulkan::quantize_q8_1_rows(strata::vulkan::stream_for("quantize_q8_1_rows", stream), x, n_rows, n_cols, y);
}
void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    if (n_in <= 0 || n_out <= 0 || ncols <= 0) return;
    strata::vulkan::native_mmvq(strata::vulkan::stream_for("native_mmvq", stream), ggml_type, weights, x_q8_1, y,
                                n_in, n_out, ncols);
}
void native_q5_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    if (n_in <= 0 || n_out <= 0 || ncols <= 0) return;
    strata::vulkan::native_q5_k_f32(strata::vulkan::stream_for("native_q5_k_f32", stream), weights, x,
                                    scratch_q8_1, y, n_in, n_out, ncols);
}
void kv_append_q4_step(uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    strata::vulkan::kv_append_q4_step(strata::vulkan::stream_for("kv_append_q4_step", stream), k_q4, v_q4,
                                      page_table, step, kcur, vcur, s, host);
}
void kv_gather_q4_step(const uint8_t* k_q4, const uint8_t* v_q4, const int32_t* page_table, const int32_t* ids,
                       const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                       uint16_t* v_scratch, void* stream) {
    strata::vulkan::kv_gather_q4_step(strata::vulkan::stream_for("kv_gather_q4_step", stream), k_q4, v_q4,
                                      page_table, ids, step, max_ids, s, k_scratch, v_scratch);
}

}  // namespace strata::kernels
