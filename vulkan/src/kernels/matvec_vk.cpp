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

#include "strata/kernels/bf16_gemv.hpp"    // bf16_gemv, bf16_gemv_split, bf16_gemv_fp32_mmvf (I4)
#include "strata/kernels/iq_kernels.hpp"   // quantize_q8_1_rows, iq_row_bytes
#include "strata/kernels/kv_q4.hpp"        // kv_append_q4_step, kv_gather_q4_step, QsaShapes
#include "strata/kernels/kv_q8.hpp"        // kv_append_q8_step, kv_gather_q8_step, KV_Q8_GROUP (I4)
#include "strata/kernels/kv_stream.hpp"    // kv_block_bytes (the KV host row), KvHostPools (I4)
#include "strata/kernels/native_mmvq.hpp"  // native_quantize_q8_1, native_mmvq, native_q5_k_f32, *_supported/bytes
#include "strata/kernels/qsa.hpp"          // kv_append_step, kv_gather_step (the fp16 KV cache)
#include "strata/kernels/quantize_act.hpp" // quantize_q8_0, quantize_q8_0_scaled, quantize_q8_K
#include "strata/kernels/s2_gemv_q8.hpp"   // s2_gemv_q8 (I4)
#include "strata/kernels/s_gemv.hpp"       // s_gemv_q8k_split, s_gemv_q8_0_split (the S-family split GEMV)

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
//        region the call itself names.  `host_layout` selects the row rule, NOT whether to write: the POOL write
//        is ALWAYS dispatched (host_layout 0; the shader itself skips a page the table maps negative), and when
//        a HOST COPY is present (KV streaming) the IDENTITY row is a SECOND dispatch binding the host buffers -
//        which is what the CUDA's `if (host.k_q4 != nullptr)` block does.  (This was a single dispatch binding
//        the pool with host_layout 1, which wrote only the host row and left the pool stale.)
void kv_append_q4_step(Stream& s, uint8_t* k_q4, uint8_t* v_q4, const int32_t* page_table, const int32_t* step,
                       const float* kcur, const float* vcur, const strata::kernels::QsaShapes& sh,
                       const strata::kernels::KvHostPools* host) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0) return;
    if (hd % 32 != 0) refuse("kv_append_q4_step", "head_dim is not a multiple of 32 (the Q4_0 group)");
    const uint64_t bytes_per_head = strata::kernels::kv_q4_bytes_per_head((int) hd);
    const uint64_t cell_bytes = bytes_per_head * (uint64_t) kh;
    Buf kv{}, vv{}, tv{}, sv{}, kcv{}, vcv{};
    if (!arena_resolve(s, k_q4, cell_bytes, kv) ||
        !arena_resolve(s, v_q4, cell_bytes, vv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, step, 20, sv) ||
        !arena_resolve(s, kcur, (uint64_t) kh * (uint64_t) hd * 4, kcv) ||
        !arena_resolve(s, vcur, (uint64_t) kh * (uint64_t) hd * 4, vcv))
        refuse("kv_append_q4_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; int32_t host_layout; } pc{
        (int32_t) kh, (int32_t) hd, (int32_t) ps, 0};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_q4_append.spv", 6, sizeof(pc));
    const uint64_t n_groups = 2 * (uint64_t) kh * (uint64_t) (hd / 32);
    s.ctx->dispatch(pipe, {&kv, &vv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), groups_for(n_groups));
    if (host != nullptr && host->k_q4 != nullptr) {   // KV streaming: the identity row, the SAME shader
        Buf hk{}, hv{};
        if (!arena_resolve(s, host->k_q4, cell_bytes, hk) || !arena_resolve(s, host->v_q4, cell_bytes, hv))
            refuse("kv_append_q4_step", "the host-copy pointer is not inside this stream's arena");
        pc.host_layout = 1;
        s.ctx->dispatch(pipe, {&hk, &hv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), groups_for(n_groups));
    }
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

// ============================================================================================================
// I4 - THE NEXT EIGHT MATVEC / GEMV / KV ENTRY POINTS (the BF16 GEMVs, the S2 GEMV, the fp16/int8 KV cache)
// ============================================================================================================
// The order is derived from the layer body's own call sites, in source order as `src/core/layer.cpp` names them
// (the plan's §3 list is not an order):
//
//   1 `bf16_gemv_fp32_mmvf`  layer.cpp:97   project_bf16's native branch (--native)
//   2 `bf16_gemv_split`      layer.cpp:98   project_bf16's split branch (gdn alpha/beta :291, router :367)
//   3 `bf16_gemv`            layer.cpp:99   project_bf16's plain branch (qsa indexer k/q :918/:962)
//   4 `s2_gemv_q8`           layer.cpp:166  gemv_quantized's S2 branch (code_bits == 2)
//   5 `kv_append_q8_step`    layer.cpp:934  qsa_layer - the INT8/K8V4 KV cache APPEND
//   6 `kv_append_step`       layer.cpp:943  qsa_layer - the FP16 KV cache APPEND
//   7 `kv_gather_q8_step`    layer.cpp:983  qsa_layer - the INT8/K8V4 KV cache GATHER
//   8 `kv_gather_step`       layer.cpp:989  qsa_layer - the FP16 KV cache GATHER
//
// Three siblings of these eight are NOT reached and are therefore deferred: `bf16_gemv_fp32_mmvf_cols`
// (layer.cpp:414, only inside `moe_route_window`, which is called from `verify.cpp:927` - the P6 verifier, not
// the decode path) and `s_gemv_q8_0_split` / `s_gemv_q8k_split`, which PORT-MAP.tsv carries as `todo` with the
// reason "no shader in this tree yet" - a wrapper cannot be proved against a case that does not exist.

// ---- 1/3 `bf16_gemv` and 2 `bf16_gemv_split` -> bf16_gemv.spv (X, W, Y; push {int n_in; int n_out}; ONE
//        WORKGROUP PER OUTPUT ROW).  The activation is BF16 bits read as 32-bit PAIRS; `threads_per_row` is
//        DROPPED (the port renders the CUDA split's warps as the workgroup barrier tree - subgroup ops are
//        banned here), so both entry points drive the same shader.
static void bf16_gemv_impl(Stream& s, const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                           const char* who) {
    if (n_in <= 0 || n_out <= 0) return;
    if ((n_in & 1) != 0) refuse(who, "n_in must be even (the activation is read as 32-bit pairs)");
    Buf xv{}, wv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n_in * 2, xv) ||
        !arena_resolve(s, w, (uint64_t) n_out * (uint64_t) n_in * 2, wv) ||
        !arena_resolve(s, y, (uint64_t) n_out * 4, yv))
        refuse(who, "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/bf16_gemv.spv", 3, 8);
    struct { int32_t n_in; int32_t n_out; } pc{(int32_t) n_in, (int32_t) n_out};
    s.ctx->dispatch(pipe, {&xv, &wv, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
}
void bf16_gemv(Stream& s, const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    bf16_gemv_impl(s, x, w, y, n_in, n_out, "bf16_gemv");
}
void bf16_gemv_split(Stream& s, const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row) {
    (void) threads_per_row;   // the port's split IS one workgroup per row; the CUDA knob is not connected here
    bf16_gemv_impl(s, x, w, y, n_in, n_out, "bf16_gemv_split");
}

// ---- 1 `bf16_gemv_fp32_mmvf` -> bf16_mmvf_f32.spv (X f32, W bf16 pairs, Y; push {int n_in; int n_out}; ONE
//        WORKGROUP PER OUTPUT ROW).  The native single-token MMVF: F32 activation, BF16 weight.
void bf16_gemv_fp32_mmvf(Stream& s, const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if ((n_in & 1) != 0) refuse("bf16_gemv_fp32_mmvf", "n_in must be even (the weight is read as 32-bit pairs)");
    Buf xv{}, wv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n_in * 4, xv) ||
        !arena_resolve(s, w, (uint64_t) n_out * (uint64_t) n_in * 2, wv) ||
        !arena_resolve(s, y, (uint64_t) n_out * 4, yv))
        refuse("bf16_gemv_fp32_mmvf", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/bf16_mmvf_f32.spv", 3, 8);
    struct { int32_t n_in; int32_t n_out; } pc{(int32_t) n_in, (int32_t) n_out};
    s.ctx->dispatch(pipe, {&xv, &wv, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
}

// ---- 4 `s2_gemv_q8` -> s2_gemv_q8.spv (ACT q8_0, CODES, SCALES, Y; push {int n_in; int n_out}; ONE WORKGROUP
//        PER ROW).  The S2 (Q2_0) weight against a Q8_0 activation - the same parallelism decision as
//        bf16_gemv_split, so `threads_per_row` is dropped too.
void s2_gemv_q8(Stream& s, const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row) {
    (void) threads_per_row;   // dropped, as bf16_gemv_split's is
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 64 != 0) refuse("s2_gemv_q8", "n_in is not a multiple of 64 (the S2 group)");
    Buf av{}, cv{}, sv{}, yv{};
    if (!arena_resolve(s, act, (uint64_t) (n_in / 32) * 34, av) ||
        !arena_resolve(s, codes, (uint64_t) n_out * (uint64_t) (n_in / 4), cv) ||
        !arena_resolve(s, scales, (uint64_t) n_out * (uint64_t) (n_in / 64) * 4, sv) ||
        !arena_resolve(s, y, (uint64_t) n_out * 4, yv))
        refuse("s2_gemv_q8", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/s2_gemv_q8.spv", 4, 8);
    struct { int32_t n_in; int32_t n_out; } pc{(int32_t) n_in, (int32_t) n_out};
    s.ctx->dispatch(pipe, {&av, &cv, &sv, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
}

// ---- `s_gemv_q8k_split` / `s_gemv_q8_0_split` -> s_gemv_q8_split.spv (ACT, CODES, SCALES, OFFSET, Y; push
//        {int n_in; int n_out; int code_bits; int byte_shift; int bias; int codebook; int group_shift;
//         int has_offset; int q8k}; ONE WORKGROUP PER OUTPUT ROW).  The TWO QUANTIZED-ACTIVATION members of the
//        S-family split GEMV, which `shared_expert`'s canonical path dispatches (shared_expert.cu:284-291): the
//        projection takes `s_gemv_q8k_split` when its `SForm::act_kind == 1` (a K-quant/IQ weight, its Q8_K
//        image) and `s_gemv_q8_0_split` otherwise (a legacy weight, its Q8_0 image).  In the CUDA the two are ONE
//        kernel templated on `<CODE_BITS, Q8K>`; the port carries the activation kind in the push constant
//        (`q8k`), which is the same one-kernel split with the branch made an argument.
//
// SHAPE, from the CUDA: one WARP per output row in `s_gemv_q8_split_kernel`; the port renders it one WORKGROUP
// per row with the shared reduction (subgroup ops are banned - see wg_reduce.glsl), so the lane's elements are
// strided by 256 rather than by 32.  The per-lane structure otherwise matches: QE = 16 consecutive elements
// under ONE scale (`i >> group_shift`), the activation block hoisted out of the sixteen loads, the sixteen
// accumulators combined by the source's own pairwise tree.  The row's answer is the workgroup sum.
void s_gemv_q8_split(Stream& s, bool q8k, const uint8_t* act, const uint8_t* codes, const float* scales,
                     const float* offset, float* y, int64_t n_in, int64_t n_out,
                     const strata::kernels::SForm& form, const char* who) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION BLOCK.  Q8_K is 292 B / 256 elems, Q8_0 is 34 B / 32 elems; the shader picks the loader
    // (and the block stride) from `q8k`.  A partial block would read past the activation, so the multiple is
    // required here rather than discovered by the driver.
    const int64_t blk_elems = q8k ? 256 : 32;
    const int64_t blk_bytes = q8k ? 292 : 34;
    if (n_in % blk_elems != 0)
        refuse(who, "n_in is not a multiple of the activation block (256 for Q8_K, 32 for Q8_0)");
    if (form.code_bits != 4 && form.code_bits != 8)
        refuse(who, "code_bits must be 4 or 8 (S2/Q8_0 is s2_gemv_q8; s2 has no Q8_K contract)");
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0)
        refuse(who, "group_elems must be a power of two (the group index is a shift)");
    // SIXTEEN, NOT FOUR: a lane-iteration takes QE = 16 consecutive elements under ONE scale, so a group smaller
    // than 16 would read the wrong scale for most of them.  The CUDA launcher makes the same check (line 622).
    if (form.group_elems % 16 != 0)
        refuse(who, "group_elems is not a multiple of 16 (the lane octet must lie in one group)");
    if (n_in % form.group_elems != 0)
        refuse(who, "n_in is not a multiple of group_elems");
    const int64_t per_byte = 8 / form.code_bits;              // 2 (S4) or 1 (S8): codes per byte
    const int64_t n_groups = n_in / form.group_elems;
    const int64_t codes_per_row = n_in / per_byte;
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const int32_t byte_shift = (per_byte == 4) ? 2 : ((per_byte == 2) ? 1 : 0);

    Buf av{}, cv{}, sv{}, ov{}, yv{};
    if (!arena_resolve(s, act, (uint64_t) (n_in / blk_elems) * blk_bytes, av) ||
        !arena_resolve(s, codes, (uint64_t) n_out * (uint64_t) codes_per_row, cv) ||
        !arena_resolve(s, scales, (uint64_t) n_out * (uint64_t) n_groups * 4, sv) ||
        !arena_resolve(s, y, (uint64_t) n_out * 4, yv))
        refuse(who, "a pointer is not inside this stream's arena");
    // OFFSET IS ALWAYS BOUND (Vulkan has no null descriptor) and never read when `has_offset` is 0, so a form
    // without an offset binds an already-resolved buffer instead of allocating a dummy - the `qsa_decode_attn`
    // "unused lanes bind the page table" shape.
    ov = sv;
    if (form.has_offset) {
        if (offset == nullptr) refuse(who, "form says has_offset but offset is null");
        if (!arena_resolve(s, offset, (uint64_t) n_out * (uint64_t) n_groups * 4, ov))
            refuse(who, "the offset pointer is not inside this stream's arena");
    }
    struct { int32_t n_in, n_out, code_bits, byte_shift, bias, codebook, group_shift, has_offset, q8k; } pc{
        (int32_t) n_in, (int32_t) n_out, form.code_bits, byte_shift, form.code_bias, (int32_t) form.codebook,
        group_shift, form.has_offset ? 1 : 0, q8k ? 1 : 0};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/s_gemv_q8_split.spv", 5, sizeof(pc));
    s.ctx->dispatch(pipe, {&av, &cv, &sv, &ov, &yv}, &pc, sizeof(pc), (uint32_t) n_out);
}

// ---- 5 `kv_append_q8_step` -> kv_q8_append.spv (KQ, VQ, KS, VS rw; TAB, STEP, KC, VC ro; push {int kv_heads;
//        int head_dim; int page_size; int host_layout}; grid = 2 * kv_heads * head_dim/64, one 64-value group
//        per thread).  THE POOL IS ALWAYS WRITTEN (host_layout 0; the shader itself skips a page the table maps
//        negative).  With a HOST COPY (KV streaming) the CUDA writes the IDENTITY row TOO - a SECOND dispatch
//        binding the HOST buffers with host_layout 1, the same shader and the other buffer.
void kv_append_q8_step(Stream& s, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* step, const float* kcur, const float* vcur,
                       const strata::kernels::QsaShapes& sh, const strata::kernels::KvHostPools* host) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0) return;
    if (hd % 64 != 0) refuse("kv_append_q8_step", "head_dim is not a multiple of 64 (the KV-Q8 group)");
    const uint64_t code_bytes = (uint64_t) kh * (uint64_t) hd;              // one cell, all heads
    const uint64_t scale_bytes = (uint64_t) kh * (uint64_t) (hd / 64) * 2;
    Buf kqv{}, vqv{}, ksv{}, vsv{}, tv{}, sv{}, kcv{}, vcv{};
    if (!arena_resolve(s, k_q, code_bytes, kqv) || !arena_resolve(s, v_q, code_bytes, vqv) ||
        !arena_resolve(s, k_scale, scale_bytes, ksv) || !arena_resolve(s, v_scale, scale_bytes, vsv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, step, 20, sv) ||
        !arena_resolve(s, kcur, (uint64_t) kh * (uint64_t) hd * 4, kcv) ||
        !arena_resolve(s, vcur, (uint64_t) kh * (uint64_t) hd * 4, vcv))
        refuse("kv_append_q8_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; int32_t host_layout; } pc{
        (int32_t) kh, (int32_t) hd, (int32_t) ps, 0};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_q8_append.spv", 8, sizeof(pc));
    const uint32_t ngroups = groups_for(2 * (uint64_t) kh * (uint64_t) (hd / 64));
    s.ctx->dispatch(pipe, {&kqv, &vqv, &ksv, &vsv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), ngroups);
    if (host != nullptr && host->k_q != nullptr) {   // KV streaming: the identity row, the SAME shader
        Buf hk{}, hv{}, hks{}, hvs{};
        if (!arena_resolve(s, host->k_q, code_bytes, hk) || !arena_resolve(s, host->v_q, code_bytes, hv) ||
            !arena_resolve(s, host->k_scale, scale_bytes, hks) || !arena_resolve(s, host->v_scale, scale_bytes, hvs))
            refuse("kv_append_q8_step", "the host-copy pointer is not inside this stream's arena");
        pc.host_layout = 1;
        s.ctx->dispatch(pipe, {&hk, &hv, &hks, &hvs, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), ngroups);
    }
}

// ---- 6 `kv_append_step` -> kv_f16_append.spv (KPOOL, VPOOL rw; TAB, STEP, KC, VC ro; push {int kv_heads;
//        int head_dim; int page_size; int host_layout}; grid = 2 * kv_heads * head_dim).  The FP16 sibling of
//        the q8 append: same row rule, fp16 in and out, one thread per element of K or V.
void kv_append_step(Stream& s, uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const strata::kernels::QsaShapes& sh,
                    const strata::kernels::KvHostPools* host) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0) return;
    const uint64_t cell_bytes = (uint64_t) kh * (uint64_t) hd * 2;
    Buf kv{}, vv{}, tv{}, sv{}, kcv{}, vcv{};
    if (!arena_resolve(s, k_pool, cell_bytes, kv) || !arena_resolve(s, v_pool, cell_bytes, vv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, step, 20, sv) ||
        !arena_resolve(s, kcur, (uint64_t) kh * (uint64_t) hd * 4, kcv) ||
        !arena_resolve(s, vcur, (uint64_t) kh * (uint64_t) hd * 4, vcv))
        refuse("kv_append_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; int32_t host_layout; } pc{
        (int32_t) kh, (int32_t) hd, (int32_t) ps, 0};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_f16_append.spv", 6, sizeof(pc));
    const uint32_t ngroups = groups_for(2 * (uint64_t) kh * (uint64_t) hd);
    s.ctx->dispatch(pipe, {&kv, &vv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), ngroups);
    if (host != nullptr && host->k_pool != nullptr) {   // KV streaming: the identity row, the SAME shader
        Buf hk{}, hv{};
        if (!arena_resolve(s, host->k_pool, cell_bytes, hk) || !arena_resolve(s, host->v_pool, cell_bytes, hv))
            refuse("kv_append_step", "the host-copy pointer is not inside this stream's arena");
        pc.host_layout = 1;
        s.ctx->dispatch(pipe, {&hk, &hv, &tv, &sv, &kcv, &vcv}, &pc, sizeof(pc), ngroups);
    }
}

// ---- 7 `kv_gather_q8_step` -> kv_q8_gather.spv (CODES, SCALES, TAB, IDS, STEP ro, SCRATCH rw; push {int
//        kv_heads; int head_dim; int page_size}).  The shader is ONE SIDE (K or V), so this is TWO dispatches -
//        the K pass and the V pass - into the shared FP16 scratch.  THE GRID IS THE CAPACITY (`max_ids`), not
//        the live count: the shader reads the real count from `step` and guards the surplus, which is what makes
//        a RECORDED launch replay-safe at a later token.
void kv_gather_q8_step(Stream& s, const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale,
                       const uint16_t* v_scale, const int32_t* page_table, const int32_t* ids, const int32_t* step,
                       int64_t max_ids, const strata::kernels::QsaShapes& sh, uint16_t* k_scratch,
                       uint16_t* v_scratch) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0 || max_ids <= 0) return;
    if (hd % 64 != 0) refuse("kv_gather_q8_step", "head_dim is not a multiple of 64 (the KV-Q8 group)");
    const uint64_t code_bytes = (uint64_t) kh * (uint64_t) hd;
    const uint64_t scale_bytes = (uint64_t) kh * (uint64_t) (hd / 64) * 2;
    const uint64_t scratch = (uint64_t) max_ids * (uint64_t) kh * (uint64_t) hd * 2;
    Buf kqv{}, vqv{}, ksv{}, vsv{}, tv{}, iv{}, sv{}, ksc{}, vsc{};
    if (!arena_resolve(s, k_q, code_bytes, kqv) || !arena_resolve(s, v_q, code_bytes, vqv) ||
        !arena_resolve(s, k_scale, scale_bytes, ksv) || !arena_resolve(s, v_scale, scale_bytes, vsv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, ids, (uint64_t) max_ids * 4, iv) ||
        !arena_resolve(s, step, 20, sv) || !arena_resolve(s, k_scratch, scratch, ksc) ||
        !arena_resolve(s, v_scratch, scratch, vsc))
        refuse("kv_gather_q8_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; } pc{(int32_t) kh, (int32_t) hd, (int32_t) ps};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_q8_gather.spv", 6, sizeof(pc));
    const uint32_t ngroups = groups_for((uint64_t) max_ids * (uint64_t) kh * (uint64_t) (hd / 4));
    s.ctx->dispatch(pipe, {&kqv, &ksv, &tv, &iv, &sv, &ksc}, &pc, sizeof(pc), ngroups);
    s.ctx->dispatch(pipe, {&vqv, &vsv, &tv, &iv, &sv, &vsc}, &pc, sizeof(pc), ngroups);
}

// ---- 8 `kv_gather_step` -> kv_f16_gather.spv (POOL, TAB, IDS, STEP ro, SCRATCH rw; push {int kv_heads; int
//        head_dim; int page_size}).  The FP16 sibling of the q8 gather, same two-dispatch K/V shape and the
//        same capacity grid; f16 in, f16 out, so nothing can round.
void kv_gather_step(Stream& s, const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table,
                    const int32_t* ids, const int32_t* step, int64_t max_ids, const strata::kernels::QsaShapes& sh,
                    uint16_t* k_scratch, uint16_t* v_scratch) {
    const int64_t kh = sh.n_head_kv, hd = sh.head_dim, ps = sh.page_size;
    if (kh <= 0 || hd <= 0 || ps <= 0 || max_ids <= 0) return;
    const uint64_t pool_bytes = (uint64_t) kh * (uint64_t) hd * 2;   // one cell, all heads
    const uint64_t scratch = (uint64_t) max_ids * (uint64_t) kh * (uint64_t) hd * 2;
    Buf kpv{}, vpv{}, tv{}, iv{}, sv{}, ksc{}, vsc{};
    if (!arena_resolve(s, k_pool, pool_bytes, kpv) || !arena_resolve(s, v_pool, pool_bytes, vpv) ||
        !arena_resolve(s, page_table, 4, tv) || !arena_resolve(s, ids, (uint64_t) max_ids * 4, iv) ||
        !arena_resolve(s, step, 20, sv) || !arena_resolve(s, k_scratch, scratch, ksc) ||
        !arena_resolve(s, v_scratch, scratch, vsc))
        refuse("kv_gather_step", "a pointer is not inside this stream's arena");
    struct { int32_t kv_heads; int32_t head_dim; int32_t page_size; } pc{(int32_t) kh, (int32_t) hd, (int32_t) ps};
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/kv_f16_gather.spv", 5, sizeof(pc));
    const uint32_t ngroups = groups_for((uint64_t) max_ids * (uint64_t) kh * (uint64_t) (hd / 4));
    s.ctx->dispatch(pipe, {&kpv, &tv, &iv, &sv, &ksc}, &pc, sizeof(pc), ngroups);
    s.ctx->dispatch(pipe, {&vpv, &tv, &iv, &sv, &vsc}, &pc, sizeof(pc), ngroups);
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

// ---- THE KV CACHE'S HOST ROW ------------------------------------------------------------------------------
// `kv_block_bytes(s, fmt)` - the BYTES OF ONE BLOCK (page) of K and V together, the size the KV-streaming path
// sizes its pinned host copy with (layer.cpp:659).  A pure function of the shapes and the storage format,
// transcribed from src/kernels/cuda/kv_stream.cu so the two agree byte for byte.  This IS a host row the layer
// reaches (it is not a bind); the other three KV-stream rows - `kv_stream_reset`, `kv_ring_table`,
// `kv_stream_resolve` - are the STREAMING RESIDENT TIER, reached only under `--kv-resident` (mode != 0), and
// they are NOT answered here: the engine calls the first two with a NULL stream (layer.cpp:707/709) and this
// backend has no default stream to fall back on, and the third needs the resolve/copy kernels whose shaders this
// tree does not build.
uint64_t kv_block_bytes(const QsaShapes& s, int fmt) {
    const uint64_t rows = (uint64_t) (s.n_head_kv * s.page_size);
    if (fmt == kKvQ4) return rows * kv_q4_bytes_per_head((int) s.head_dim) * 2;
    return fmt == kKvInt8 ? rows * (uint64_t) s.head_dim * 2 + rows * (uint64_t) (s.head_dim / KV_Q8_GROUP) * 2 * 2
                          : rows * (uint64_t) s.head_dim * 2 * 2;
}

// ---- the I4 entry points: the symbols include/strata/kernels/*.hpp declare --------------------------------
void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::bf16_gemv(strata::vulkan::stream_for("bf16_gemv", stream), x, w, y, n_in, n_out);
}
void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::bf16_gemv_split(strata::vulkan::stream_for("bf16_gemv_split", stream), x, w, y, n_in, n_out,
                                    threads_per_row);
}
void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::bf16_gemv_fp32_mmvf(strata::vulkan::stream_for("bf16_gemv_fp32_mmvf", stream), x, w, y, n_in,
                                        n_out);
}
void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in, int64_t n_out,
                int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::s2_gemv_q8(strata::vulkan::stream_for("s2_gemv_q8", stream), act, codes, scales, y, n_in, n_out,
                               threads_per_row);
}
void kv_append_q8_step(int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale, const int32_t* page_table,
                       const int32_t* step, const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                       const KvHostPools* host) {
    strata::vulkan::kv_append_q8_step(strata::vulkan::stream_for("kv_append_q8_step", stream), k_q, v_q, k_scale,
                                      v_scale, page_table, step, kcur, vcur, s, host);
}
void kv_append_step(uint16_t* k_pool, uint16_t* v_pool, const int32_t* page_table, const int32_t* step,
                    const float* kcur, const float* vcur, const QsaShapes& s, void* stream, const KvHostPools* host) {
    strata::vulkan::kv_append_step(strata::vulkan::stream_for("kv_append_step", stream), k_pool, v_pool, page_table,
                                   step, kcur, vcur, s, host);
}
void kv_gather_q8_step(const int8_t* k_q, const int8_t* v_q, const uint16_t* k_scale, const uint16_t* v_scale,
                       const int32_t* page_table, const int32_t* ids, const int32_t* step, int64_t max_ids,
                       const QsaShapes& s, uint16_t* k_scratch, uint16_t* v_scratch, void* stream) {
    strata::vulkan::kv_gather_q8_step(strata::vulkan::stream_for("kv_gather_q8_step", stream), k_q, v_q, k_scale,
                                      v_scale, page_table, ids, step, max_ids, s, k_scratch, v_scratch);
}
void kv_gather_step(const uint16_t* k_pool, const uint16_t* v_pool, const int32_t* page_table, const int32_t* ids,
                    const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                    uint16_t* v_scratch, void* stream) {
    strata::vulkan::kv_gather_step(strata::vulkan::stream_for("kv_gather_step", stream), k_pool, v_pool, page_table,
                                   ids, step, max_ids, s, k_scratch, v_scratch);
}

// ---- the S-FAMILY SPLIT GEMV's two quantized-activation members ------------------------------------------
// `s_gemv_q8k_split` (Q8_K image, act_kind 1) and `s_gemv_q8_0_split` (Q8_0 image) - the pair `shared_expert`'s
// canonical path reaches, and the pair that BLOCKED wiring it (ple_vk.cpp's earlier note).  Both drive the ONE
// shader `s_gemv_q8_split.spv`; the activation kind is the push constant's `q8k`.
void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::s_gemv_q8_split(strata::vulkan::stream_for("s_gemv_q8k_split", stream), /*q8k=*/true, x_q8k,
                                    codes, scales, offset, y, n_in, n_out, form, "s_gemv_q8k_split");
}
void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    strata::vulkan::s_gemv_q8_split(strata::vulkan::stream_for("s_gemv_q8_0_split", stream), /*q8k=*/false, x_q8_0,
                                    codes, scales, offset, y, n_in, n_out, form, "s_gemv_q8_0_split");
}

}  // namespace strata::kernels
