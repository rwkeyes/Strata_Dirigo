// vulkan/src/kernels/prefill_vk.cpp - the Vulkan backend's PREFILL entry points (the prompt path, src/prefill/).
//
// ============================================================================================================
// WHAT THIS BATCH PORTS, AND THE SHAPE OF THE REST
// ============================================================================================================
// The prompt path is a SUBSYSTEM (src/prefill/kernels.cu 53 KB + gemm.cu 25 + moe_fused*.cu 58 + moe_mmq.cu 12),
// and `--prefill 1` is the flag that makes a native pack able to start a decode at all (generate.cpp:7564 sets
// `pos_start` only INSIDE the prefill block; the verify loop's guard `spec_pos > 0` is what never comes true
// without it).  This TU ports the part of it that a chunk reaches FIRST, strongly preferring THIN DRIVERS over
// arithmetic the DECODE path already has, and reserving real new work for `strata::prefill::Gemm`:
//
//   * `Gemm` - the whole class (init_external / init / rebind / f16 / bf16 / native).  f16/bf16 ride the
//     ALREADY-GATED `gemm_prefill_fma.spv` (the port's prefill GEMM, verified against a double reference at
//     five shapes including the ragged edge) plus the engine's own bf16->f16 conversion route
//     (`bf16_to_f16.spv`, `bf16_to_f16_kernel`).  `native` rides the port's ALREADY-GATED decode arithmetic
//     (`native_quantize_q8_1` + `native_mmvq`), token by token - see the note on `Gemm::native` below.
//   * the hyper-connection (GR) family - `gr_broadcast`, `gr_norm`, `gr_norm_rs`, `gr_mix`, `gr_mix_r`,
//     `gr_silu`, `gr_write`, `gr_write_norm_rs` - five NEW shaders (`pf_gr_*.comp`) transcribed from
//     `src/prefill/kernels.cu`, because the decode's GR decomposition (`gr_norm`/`gr_down`/`gr_gate`/...) is a
//     DIFFERENT set of intermediates and no decode shader computes these five.  They are the first layer-body
//     ops after the embedding, so nothing else in the chunk loop is reachable without them.
//   * the THIN bind-and-dispatch set over existing shaders: `to_f16`, `to_bf16`, `copy_f32_wide`, `copy_i32`,
//     `gather_rows16`, `rms_rows`, `route`.
//   * `qsa_block_scores_tc` -> FALSE.  It is a CAPABILITY predicate (the engine falls back to the ported
//     `qsa_block_scores` when it answers false), so the honest Vulkan answer - this backend has no tensor-core
//     block-scores kernel - is a real implementation, not a stub.
//
// WHAT IS STILL A LOUD REFUSAL (refusals_prefill_vk.cpp): the GDN batched pair, the QSA prompt attention and
// indexer, the KV append family, the MoE combine/group, `blob_dequant_f16`, `swiglu_*`, and the IQ dequantiser
// (`iq_dequant_f16`/`iq_dequant_gu_f16`).  Those are the DELIVERABLE-B remaining list.
//
// ============================================================================================================
// THE PRE-FILL GEOMETRY IS THE ENGINE'S OWN, AND IT IS COMPILE-TIME THERE
// ============================================================================================================
// `src/prefill/kernels.cu:18-19` bakes `N=2560, HC=4, D=10240, LR=320, S=128, HK=16, HV=48, C=10240` into the
// prompt kernels.  Several prefill signatures carry no dimension at all (`gr_broadcast(e, R, T)`), so a port
// wrapper cannot derive them from its arguments - it has to name them the way the engine does, at the same
// layer of the stack.  These mirror the constants above, in ONE place, and every wrapper range-checks the
// pointers it is handed against them by asking `arena_resolve` for the byte count the geometry implies, so a
// mismatched buffer is a LOUD refusal rather than a wrong read.
#if !defined(STRATA_ENABLE_VULKAN)
#error "prefill_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/prefill/gemm.hpp"      // Gemm
#include "strata/prefill/kernels.hpp"   // the batched prompt kernels
#include "strata/kernels/native_mmvq.hpp"   // native_quantize_q8_1 / native_mmvq / native_q8_1_bytes
#include "strata/kernels/router_top10.hpp"  // router_top10
#include "strata/kernels/qsa_select.hpp"    // qsa_block_scores_tc
#include "strata/kernels/native_gdn.hpp"            // native_gdn_step (the DECODE recurrence, reused)
#include "strata/kernels/native_gdn_preprocess.hpp" // native_gdn_out_norm (the DECODE closing norm, reused)
#include "strata/kernels/native_rope.hpp"           // native_rope_apply (the DECODE rotation, reused)
#include "strata/vulkan/vk_backend.hpp"     // Stream, stream_of, rms_norm_weighted
#include "vk_arena.hpp"                     // the arena + pointer->buffer resolution

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::vulkan {
// Declared where it is defined (elementwise_vk.cpp) - used here for `beta != 0` (the opt-in bf16x2 remainder).
void add_inplace(Stream& s, float* dst, const float* src, int64_t n);
}  // namespace strata::vulkan

namespace {

using strata::vulkan::Buf;
using strata::vulkan::Stream;
using strata::vulkan::arena_alloc;
using strata::vulkan::arena_resolve;
using strata::vulkan::mapped_resolve;
using strata::vulkan::stream_write;

constexpr uint32_t kLocal = 256;
uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocal - 1) / kLocal); }

// The engine's own prefill geometry (src/prefill/kernels.cu:18-19).  Model-level constants, not tuning knobs.
constexpr int64_t kN = 2560;    // n_embd
constexpr int64_t kHC = 4;      // hyper-connection streams
constexpr int64_t kD = kN * kHC;
constexpr int64_t kLR = 320;    // hc_lr
constexpr int64_t kS = 128;     // GDN state width
constexpr int64_t kHK = 16;     // GDN k heads (the q heads too)
constexpr int64_t kHV = 48;     // GDN v heads
constexpr int64_t kC = 10240;   // GDN conv channels

[[noreturn]] void refuse(const char* sym, const char* why) {
    std::fprintf(stderr, "%s: %s\n  (the Vulkan prefill port refuses rather than degrade)\n", sym, why);
    std::exit(2);
}

Stream* need(Stream* s, const char* who) {
    if (s == nullptr) refuse(who, "the stream handle is not a live Vulkan stream");
    return s;
}

// Resolve a pointer to the live END of the arena (no size known, as the engine's own gather's need none).
bool arena_resolve_span(const Stream& s, const void* p, Buf& out) {
    if (p == nullptr) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (a < Stream::kArenaBase) return false;
    const uint64_t off = (uint64_t) (a - Stream::kArenaBase);
    if (off >= s.bump) return false;
    return arena_resolve(s, p, s.bump - off, out);
}

// ============================================================================================================
// POINTER -> BUFFER, TWO SOURCES, ONCE.
// ============================================================================================================
// The engine legitimately hands this backend pointers from MORE THAN ONE allocator: arena allocations
// (cudaMalloc / arena_alloc) AND mapped pinned host regions (`cudaHostAlloc` - the shim returns the mapping of a
// HOST_VISIBLE | HOST_COHERENT device block, and `cudaHostGetDevicePointer` returns that same host address, so a
// "device" pointer like the prefill's `grp_dev` IS a live mapped region).  A Vulkan shader binds a BUFFER, so
// every prefill entry that takes a raw pointer resolves through BOTH and refuses loudly when neither matches -
// the same handshake `iq_embed_rows` and `copy_from_mapped` use.  Four separate refusals (doorbell ->
// doorbell/x -> iq_embed_rows -> copy_i32) were each one more entry point missing this; it is answered HERE.
bool resolve_dev(const Stream& s, const void* p, uint64_t bytes, Buf& out) {
    return arena_resolve(s, p, bytes, out) || mapped_resolve(p, bytes, out);
}

// ---- per-stream scratch, from the ARENA (a `ctx->alloc` buffer is NOT resolvable to a device pointer) -------
struct Scratch { uint8_t* p = nullptr; uint64_t bytes = 0; };
Scratch& scratch_gr(std::unordered_map<Stream*, Scratch>& m, Stream& s, uint64_t bytes) {
    Scratch& c = m[&s];
    if (c.bytes < bytes) { c.p = (uint8_t*) arena_alloc(s, bytes); c.bytes = bytes; }
    return c;
}
std::unordered_map<Stream*, Scratch> g_x16, g_xf32, g_q8, g_temp;
Scratch& x16(Stream& s, uint64_t b) { return scratch_gr(g_x16, s, b); }
Scratch& xf32(Stream& s, uint64_t b) { return scratch_gr(g_xf32, s, b); }
Scratch& q8(Stream& s, uint64_t b) { return scratch_gr(g_q8, s, b); }
Scratch& tempf(Stream& s, uint64_t b) { return scratch_gr(g_temp, s, b); }

// f16 -> f32 widening (pf_f16_to_f32.spv), arena to arena.
void f16_to_f32(Stream& s, const uint16_t* x, float* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!resolve_dev(s, x, (uint64_t) n * 2, xv) || !resolve_dev(s, y, (uint64_t) n * 4, yv))
        refuse("prefill::Gemm::native", "the activation is neither in this arena nor a live mapped region");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/pf_f16_to_f32.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s.ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// bf16 -> f16 (bf16_to_f16.spv), arena to arena.
void bf16_to_f16(Stream& s, const uint16_t* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!resolve_dev(s, x, (uint64_t) n * 2, xv) || !resolve_dev(s, y, (uint64_t) n * 2, yv))
        refuse("prefill::Gemm::bf16", "an operand is neither in this arena nor a live mapped region");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/bf16_to_f16.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s.ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// THE PREFILL GEMM, the layout the engine's `Gemm::f16` calls: Y[T x ldy] = X[T x K] . W[N x K]^T.
// One invocation per output element (gemm_prefill_fma, no shape precondition - a 1-token chunk has t=1, which
// the cooperative-matrix kernel's `tiles_t = t / TM` would silently round to zero rows and compute NOTHING).
void gemm_f16(Stream& s, const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K,
              int64_t ldy, float beta) {
    Buf xv{}, wv{}, yv{};
    if (!resolve_dev(s, X, (uint64_t) T * K * 2, xv) || !resolve_dev(s, W, (uint64_t) N * K * 2, wv) ||
        !resolve_dev(s, Y, (uint64_t) T * ldy * 4, yv))
        refuse("prefill::Gemm::f16", "an operand is neither in this arena nor a live mapped region");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/gemm_prefill_fma.spv", 3, 16);
    struct { uint32_t t, n, k, ldy; } pc{(uint32_t) T, (uint32_t) N, (uint32_t) K, (uint32_t) ldy};
    if (beta == 0.0f) {
        s.ctx->dispatch(p, {&xv, &wv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) T * N));
        return;
    }
    // beta != 0 (the opt-in bf16x2 remainder): the FMA kernel does not accumulate, so compute a fresh tile and
    // add it - the same value the engine's `beta = 1` product adds.
    float* t = (float*) tempf(s, (uint64_t) T * ldy * 4).p;
    Buf tv{};
    if (!arena_resolve(s, t, (uint64_t) T * ldy * 4, tv)) refuse("prefill::Gemm", "no temp scratch");
    s.ctx->dispatch(p, {&xv, &wv, &tv}, &pc, sizeof(pc), groups_for((uint64_t) T * N));
    strata::vulkan::add_inplace(s, Y, t, T * ldy);
}

}  // namespace

namespace strata::prefill {

// ================================ Gemm ======================================================================
Gemm::~Gemm() {}   // every buffer is arena-owned; nothing to free

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    if (stream == nullptr) { err = "prefill gemm: a null stream"; return false; }
    stream_ = stream;
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    external_ = true;
    (void) ws_bytes;
    return true;
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) { err = "prefill gemm: the stream handle is not a live Vulkan stream"; return false; }
    stream_ = stream;
    if (scratch_elems > 0) scratch_ = (uint16_t*) arena_alloc(*s, (uint64_t) scratch_elems * 2);
    scratch_elems_ = scratch_elems;
    external_ = false;
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    (void) ws_bytes;
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    if (ldy <= 0) ldy = N;
    Stream* s = need(strata::vulkan::stream_of(stream_), "prefill::Gemm::f16");
    gemm_f16(*s, X, W, Y, T, N, K, ldy, beta);
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    if (ldy <= 0) ldy = N;
    Stream* s = need(strata::vulkan::stream_of(stream_), "prefill::Gemm::bf16");
    // THE ENGINE'S OWN ROUTE: convert both operands bf16 -> f16 (exact in f16's normal range, with the engine's
    // #540 clamp past +-65504) and run the f16 GEMM.  This is what `gemm.cu`'s `bf16_path()` selects on cards
    // with no BF16 tensor cores, and Intel has no bf16 cooperative-matrix config either.
    const uint64_t wbytes = (uint64_t) N * K * 2;
    if ((uint64_t) N * K > (uint64_t) std::max<int64_t>(scratch_elems_, 0)) {
        // W does not fit the caller's scratch at once: row slices, as the engine's `native` does.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) refuse("prefill::Gemm::bf16", "the render scratch is too small for one row of K");
        uint16_t* xf = (uint16_t*) x16(*s, (uint64_t) T * K * 2).p;
        bf16_to_f16(*s, X, xf, T * K);
        Buf yv{};
        if (!arena_resolve(*s, Y, (uint64_t) T * ldy * 4, yv)) refuse("prefill::Gemm::bf16", "Y not in arena");
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = std::min<int64_t>(rows, N - r0);
            bf16_to_f16(*s, W + r0 * K, scratch_, n * K);
            gemm_f16(*s, xf, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    uint16_t* wf = scratch_;
    uint16_t* xf = (uint16_t*) x16(*s, (uint64_t) T * K * 2).p;
    for (int64_t i = 0; i < T * K; i += (1 << 20)) {   // chunk the conversion so no single call is huge
        const int64_t n = std::min<int64_t>(1 << 20, T * K - i);
        bf16_to_f16(*s, X + i, xf + i, n);
    }
    if (wbytes > 0) bf16_to_f16(*s, W, wf, N * K);
    gemm_f16(*s, xf, wf, Y, T, N, K, ldy, beta);
}

// Y[T x ldy] = X[T x K] . W[N x K]^T, with W in NATIVE GGUF blocks.  The engine dequantizes W to f16 and runs
// the f16 GEMM; this port has no ggml dequantiser, so it answers the SAME product through the port's own
// ALREADY-GATED decode arithmetic - quantize the activation to q8_1 and dot the native blocks (native_mmvq),
// which is what the DECODE path computes for these very projections.  The prompt's conditioning is then
// computed by the same arithmetic as the decode steps that follow it, which is the property the milestone wants.
void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (T <= 0 || N <= 0 || K <= 0) return;
    if (ldy <= 0) ldy = N;
    if (K % 32 != 0) refuse("prefill::Gemm::native", "K is not a multiple of 32 (the q8_1 activation block)");
    Stream* s = need(strata::vulkan::stream_of(stream_), "prefill::Gemm::native");
    float* xf = (float*) xf32(*s, (uint64_t) K * 4).p;
    uint8_t* qb = q8(*s, (uint64_t) strata::kernels::native_q8_1_bytes((int) K, 1)).p;
    float* t = (beta != 0.0f) ? (float*) tempf(*s, (uint64_t) N * 4).p : nullptr;
    for (int64_t tk = 0; tk < T; ++tk) {
        f16_to_f32(*s, X + tk * K, xf, K);
        strata::kernels::native_quantize_q8_1(xf, qb, (int) K, 1, stream_);
        float* dst = (t != nullptr) ? t : Y + tk * ldy;
        strata::kernels::native_mmvq(ggml_type, W_blocks, qb, dst, (int) K, (int) N, 1, stream_);
        if (t != nullptr) strata::vulkan::add_inplace(*s, Y + tk * ldy, t, N);
    }
}

// ================================ the thin elementwise / copy set ===========================================
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::to_f16");
    Buf xv{}, yv{};
    if (!resolve_dev(*s, x, (uint64_t) n * 4, xv) || !resolve_dev(*s, y, (uint64_t) n * 2, yv))
        refuse("prefill::to_f16", "x or y is neither inside this stream's arena nor a live mapped region");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/f32_to_f16.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    if (ylo != nullptr) refuse("prefill::to_bf16", "the bf16x2 low image (STRATA_PREFILL_BF16X2) is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::to_bf16");
    Buf xv{}, yv{};
    if (!resolve_dev(*s, x, (uint64_t) n * 4, xv) || !resolve_dev(*s, y, (uint64_t) n * 2, yv))
        refuse("prefill::to_bf16", "x or y is neither inside this stream's arena nor a live mapped region");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/f32_to_bf16.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::copy_f32_wide");
    Buf sv{}, dv{};
    if (!resolve_dev(*s, src, (uint64_t) n * 4, sv) || !resolve_dev(*s, dst, (uint64_t) n * 4, dv))
        refuse("prefill::copy_f32_wide", "src or dst is neither inside this arena nor a live mapped region");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/copy.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&sv, &dv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// The int copy: a word-for-word device copy, EVERY side resolved through BOTH the arena and the live mapped
// regions (see resolve_dev).  A DISPATCH records under capture (re-reading the source BUFFER at submit - the
// mapped-region ordering contract) and submits+fences otherwise, so one mechanism covers both and a mapped
// destination like the prefill's `grp_dev` is handled instead of refused.
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::copy_i32");
    const uint64_t bytes = (uint64_t) n * 4;
    Buf dv{}, sv{};
    const bool dok = resolve_dev(*s, dst, bytes, dv);
    const bool sok = resolve_dev(*s, src, bytes, sv);
    if (!dok || !sok)
        refuse("prefill::copy_i32", !dok ? "dst is neither inside this stream's arena nor a live mapped host region"
                                        : "src is neither inside this stream's arena nor a live mapped host region");
    if ((uint64_t) n > 0xFFFFFFFFull) refuse("prefill::copy_i32", "n overflows the shader's uint");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_copy_u32.spv", 2, 4);
    struct { uint32_t n; } pc{(uint32_t) n};
    s->ctx->dispatch(p, {&sv, &dv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

void gather_rows16(const uint16_t* x16p, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width,
                   void* stream) {
    if (n <= 0 || width <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gather_rows16");
    const uint64_t total = (uint64_t) n * (uint64_t) width * 2;   // output bytes
    Buf sv{}, iv{}, dv{};
    if (!arena_resolve_span(*s, x16p, sv) || !resolve_dev(*s, src, (uint64_t) n * 4, iv) ||
        !resolve_dev(*s, dst16, total, dv))
        refuse("prefill::gather_rows16",
               "the table/ids/dst is neither inside this stream's arena nor a live mapped region");
    if (total > 0xFFFFFFFFull) refuse("prefill::gather_rows16", "n*row_bytes overflows the shader's uint");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/gather_rows.spv", 3, 8);
    struct { uint32_t row_bytes, n; } pc{(uint32_t) (width * 2), (uint32_t) n};
    s->ctx->dispatch(p, {&sv, &iv, &dv}, &pc, sizeof(pc), groups_for(total));
}

void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::rms_rows");
    if (ld != cols)
        refuse("prefill::rms_rows", "a row stride != cols: rms_norm.spv has no stride and the engine never asks");
    // the port's already-gated weighted RMS over rows (rms_norm.spv, one workgroup per row)
    strata::vulkan::rms_norm_weighted(*s, x, w, rows, cols, eps);
}

void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    if (T <= 0 || n_expert <= 0) return;
    // the native router: softmax over n_expert, top-10, weights renormalised over the ten (router_top10.spv,
    // the SAME kernel the decode path uses - the prompt's routing is the decode's routing).
    strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, (int*) ids, weights, stream);
}

// ================================ the hyper-connection (GR) family ==========================================
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_broadcast");
    Buf ev{}, rv{};
    if (!arena_resolve(*s, e, (uint64_t) T * kN * 4, ev) || !arena_resolve(*s, R, (uint64_t) T * kD * 4, rv))
        refuse("prefill::gr_broadcast", "e/R does not cover T*n_embd / T*hc*n_embd floats");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gr_bcast.spv", 2, 12);
    struct { int32_t T, N, HC; } pc{(int32_t) T, (int32_t) kN, (int32_t) kHC};
    s->ctx->dispatch(p, {&ev, &rv}, &pc, sizeof(pc), groups_for((uint64_t) T * kD));
}

// modes 0/1/2 of pf_gr_norm.spv (gr_norm / gr_norm_rs / gr_write_norm_rs)
static void gr_norm_mode(Stream& s, int mode, float* R, const float* w, float eps, float* out, uint16_t* xn16,
                         int64_t T, const float* bo, const float* inj, int64_t inj_ld) {
    Buf rv{}, wv{}, ov{}, xv{}, bv{}, iv{};
    if (!arena_resolve(s, R, (uint64_t) T * kD * 4, rv) ||
        !arena_resolve(s, w, (uint64_t) kHC * kN * 4, wv) ||
        !arena_resolve(s, xn16, (uint64_t) T * kD * 2, xv))
        refuse("prefill::gr_norm", "R/w_norm/xn16 does not cover T*hc*n_embd");
    const uint64_t out_bytes = (mode == 0) ? (uint64_t) T * kD * 4 : (uint64_t) T * kHC * 4;
    if (!arena_resolve(s, out, out_bytes, ov)) refuse("prefill::gr_norm", "the output is not large enough");
    if (!arena_resolve(s, (bo != nullptr) ? (const void*) bo : (const void*) R, (uint64_t) T * kN * 4, bv) ||
        !arena_resolve(s, (inj != nullptr) ? (const void*) inj : (const void*) R, (uint64_t) T * inj_ld * 4, iv))
        refuse("prefill::gr_norm", "bo/inject is not inside this stream's arena");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/pf_gr_norm.spv", 6, 20);
    struct { int32_t mode, N, HC, inj_ld; float eps; } pc{mode, (int32_t) kN, (int32_t) kHC, (int32_t) inj_ld, eps};
    s.ctx->dispatch(p, {&rv, &wv, &ov, &xv, &bv, &iv}, &pc, sizeof(pc), (uint32_t) (T * kHC));
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    if (T <= 0) return;
    if (xn16_lo != nullptr) refuse("prefill::gr_norm", "the bf16x2 low image (STRATA_PREFILL_BF16X2) is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_norm");
    gr_norm_mode(*s, 0, const_cast<float*>(R), w_norm, eps, xn, xn16, T, nullptr, nullptr, 0);
}

void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    if (T <= 0) return;
    if (xn16_lo != nullptr) refuse("prefill::gr_norm_rs", "the bf16x2 low image is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_norm_rs");
    gr_norm_mode(*s, 1, const_cast<float*>(R), w_norm, eps, rs, xn16, T, nullptr, nullptr, 0);
}

void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next,
                      float eps, float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    if (T <= 0) return;
    if (xn16_lo != nullptr) refuse("prefill::gr_write_norm_rs", "the bf16x2 low image is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_write_norm_rs");
    gr_norm_mode(*s, 2, R, w_norm_next, eps, rs, xn16, T, bo, inj, inj_ld);
}

void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    if (T <= 0) return;
    if (mixed16_lo != nullptr) refuse("prefill::gr_mix", "the bf16x2 low image is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_mix");
    Buf xv{}, gv{}, mv{}, m16{}, mh{};
    if (!arena_resolve(*s, xn, (uint64_t) T * kD * 4, xv) || !arena_resolve(*s, gated, (uint64_t) T * kD * 4, gv) ||
        !arena_resolve(*s, mixed, (uint64_t) T * kN * 4, mv) || !arena_resolve(*s, mixed16, (uint64_t) T * kN * 2, m16) ||
        !arena_resolve(*s, mixed_h, (uint64_t) T * kN * 2, mh))
        refuse("prefill::gr_mix", "an argument is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gr_mix.spv", 8, 16);
    struct { int32_t mode, T, N, HC; } pc{0, (int32_t) T, (int32_t) kN, (int32_t) kHC};
    s->ctx->dispatch(p, {&xv, &xv, &xv, &xv, &gv, &mv, &m16, &mh}, &pc, sizeof(pc), groups_for((uint64_t) T * kN));
}

void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed,
              uint16_t* mixed16, int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    if (T <= 0) return;
    if (mixed16_lo != nullptr) refuse("prefill::gr_mix_r", "the bf16x2 low image is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_mix_r");
    Buf rv{}, sv{}, wv{}, gv{}, mv{}, m16{}, mh{};
    if (!arena_resolve(*s, R, (uint64_t) T * kD * 4, rv) || !arena_resolve(*s, rs, (uint64_t) T * kHC * 4, sv) ||
        !arena_resolve(*s, w_norm, (uint64_t) kHC * kN * 4, wv) ||
        !arena_resolve(*s, gated, (uint64_t) T * kD * 4, gv) ||
        !arena_resolve(*s, mixed, (uint64_t) T * kN * 4, mv) || !arena_resolve(*s, mixed16, (uint64_t) T * kN * 2, m16) ||
        !arena_resolve(*s, mixed_h, (uint64_t) T * kN * 2, mh))
        refuse("prefill::gr_mix_r", "an argument is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gr_mix.spv", 8, 16);
    struct { int32_t mode, T, N, HC; } pc{1, (int32_t) T, (int32_t) kN, (int32_t) kHC};
    s->ctx->dispatch(p, {&rv, &sv, &wv, &rv, &gv, &mv, &m16, &mh}, &pc, sizeof(pc), groups_for((uint64_t) T * kN));
}

void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    if (T <= 0) return;
    if (lo16_lo != nullptr) refuse("prefill::gr_silu", "the bf16x2 low image is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_silu");
    const uint64_t n = (uint64_t) T * kLR;
    Buf lv{}, ov{};
    if (!arena_resolve(*s, lo, n * 4, lv) || !arena_resolve(*s, lo16, n * 2, ov))
        refuse("prefill::gr_silu", "lo/lo16 does not cover T*hc_lr");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gr_silu.spv", 2, 8);
    struct { int32_t n, hc; } pc{(int32_t) n, (int32_t) kHC};
    s->ctx->dispatch(p, {&lv, &ov}, &pc, sizeof(pc), groups_for(n));
}

void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gr_write");
    Buf rv{}, bv{}, iv{};
    if (!arena_resolve(*s, R, (uint64_t) T * kD * 4, rv) || !arena_resolve(*s, bo, (uint64_t) T * kN * 4, bv) ||
        !arena_resolve(*s, inj, (uint64_t) T * inj_ld * 4, iv))
        refuse("prefill::gr_write", "an argument is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gr_write.spv", 3, 16);
    struct { int32_t T, N, HC, inj_ld; } pc{(int32_t) T, (int32_t) kN, (int32_t) kHC, (int32_t) inj_ld};
    s->ctx->dispatch(p, {&rv, &bv, &iv}, &pc, sizeof(pc), groups_for((uint64_t) T * kD));
}

// ================================ the GDN (DeltaNet) batched pair ===========================================
void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T,
               void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gdn_gates");
    Buf a{}, d{}, sa{}, g{}, b{};
    if (!arena_resolve(*s, ab, (uint64_t) T * 2 * kHV * 4, a) || !arena_resolve(*s, dt, (uint64_t) kHV * 4, d) ||
        !arena_resolve(*s, ssm_a, (uint64_t) kHV * 4, sa) || !arena_resolve(*s, gate, (uint64_t) T * kHV * 4, g) ||
        !arena_resolve(*s, beta, (uint64_t) T * kHV * 4, b))
        refuse("prefill::gdn_gates", "an argument is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gdn_gates.spv", 5, 8);
    struct { int32_t HV, T; } pc{(int32_t) kHV, (int32_t) T};
    s->ctx->dispatch(p, {&a, &d, &sa, &g, &b}, &pc, sizeof(pc), groups_for((uint64_t) T * kHV));
}

void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gdn_conv");
    Buf hi{}, q{}, w{}, hb{};
    if (!arena_resolve(*s, history, (uint64_t) kC * 3 * 4, hi) || !arena_resolve(*s, qkv, (uint64_t) T * kC * 4, q) ||
        !arena_resolve(*s, conv_w, (uint64_t) kC * 4 * 4, w) || !arena_resolve(*s, h, (uint64_t) T * kC * 4, hb))
        refuse("prefill::gdn_conv", "an argument is not inside this stream's arena");
    {   // the 4-tap conv + SiLU, one invocation per channel walking the chunk
        VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gdn_conv.spv", 4, 8);
        struct { int32_t C, T; } pc{(int32_t) kC, (int32_t) T};
        s->ctx->dispatch(p, {&hi, &q, &w, &hb}, &pc, sizeof(pc), groups_for((uint64_t) kC));
    }
    {   // the q/k heads' L2, the engine's SECOND launch
        const int64_t NH = 2 * kHK;
        VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gdn_l2.spv", 1, 20);
        struct { int32_t C, T, S, NH; float eps; } pc{(int32_t) kC, (int32_t) T, (int32_t) kS, (int32_t) NH, eps};
        s->ctx->dispatch(p, {&hb}, &pc, sizeof(pc), (uint32_t) (T * NH));
    }
}

// ================================ the GDN recurrence (the DECODE step, per prompt token) ==================
// THE SHAPE.  `src/prefill/kernels.cu`'s `gdn_rec_kernel` walks the chunk INSIDE one launch, carrying the state in
// registers - but the recurrence is a sequential walk in time, so for a 1-2 token prompt the "batched" kernel
// computes exactly what the DECODE step computes, one token at a time.  This wrapper therefore reuses the port's
// ALREADY-GATED decode recurrence and closing norm per token, rather than transcribing the register-block kernel:
//
//   * `strata::kernels::native_gdn_step`  -> native_gdn_step.spv         (case_native_gdn_step)
//   * `strata::kernels::native_gdn_out_norm` -> native_gdn_out_norm.spv  (case_native_gdn_out_norm)
//   * `to_f16` -> f32_to_f16.spv (the FP16 image `y16` the out-projection reads; the CUDA writes hf(v) itself)
//
// The two are the SAME arithmetic, proved against the CUDA source rather than assumed.  `native_gdn.cu`'s `step`
// (line 46-81): q_head = head % h_k, state[(i*h_v + head)*S + col] - which IS `gdn_rec_kernel`'s
// `base = state + ((rg*RPG)*HV + head)*S + col`, rs = HV*S` with i = rg*RPG + r; dec = expf(gate[head]); the rank-1
// update is applied AFTER the decay (line 71: `s = g*s + k*delta`), and the readout is `sum s*q*scale`,
// scale = 1/sqrt(S) - exactly the prefill's `oc = (sum)*rsqrtf((float)S)`.  `gdn_out_norm` then applies
// `y = rms_norm(o, eps) * gamma * sigmoid(z)`, which is the prefill's
// `y = oc * rsqrtf(ss/S + eps) * gamma * sigmoid(z)`.  The state layout, the head pairing and the readout scale
// all agree, so this is a re-expression, not an approximation.
//
// GEOMETRY.  `h` is [T, C=10240] laid out [q 16*128 | k 16*128 | v 48*128]; q/k/v are passed as CONTIGUOUS slices
// of the token's row (the engine's own three headers live in that one row).  `state` is (S=128, HV=48, S=128), the
// same buffer/layout the decode path owns.  `z` is [T, HV*S]; `y` is the FP32 scratch and `y16` its FP16 image.
void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gdn_recurrence");
    const strata::kernels::GdnShapes sh{kS, kHK, kHV};
    float* o = (float*) xf32(*s, (uint64_t) kHV * kS * 4).p;
    for (int64_t t = 0; t < T; ++t) {
        const float* ht = h + t * kC;
        strata::kernels::native_gdn_step(state, ht, ht + kHK * kS, ht + 2 * kHK * kS, gate + t * kHV,
                                         beta + t * kHV, o, sh, stream);
        strata::kernels::native_gdn_out_norm(o, z + t * kHV * kS, gamma, y + t * kHV * kS, kHV, kS, eps, stream);
    }
    if (y16 != nullptr) to_f16(y, y16, T * kHV * kS, stream);
}

// ================================ the MoE combination and the expert SwiGLU ==============================
// `src/prefill/kernels.cu`'s `moe_combine_kernel` (line 704): one thread per element, NOT the decode's
// `native_moe_combine` (that one sums weight-indexed `parts` rows and adds the shared row plain; this one
// indirects through `slot` and scales the shared row by `sigmoid(sg[t])`).  K is baked at 10 in the CUDA loop.
void moe_combine(const float* D, const int32_t* slot, const float* w, const float* shared, const float* sg,
                 float* bo, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::moe_combine");
    constexpr int64_t kK = 10;
    Buf dv{}, sv{}, wv{}, shv{}, sgv{}, bov{};
    if (!arena_resolve(*s, D, (uint64_t) T * kK * kN * 4, dv) ||
        !arena_resolve(*s, slot, (uint64_t) T * kK * 4, sv) ||
        !arena_resolve(*s, w, (uint64_t) T * kK * 4, wv) ||
        !arena_resolve(*s, shared, (uint64_t) T * kN * 4, shv) ||
        !arena_resolve(*s, sg, (uint64_t) T * 4, sgv) ||
        !arena_resolve(*s, bo, (uint64_t) T * kN * 4, bov))
        refuse("prefill::moe_combine", "an argument is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_moe_combine.spv", 6, 12);
    struct { int32_t T, N, K; } pc{(int32_t) T, (int32_t) kN, (int32_t) kK};
    s->ctx->dispatch(p, {&dv, &sv, &wv, &shv, &sgv, &bov}, &pc, sizeof(pc), groups_for((uint64_t) T * kN));
}

// `swiglu_pair_kernel` / `swiglu_il_kernel`: h16 = hf_sat(silu(gate) * up), one 256-lane element-wise pass.
// The two shapes are ONE shader with a mode (see pf_swiglu16.comp); the element count is n*640 either way, and the
// interleaved form's stride is 2*640 (the engine's own 1280).
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::swiglu_pair");
    const uint64_t m = (uint64_t) n * 640;
    Buf gv{}, uv{}, ov{};
    if (!arena_resolve(*s, g, m * 4, gv) || !arena_resolve(*s, u, m * 4, uv) || !arena_resolve(*s, h16, m * 2, ov))
        refuse("prefill::swiglu_pair", "an operand is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_swiglu16.spv", 3, 12);
    struct { int32_t n, mode, ld; } pc{(int32_t) m, 0, 0};
    s->ctx->dispatch(p, {&gv, &uv, &ov}, &pc, sizeof(pc), groups_for(m));
}

void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::swiglu_interleaved");
    const uint64_t m = (uint64_t) n * 640;
    Buf gv{}, ov{};
    if (!arena_resolve(*s, gu, m * 2 * 4, gv) || !arena_resolve(*s, h16, m * 2, ov))
        refuse("prefill::swiglu_interleaved", "an operand is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_swiglu16.spv", 3, 12);
    struct { int32_t n, mode, ld; } pc{(int32_t) m, 1, 1280};
    s->ctx->dispatch(p, {&gv, &gv, &ov}, &pc, sizeof(pc), groups_for(m));
}

// ================================ RoPE (the DECODE rotation, per prompt row) ==============================
// `src/prefill/kernels.cu`'s `rope_kernel` rotates the first n_rot=64 channels of every (t, head) row in NEOX
// pairs (pair, pair+32), at angle `mrope_pos(..., pos0 + t, pair) * powf(theta_scale, pair)`.  The port's
// ALREADY-GATED DECODE `native_rope_apply` (native_rope_apply.spv, case_native_rope_apply) computes the SAME
// analytic angle ON DEVICE, per ROW, from an explicit `positions` array - so the prompt's T*heads rows are ONE
// call with positions[r] = pos0 + r/heads, in place (the kernel supports exact x == out).  The engine's call is
// contiguous (ld == heads*dim), which is the shape the decode kernel assumes; a strided call would be a
// silently wrong view, so it refuses.  (STRATA_ROPE_TABLE=1 is refused by the decode wrapper itself.)
void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    if (T <= 0 || heads <= 0 || dim <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::rope");
    if (ld != heads * dim)
        refuse("prefill::rope", "a row stride != heads*dim (the decode rope kernel's rows are contiguous)");
    if (dim < 64) refuse("prefill::rope", "dim < 64: the rotation needs the first 64 channels");
    // CAPTURE DISCIPLINE: the positions array is a HOST-STAGED input (a `stream_write`).  A host transfer
    // records NOTHING, so a replay would read the capture-time positions - every token after the first would
    // rotate at the wrong position.  The prefill is not captured today, but a wrapper that cannot record must
    // REFUSE rather than replay stale bytes.
    if (s->ctx != nullptr && s->ctx->capturing())
        refuse("prefill::rope", "the positions upload is host-staged and cannot be recorded under capture");
    const int64_t rows = T * heads;
    int32_t* pos = (int32_t*) xf32(*s, (uint64_t) rows * 4).p;
    std::vector<int32_t> hp((size_t) rows);
    for (int64_t r = 0; r < rows; ++r) hp[(size_t) r] = (int32_t) (pos0 + r / heads);
    stream_write(*s, pos, hp.data(), (uint64_t) rows * 4);
    strata::kernels::native_rope_apply(x, x, (int) rows, (int) dim, 64, scaling, pos, stream);
}

// ================================ the QSA q split and attention gate ======================================
// `split_q_kernel` / `gate_attn_kernel` (kernels.cu:750/756): element-wise gather and FP16 gate over the
// [T,24,512] q_full.  One thread per output element; the two are separate shaders (pf_split_q / pf_gate_attn).
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::split_q");
    const uint64_t n = (uint64_t) T * 24 * 256;
    Buf qv{}, ov{};
    if (!resolve_dev(*s, q_full, (uint64_t) T * 24 * 512 * 4, qv) || !resolve_dev(*s, q, n * 4, ov))
        refuse("prefill::split_q", "q_full or q is neither in this arena nor a live mapped region");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_split_q.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&qv, &ov}, &pc, sizeof(pc), groups_for(n));
}

void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    if (T <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gate_attn");
    const uint64_t n = (uint64_t) T * 24 * 256;
    Buf av{}, qv{}, ov{};
    if (!resolve_dev(*s, attn, n * 4, av) || !resolve_dev(*s, q_full, (uint64_t) T * 24 * 512 * 4, qv) ||
        !resolve_dev(*s, out16, n * 2, ov))
        refuse("prefill::gate_attn", "an argument is neither in this arena nor a live mapped region");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/pf_gate_attn.spv", 3, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&av, &qv, &ov}, &pc, sizeof(pc), groups_for(n));
}

}  // namespace strata::prefill

// ---- the prefill-path kernels-NAMESPACE symbols this batch answers ------------------------------------------
namespace strata::kernels {

// `qsa_block_scores_tc` (qsa_select.hpp): a CAPABILITY predicate - the engine falls back to the ported
// `qsa_block_scores` when it answers false.  This backend has no tensor-core block-scores kernel, so the
// honest answer is false; a stub that returned true would route the prompt at a kernel the port does not have.
bool qsa_block_scores_tc(const float*, const float*, const float*, const int32_t*, int64_t, int64_t,
                         const QsaShapes&, float*, void*, int64_t) {
    return false;
}

}  // namespace strata::kernels
