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
#include "strata/vulkan/vk_backend.hpp"     // Stream, stream_of, rms_norm_weighted
#include "vk_arena.hpp"                     // the arena + pointer->buffer resolution

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

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
    if (!arena_resolve(s, x, (uint64_t) n * 2, xv) || !arena_resolve(s, y, (uint64_t) n * 4, yv))
        refuse("prefill::Gemm::native", "the activation is not inside this stream's arena");
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/pf_f16_to_f32.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s.ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// bf16 -> f16 (bf16_to_f16.spv), arena to arena.
void bf16_to_f16(Stream& s, const uint16_t* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n * 2, xv) || !arena_resolve(s, y, (uint64_t) n * 2, yv))
        refuse("prefill::Gemm::bf16", "an operand is not inside this stream's arena");
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
    if (!arena_resolve(s, X, (uint64_t) T * K * 2, xv) || !arena_resolve(s, W, (uint64_t) N * K * 2, wv) ||
        !arena_resolve(s, Y, (uint64_t) T * ldy * 4, yv))
        refuse("prefill::Gemm::f16", "an operand is not inside this stream's arena");
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
    if (!arena_resolve(*s, x, (uint64_t) n * 4, xv) || !arena_resolve(*s, y, (uint64_t) n * 2, yv))
        refuse("prefill::to_f16", "a pointer is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/f32_to_f16.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    if (n <= 0) return;
    if (ylo != nullptr) refuse("prefill::to_bf16", "the bf16x2 low image (STRATA_PREFILL_BF16X2) is not ported");
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::to_bf16");
    Buf xv{}, yv{};
    if (!arena_resolve(*s, x, (uint64_t) n * 4, xv) || !arena_resolve(*s, y, (uint64_t) n * 2, yv))
        refuse("prefill::to_bf16", "a pointer is not inside this stream's arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/f32_to_bf16.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

void copy_f32_wide(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::copy_f32_wide");
    Buf sv{}, dv{};
    const bool ok_src = mapped_resolve(src, (uint64_t) n * 4, sv) || arena_resolve(*s, src, (uint64_t) n * 4, sv);
    if (!ok_src || !arena_resolve(*s, dst, (uint64_t) n * 4, dv))
        refuse("prefill::copy_f32_wide", "a pointer is neither a live mapped region nor inside this arena");
    VkPipeline p = s->ctx->pipeline(s->spv_dir + "/copy.spv", 2, 4);
    struct { int32_t n; } pc{(int32_t) n};
    s->ctx->dispatch(p, {&sv, &dv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// The int copy: a mapped/arena byte copy.  Under CAPTURE it must RECORD a device copy (a host-staged copy
// records nothing and every replay would see the capture-time bytes); outside a capture it keeps the fenced
// host-staged write.  Byte copies either way, so the int32 payload is bit-exact (copy.spv is float-typed).
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::copy_i32");
    Buf dv{};
    if (!arena_resolve(*s, dst, (uint64_t) n * 4, dv))
        refuse("prefill::copy_i32", "dst is not inside this stream's arena");
    if (s->ctx != nullptr && s->ctx->capturing()) {
        Buf sv{};
        if (!mapped_resolve(src, (uint64_t) n * 4, sv) && !arena_resolve(*s, src, (uint64_t) n * 4, sv))
            refuse("prefill::copy_i32", "the source is not a live mapped region nor in this arena (under capture)");
        s->ctx->capture_copy(dv, sv, (uint64_t) n * 4);
        return;
    }
    stream_write(*s, dst, src, (uint64_t) n * 4);
}

void gather_rows16(const uint16_t* x16p, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width,
                   void* stream) {
    if (n <= 0 || width <= 0) return;
    Stream* s = need(strata::vulkan::stream_of(stream), "prefill::gather_rows16");
    const uint64_t total = (uint64_t) n * (uint64_t) width * 2;   // output bytes
    Buf sv{}, iv{}, dv{};
    if (!arena_resolve_span(*s, x16p, sv) || !arena_resolve(*s, src, (uint64_t) n * 4, iv) ||
        !arena_resolve(*s, dst16, total, dv))
        refuse("prefill::gather_rows16", "src/ids/dst is not inside this stream's arena");
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
