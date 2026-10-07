// ports/vulkan/bench/vk_bench.cpp - THE PORT-SIDE THROUGHPUT HARNESS.
//
// WHY IT EXISTS.  The port is correct-but-slow by construction: every fast path was dodged by a capability
// contract (native_gdn_enabled() == false, gr_set_native_mmvf(false), layer_set_fused_gr(false)) and the
// legacy/unfused kernels were ported as the reference.  Building the native/fused kernels is the next tier,
// and a performance claim can only be made against NUMBERS THIS HARNESS PRODUCES.  The numeric gate
// (gates/run_gate.sh + harness/vk_gate.cpp) remains the correctness authority; this is a separate, opt-in
// throughput measurement.  It reuses the port's own device layer (harness/vk_compute.*) rather than inventing
// one, and it lives in its own build tree (bench/build) so the gate's build is untouched.
//
// WHAT IS TIMED.  The kernel dispatch only: every input is uploaded once, outside the timed region, and the
// timed region is N dispatches of the kernel back to back inside ONE recorded command buffer, replayed.  No
// transfer is inside the loop, so the unit is the kernel's throughput on RESIDENT data.
//
// HOW THE SYNC IS DONE (wall clock around a fence - and why, not device timestamps).  The port's device layer
// exposes no VkQueryPool/timestamp path (harness/vk_compute.* has one-shot submit+fence and the recorded-step
// path only), so this harness times std::chrono::steady_clock around the recorded replay, which ends in
// vkWaitForFences.  That is a wall-clock measurement of a GPU-bounded region, not of GPU time alone: it
// includes the submit and one fence wait per BATCH (amortised over the batch's dispatches, and the median is
// taken, so a scheduling hiccup is not the reported figure).  The limit is honest and measured below: kernels
// whose whole batch is smaller than the dispatch overhead read as overhead-bound.  A device-timestamp
// version is the natural next step for the performance tier; it is not available from this device layer today.
//
// THE BATCH AND THE MEDIAN.  Each kernel is recorded `batch` times into one command buffer (a compute->compute
// barrier between dispatches, so they do not overlap), then replayed `warmups` times untimed and `reps` times
// timed; each timed replay is one submit + fence wait of the whole batch.  The reported figure is the MEDIAN
// per-dispatch time over `reps` timed replays, with the min and max beside it so the variance is visible.
//
// Usage:
//   vk_bench [--spv-dir D] [--device N] [--reps R] [--warmups W] [--list]
//   env: STRATA_VK_DESKTOP_RESERVE_MIB / STRATA_VK_RESERVE_FLOOR_MIB / STRATA_VK_MAX_BUDGET_MIB (the display
//   contract - configure_display_reserve() prints what it decided; run_bench.sh runs with the reserve at 0
//   for the same reason the gate does, and says so).

#include "../harness/vk_compute.hpp"
#include "../harness/iq_grids.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using portvk::Buf;
using portvk::Ctx;

namespace {

// ---------------------------------------------------------------------------------------------------------
// A small deterministic generator, so the fixture data is reproducible without linking the gate's RNG.
uint32_t g_state = 0x12345678u;
uint32_t next_rand() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return g_state;
}
float rndf() { return (float) ((int32_t) (next_rand() & 0xFFFFFFu) - 0x800000) / 8388608.0f; }

struct Timing {
    double med = 0, lo = 0, hi = 0;
    int reps = 0, batch = 0;
};

// Record `batch` dispatches into one command buffer, warm up, then time `reps` replays.  See the header for
// why this shape and not a per-dispatch submit.
Timing time_kernel(Ctx& ctx, VkPipeline p, const std::vector<const Buf*>& bufs, const void* push,
                   uint32_t push_bytes, uint32_t groups, uint32_t groups_y, int batch, int reps, int warmups) {
    ctx.record_begin();
    for (int i = 0; i < batch; ++i) ctx.record_dispatch(p, bufs, push, push_bytes, groups, groups_y);
    ctx.record_end_and_submit();          // the first submit; also the first warmup of the pipeline
    for (int w = 0; w < warmups; ++w) ctx.replay_recorded();
    std::vector<double> ms;
    ms.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx.replay_recorded();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) batch);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.reps = reps;
    t.batch = batch;
    t.med = ms[ms.size() / 2];
    t.lo = ms.front();
    t.hi = ms.back();
    return t;
}

// Time a TWO-KERNEL chain: per batch iteration, record kernel A then kernel B into the same command buffer
// (`record_dispatch` inserts the compute->compute barrier between them).  This exists for ONE pair: the GDN
// conv, where the legacy branch runs `gdn_conv_step` and then a SEPARATE `silu_f32` (src/core/layer.cpp:255-257)
// while the native kernel fuses them into one dispatch.  The reported figure is the median per-ITERATION time
// (both dispatches) / batch, so it is directly comparable to a single-dispatch `time_kernel` row.
Timing time_two(Ctx& ctx, VkPipeline pa, const std::vector<const Buf*>& ba, const void* pca, uint32_t pca_bytes,
                uint32_t ga, uint32_t gay, VkPipeline pb, const std::vector<const Buf*>& bb, const void* pcb,
                uint32_t pcb_bytes, uint32_t gb, uint32_t gby, int batch, int reps, int warmups) {
    ctx.record_begin();
    for (int i = 0; i < batch; ++i) {
        ctx.record_dispatch(pa, ba, pca, pca_bytes, ga, gay);
        ctx.record_dispatch(pb, bb, pcb, pcb_bytes, gb, gby);
    }
    ctx.record_end_and_submit();
    for (int w = 0; w < warmups; ++w) ctx.replay_recorded();
    std::vector<double> ms;
    ms.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx.replay_recorded();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) batch);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.reps = reps;
    t.batch = batch;
    t.med = ms[ms.size() / 2];
    t.lo = ms.front();
    t.hi = ms.back();
    return t;
}

// Time an N-KERNEL chain: per batch iteration, record each step in order into the same command buffer
// (`record_dispatch` inserts the compute->compute barrier between consecutive dispatches).  This is the
// generalized `time_two`, used for the fused paths' multi-dispatch comparisons (a 3- or 4-kernel legacy/native
// chain).  The reported figure is the median per-ITERATION time (all dispatches) / batch, directly comparable
// to a single-dispatch `time_kernel` row.
struct ChainStep {
    VkPipeline p;
    std::vector<const Buf*> bufs;
    const void* pc;
    uint32_t pc_bytes;
    uint32_t groups;
    uint32_t groups_y;
};
Timing time_chain(Ctx& ctx, const std::vector<ChainStep>& steps, int batch, int reps, int warmups) {
    ctx.record_begin();
    for (int i = 0; i < batch; ++i)
        for (const ChainStep& s : steps) ctx.record_dispatch(s.p, s.bufs, s.pc, s.pc_bytes, s.groups, s.groups_y);
    ctx.record_end_and_submit();
    for (int w = 0; w < warmups; ++w) ctx.replay_recorded();
    std::vector<double> ms;
    ms.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx.replay_recorded();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) batch);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.reps = reps;
    t.batch = batch;
    t.med = ms[ms.size() / 2];
    t.lo = ms.front();
    t.hi = ms.back();
    return t;
}

// The one row format.  `elems` is the number of logical output elements the dispatch produces; `macs` is the
// multiply-accumulate count it performs (0 where the kernel does no reduction, and then the unit is elems/s).
// `g_rows` counts them: it is the EVIDENCE an arm produced, and the arm ledger below turns "an arm printed
// nothing" into a named failure instead of an absence that reads as a pass.
uint64_t g_rows = 0;
void report(const char* kernel, const std::string& shape, const Timing& t, double elems, double macs) {
    ++g_rows;
    const double elem_s = elems * 1000.0 / t.med;
    const double mac_s = macs * 1000.0 / t.med;
    char macbuf[40];
    std::snprintf(macbuf, sizeof macbuf, "%s", macs > 0 ? "" : "        -");
    if (macs > 0) std::snprintf(macbuf, sizeof macbuf, "%8.3f", mac_s / 1e9);
    std::printf("ROW %-20s | %-30s | med %9.4f ms | min %9.4f | max %9.4f | reps %2d | batch %3d | %10.0f elems | %9.3f Melem/s | %s GMAC/s\n",
                kernel, shape.c_str(), t.med, t.lo, t.hi, t.reps, t.batch, elems, elem_s / 1e6, macbuf);
}

Buf alloc(Ctx& ctx, size_t bytes) { return ctx.alloc(bytes ? bytes : 4); }

std::vector<float> floats(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = rndf();
    return v;
}

// A finite, non-zero f16 fixture value: a random sign, exponent 14 or 15 (so |v| is in [0.5, 2)) and a random
// mantissa.  No NaN and no denormal, so a timing arm cannot be measuring a subnormal flush.
uint16_t rnd_half() {
    const uint32_t r = next_rand();
    const uint16_t sign = (uint16_t) ((r >> 20) & 0x8000u);
    const uint16_t exp = (uint16_t) (0x3800u | ((r >> 8) & 0x0400u));
    return (uint16_t) (sign | exp | (uint16_t) (r & 0x3FFu));
}

// =========================================================================================================
// THE PREFILL GEMM - `strata::prefill::Gemm::f16`, i.e. Y[T x ldy] = X[T x K] . W[N x K]^T with f16 operands
// and an f32 accumulate.  THE ONE DEEP KERNEL OF THE PROMPT PATH: it is what `gemm_prefill_*` compute, and the
// rows below are its IN-STREAM MARGINAL COST (one dispatch per replay of a batch, median of `reps`), the same
// instrument and the same documented limit as every other row in this file - the harness's buffers are the
// mapped host-visible type, so a row is a per-dispatch cost for comparing two kernels and NOT a bandwidth
// figure.
//
// THE SHAPES ARE THE ENGINE'S.  An expert's gate/up projection is N = n_ff = 1280, K = n_embd = 2560; its down
// projection is N = n_embd = 2560, K = n_ff/2 = 640; and the TOKEN COUNT is the row-batch routed to that
// expert, so 8/16/64/199 are the arms that matter rather than one prompt-sized shape.
//
// THE GRID IS THE CALLER'S CONTRACT, and each variant's is taken from the caller that really dispatches it:
// the untiled small-T kernel is one invocation per output element; the cooperative-matrix kernel is
// dispatched as an upper bound of one workgroup per tile (the gate's own `(t/8)*(n/16) + 8`); the tiled FMA
// kernel owns an (N/64, T/16) grid.  A variant given the wrong grid measures the grid, not the kernel.
Timing bench_gemm_one(Ctx& ctx, const std::string& dir, const char* spv, uint32_t T, uint32_t N, uint32_t K,
                      int reps, int warmups) {
    std::vector<uint16_t> x((size_t) T * K), w((size_t) N * K);
    for (auto& v : x) v = rnd_half();
    for (auto& v : w) v = rnd_half();
    Buf bx = alloc(ctx, x.size() * 2), bw = alloc(ctx, w.size() * 2), by = alloc(ctx, (size_t) T * N * 4);
    ctx.write(bx, x.data(), x.size() * 2);
    ctx.write(bw, w.data(), w.size() * 2);
    VkPipeline p = ctx.pipeline(dir + "/" + spv, 3, 16);
    struct { uint32_t t, n, k, ldy; } pc{T, N, K, N};
    const bool untiled = std::strstr(spv, "_small") != nullptr;
    const bool cma = std::strstr(spv, "f16_m8") != nullptr;
    uint32_t gx, gy = 1;
    if (untiled) gx = (uint32_t) (((uint64_t) T * N + 255) / 256);
    else if (cma) gx = (T / 8u) * (N / 16u) + 8u;
    else { gx = (N + 63u) / 64u; gy = (T + 15u) / 16u; }
    Timing t = time_kernel(ctx, p, {&bx, &bw, &by}, &pc, sizeof(pc), gx, gy, 8, reps, warmups);
    ctx.free(bx); ctx.free(bw); ctx.free(by);
    return t;
}

// The A/B table: the same shape on one device, both cooperative-matrix schedules (the shipped one, which
// `coopMatLoad`s straight from GLOBAL memory, and the shared-memory STAGED one), and both FMA paths for
// reference.  Printed as rows plus an XPAIR line so the ratio is read off one instrument.
void bench_gemm_prefill(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    if (!ctx.info().storage_buffer_16bit) {
        std::printf("SKIP gemm_prefill              | device lacks storageBuffer16BitAccess (the operands are f16)\n");
        return;
    }
    const bool cma = ctx.info().cooperative_matrix && ctx.info().cm_m == 8 && ctx.info().cm_n == 16 &&
                     ctx.info().cm_k == 16;
    struct Shape { uint32_t t, n, k; const char* proj; };
    const Shape shapes[] = {
        {8, 1280, 2560, "gate/up"},   {16, 1280, 2560, "gate/up"},  {64, 1280, 2560, "gate/up"},
        {199, 1280, 2560, "gate/up"}, {8, 2560, 640, "down"},       {64, 2560, 640, "down"},
        {199, 2560, 640, "down"},
    };
    for (const Shape& s : shapes) {
        char shape[80];
        Timing tg{}, ts{}, tu{}, tt{};
        if (cma) {
            tg = bench_gemm_one(ctx, dir, "gemm_prefill_f16_m8.spv", s.t, s.n, s.k, reps, warmups);
            std::snprintf(shape, sizeof shape, "%s T=%u N=%u K=%u", s.proj, s.t, s.n, s.k);
            report("gemm_prefill", std::string("cm-global  ") + shape, tg, (double) s.t * s.n,
                   (double) s.t * s.n * s.k);
            ts = bench_gemm_one(ctx, dir, "gemm_prefill_f16_m8_staged.spv", s.t, s.n, s.k, reps, warmups);
            report("gemm_prefill", std::string("cm-staged  ") + shape, ts, (double) s.t * s.n,
                   (double) s.t * s.n * s.k);
        } else {
            std::snprintf(shape, sizeof shape, "%s T=%u N=%u K=%u", s.proj, s.t, s.n, s.k);
        }
        tu = bench_gemm_one(ctx, dir, "gemm_prefill_fma_small.spv", s.t, s.n, s.k, reps, warmups);
        report("gemm_prefill", std::string("fma-untiled") + shape, tu, (double) s.t * s.n, (double) s.t * s.n * s.k);
        tt = bench_gemm_one(ctx, dir, "gemm_prefill_fma.spv", s.t, s.n, s.k, reps, warmups);
        report("gemm_prefill", std::string("fma-tiled  ") + shape, tt, (double) s.t * s.n, (double) s.t * s.n * s.k);
        if (cma)
            std::printf("XPAIR gemm_prefill %s | cm-staged/cm-global %.3f | cm-global/fma-untiled %.3f | "
                        "cm-staged/fma-untiled %.3f | fma-tiled/fma-untiled %.3f\n",
                        shape, ts.med / tg.med, tg.med / tu.med, ts.med / tu.med, tt.med / tu.med);
    }
}

// =========================================================================================================
// THE GDN / DELTA-NET MIXER CHAIN (36 of the model's 48 layers, under native_gdn_enabled() == false).
// Shapes are the artifact's where the kernel has one (S=128, h_k=16, h_v=48, d_conv=4, n_embd=2560).
// =========================================================================================================

void bench_gdn_conv_step(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int C = 2560, dc = 4;                         // n_embd, d_conv
    const size_t hist = (size_t) C * (dc - 1);
    std::vector<float> cs = floats(hist), x = floats((size_t) C), kw = floats((size_t) C * dc);
    Buf bcs = alloc(ctx, hist * 4), bx = alloc(ctx, (size_t) C * 4), bkw = alloc(ctx, (size_t) C * dc * 4),
        bo = alloc(ctx, (size_t) C * 4);
    ctx.write(bcs, cs.data(), hist * 4);
    ctx.write(bx, x.data(), (size_t) C * 4);
    ctx.write(bkw, kw.data(), (size_t) C * dc * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_conv_step.spv", 4, 8);
    struct { int32_t channels; int32_t d_conv; } pc{C, dc};
    Timing t = time_kernel(ctx, p, {&bcs, &bx, &bkw, &bo}, &pc, sizeof(pc), (uint32_t) ((C + 255) / 256), 1, 64,
                           reps, warmups);
    char shape[64];
    std::snprintf(shape, sizeof shape, "C=%d d_conv=%d", C, dc);
    report("gdn_conv_step", shape, t, (double) C, 0.0);
    ctx.free(bcs); ctx.free(bx); ctx.free(bkw); ctx.free(bo);
}

void bench_gdn_l2_norm(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int rows = 48, cols = 128;                    // h_v heads, S
    const uint64_t n = (uint64_t) rows * cols;
    std::vector<float> x = floats(n);
    Buf bx = alloc(ctx, n * 4);
    ctx.write(bx, x.data(), n * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_l2_norm.spv", 1, 12);
    struct { int32_t rows; int32_t cols; float eps; } pc{rows, cols, 1e-6f};
    Timing t = time_kernel(ctx, p, {&bx}, &pc, sizeof(pc), (uint32_t) rows, 1, 64, reps, warmups);
    report("gdn_l2_norm", "rows=48 cols=128", t, (double) n, 0.0);
    ctx.free(bx);
}

void bench_gdn_beta_gate(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48;
    std::vector<float> b = floats((size_t) h_v);
    Buf bb = alloc(ctx, (size_t) h_v * 4);
    ctx.write(bb, b.data(), (size_t) h_v * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_beta_gate.spv", 1, 4);
    struct { int32_t n; } pc{h_v};
    Timing t = time_kernel(ctx, p, {&bb}, &pc, sizeof(pc), (uint32_t) ((h_v + 255) / 256), 1, 128, reps, warmups);
    report("gdn_beta_gate", "h_v=48", t, (double) h_v, 0.0);
    ctx.free(bb);
}

void bench_gdn_gate(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48, n_tokens = 64;
    const int n = h_v * n_tokens;
    std::vector<float> alpha = floats((size_t) n), dt = floats((size_t) h_v), a = floats((size_t) h_v);
    for (auto& v : a) v = -std::fabs(v) - 0.1f;
    Buf bA = alloc(ctx, (size_t) n * 4), bD = alloc(ctx, (size_t) h_v * 4), bS = alloc(ctx, (size_t) h_v * 4),
        bG = alloc(ctx, (size_t) n * 4);
    ctx.write(bA, alpha.data(), (size_t) n * 4);
    ctx.write(bD, dt.data(), (size_t) h_v * 4);
    ctx.write(bS, a.data(), (size_t) h_v * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_gate.spv", 4, 8);
    struct { int32_t h_v; int32_t n_tokens; } pc{h_v, n_tokens};
    Timing t = time_kernel(ctx, p, {&bA, &bD, &bS, &bG}, &pc, sizeof(pc), (uint32_t) ((n + 255) / 256), 1, 64, reps,
                           warmups);
    report("gdn_gate", "h_v=48 n_tokens=64", t, (double) n, 0.0);
    ctx.free(bA); ctx.free(bD); ctx.free(bS); ctx.free(bG);
}

void bench_gdn_step(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    Buf bst = alloc(ctx, nstate * 4), bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4),
        bv = alloc(ctx, no * 4), bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4),
        bo = alloc(ctx, no * 4);
    ctx.write(bst, st.data(), nstate * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_step.spv", 7, 12);
    struct { int32_t S; int32_t h_k; int32_t h_v; } pc{S, h_k, h_v};
    Timing t = time_kernel(ctx, p, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pc, sizeof(pc),
                           (uint32_t) ((no + 255) / 256), 1, 8, reps, warmups);
    // per output (h,j): one S-long contract, one S-long rank-1 update, one S-long readout => 3*S MACs.
    report("gdn_step", "S=128 h_k=16 h_v=48", t, (double) no, (double) no * 3.0 * (double) S);
    ctx.free(bst); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2); ctx.free(bo);
}

void bench_gdn_out_norm(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48, S = 128;
    const size_t n = (size_t) h_v * S;
    std::vector<float> o = floats(n), z = floats(n), sn = floats((size_t) S);
    Buf bo = alloc(ctx, n * 4), bz = alloc(ctx, n * 4), bsn = alloc(ctx, (size_t) S * 4), by = alloc(ctx, n * 4);
    ctx.write(bo, o.data(), n * 4);
    ctx.write(bz, z.data(), n * 4);
    ctx.write(bsn, sn.data(), (size_t) S * 4);
    VkPipeline p = ctx.pipeline(dir + "/gdn_out_norm.spv", 4, 12);
    struct { int32_t h_v; int32_t S; float eps; } pc{h_v, S, 1e-6f};
    Timing t = time_kernel(ctx, p, {&bo, &bz, &bsn, &by}, &pc, sizeof(pc), (uint32_t) h_v, 1, 64, reps, warmups);
    report("gdn_out_norm", "h_v=48 S=128", t, (double) n, 0.0);
    ctx.free(bo); ctx.free(bz); ctx.free(bsn); ctx.free(by);
}

// =========================================================================================================
// THE IQ / BF16 DEQUANTISER (shaders/iq_dequant_f32.comp, dq_dispatch).  One workgroup per 256-element
// superblock (32 of its 256 lanes active - the engine's own thread mapping); the grid is the superblock
// count.  bf16 (ty 30), IQ4_NL (ty 20) and IQ2_S (ty 11, the format
// 20 of the resident model's 48 layers store their gate/up experts in) are measured.
// =========================================================================================================

void bench_iq_dequant(Ctx& ctx, const std::string& dir, int ty, const char* fmt, uint32_t sb, uint32_t n_sb,
                      int reps, int warmups) {
    const size_t n = (size_t) n_sb * 256;
    std::vector<uint8_t> w((size_t) n_sb * sb);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (i * 37 + 13);
    Buf b_w = alloc(ctx, w.size()), b_y = alloc(ctx, n * 4);
    Buf g[6];
    g[0] = alloc(ctx, sizeof(strata::vkport::kIq1sGrid));
    g[1] = alloc(ctx, sizeof(strata::vkport::kIq2sGrid));
    g[2] = alloc(ctx, sizeof(strata::vkport::kIq3xxsGrid));
    g[3] = alloc(ctx, sizeof(strata::vkport::kIq3sGrid));
    g[4] = alloc(ctx, sizeof(strata::vkport::kIq2xxsGrid));
    g[5] = alloc(ctx, sizeof(strata::vkport::kIq2xsGrid));
    ctx.write(b_w, w.data(), w.size());
    ctx.write(g[0], strata::vkport::kIq1sGrid, sizeof(strata::vkport::kIq1sGrid));
    ctx.write(g[1], strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    ctx.write(g[2], strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid));
    ctx.write(g[3], strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid));
    ctx.write(g[4], strata::vkport::kIq2xxsGrid, sizeof(strata::vkport::kIq2xxsGrid));
    ctx.write(g[5], strata::vkport::kIq2xsGrid, sizeof(strata::vkport::kIq2xsGrid));
    VkPipeline p = ctx.pipeline(dir + "/iq_dequant_f32.spv", 8, 4);
    struct { int ty; } pc{ty};
    Timing t = time_kernel(ctx, p, {&b_w, &g[0], &g[1], &g[2], &g[3], &g[4], &g[5], &b_y}, &pc, sizeof(pc), n_sb,
                           1, 8, reps, warmups);
    char name[48], shape[64];
    std::snprintf(name, sizeof name, "iq_dequant_f32/%s", fmt);
    std::snprintf(shape, sizeof shape, "n_sb=%u (%u floats)", n_sb, (unsigned) n);
    report(name, shape, t, (double) n, 0.0);
    ctx.free(b_w); ctx.free(b_y);
    for (Buf& b : g) ctx.free(b);
}

// =========================================================================================================
// AN MMVQ-FAMILY KERNEL - the quantised matvec.  iq2s_mmvq is the format that dominates the resident
// model's experts.  Two n_out values are measured on purpose (512 and 2048, a 4x work ratio): the SAME
// kernel at 4x the work must take ~4x the time, which is one of the two proofs that the timer resolves real
// work (the other is the cross-ICD gap run_bench.sh produces).  Both sizes are large enough to be
// work-bound, not dispatch-bound.
// =========================================================================================================

// =========================================================================================================
// IQ1_M MMVQ - THE WINDOW'S DOMINANT KERNEL, AND THE ONE ROW THIS HARNESS WAS MISSING.
// The RECORDED-arm histogram says the verify window's replay is `quantize_q8_1` 1,875 + `swiglu_f32` 1,296
// + `native_gu_any`/`native_down_any` 1,152 EACH + the four `fused_gr_*` at 576 each, i.e. ~2,300
// WEIGHT-READING GEMV dispatches per round against the pack's IQ1_M/IQ2_S/IQ3_*/IQ4_XS expert blobs.
// Every row this harness published before now was either elementwise or IQ2_S: `iq2s_mmvq` measures
// 22-46 us, NOT the "5-20 us" the engine's 64 us per dispatch was compared against.  This arm measures the
// format the resident model's gate/up rows actually use, at the engine's own shapes, in BOTH the harness's
// mapped memory and the engine's DEVICE_LOCAL arena type - so the engine's per-dispatch average can be read
// against a row for the kernel that actually fills it.
// =========================================================================================================
void bench_iq1m(Ctx& ctx, const std::string& dir, int n_out, int ncols, bool dev, int reps, int warmups) {
    const int n_in = 2560, nb = n_in / 256, row_bytes = nb * 56;   // IQ1_M: 56 bytes per 256-value block
    std::vector<uint8_t> w((size_t) n_out * row_bytes);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (i * 29 + 7);
    std::vector<uint8_t> act((size_t) ncols * (n_in / 32) * 36);
    for (size_t i = 0; i < act.size(); ++i) act[i] = (uint8_t) (i * 11 + 3);
    Buf b_w = dev ? ctx.alloc_device(w.size()) : alloc(ctx, w.size());
    Buf b_a = alloc(ctx, act.size()), b_g = alloc(ctx, sizeof(strata::vkport::kIq1sGrid)),
        b_y = alloc(ctx, (size_t) n_out * 4 + 64);
    ctx.write(b_w, w.data(), w.size());
    ctx.write(b_a, act.data(), act.size());
    ctx.write(b_g, strata::vkport::kIq1sGrid, sizeof(strata::vkport::kIq1sGrid));
    VkPipeline p = ctx.pipeline(dir + "/iq1m_mmvq.spv", 4, 16);
    struct { int n_in; int n_out; int row_bytes; int ncols; } pc{n_in, n_out, row_bytes, ncols};
    Timing t = time_kernel(ctx, p, {&b_w, &b_a, &b_g, &b_y}, &pc, sizeof(pc), (uint32_t) n_out, 1,
                           n_out <= 8 ? 32 : 8, reps, warmups);
    char name[40], shape[96];
    std::snprintf(name, sizeof name, "iq1m_mmvq/%s", dev ? "dev" : "mapped");
    std::snprintf(shape, sizeof shape, "n_in=2560 n_out=%d ncols=%d row_bytes=%d wbytes=%zu", n_out, ncols,
                  row_bytes, w.size());
    report(name, shape, t, (double) n_out * ncols, (double) n_in * n_out * ncols);
    ctx.free(b_w); ctx.free(b_a); ctx.free(b_g); ctx.free(b_y);
}

void bench_iq2s(Ctx& ctx, const std::string& dir, int n_out, int reps, int warmups) {
    const int n_in = 2560, ncols = 1;
    const int nb = n_in / 256, row_bytes = nb * 82;
    std::vector<uint8_t> w((size_t) n_out * row_bytes);
    for (size_t i = 0; i < w.size(); ++i) w[i] = (uint8_t) (i * 29 + 7);
    std::vector<uint8_t> act((size_t) (n_in / 32) * 36);
    for (size_t i = 0; i < act.size(); ++i) act[i] = (uint8_t) (i * 11 + 3);
    Buf b_w = alloc(ctx, w.size()), b_a = alloc(ctx, act.size()),
        b_g = alloc(ctx, sizeof(strata::vkport::kIq2sGrid)), b_y = alloc(ctx, (size_t) n_out * 4 + 64);
    ctx.write(b_w, w.data(), w.size());
    ctx.write(b_a, act.data(), act.size());
    ctx.write(b_g, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    VkPipeline p = ctx.pipeline(dir + "/iq2s_mmvq.spv", 4, 16);
    struct { int n_in; int n_out; int row_bytes; int ncols; } pc{n_in, n_out, row_bytes, ncols};
    Timing t = time_kernel(ctx, p, {&b_w, &b_a, &b_g, &b_y}, &pc, sizeof(pc), (uint32_t) n_out, 1,
                           n_out <= 8 ? 32 : 8, reps, warmups);
    char shape[80];
    std::snprintf(shape, sizeof shape, "n_in=2560 n_out=%d ncols=1", n_out);
    report("iq2s_mmvq", shape, t, (double) n_out * ncols, (double) n_in * n_out * ncols);
    ctx.free(b_w); ctx.free(b_a); ctx.free(b_g); ctx.free(b_y);
}

// =========================================================================================================
// THE SAMPLER - the ENGINE'S DEFAULT measured against the ONE-BLOCK f32 fallback, at the same shape and on
// the same device.  `sample_tokens` (src/kernels/cuda/sampler.cu:1005-1026) takes the SPLIT whenever
// `sampled_path() == Split` (the default - neither STRATA_OLD_SAMPLER nor STRATA_SAMPLER_ONE_BLOCK set),
// `n_blocks = ceil(n_vocab / 4096) <= 64` and `n_tokens <= 64` (kSplitMaxBlocks / kSplitMaxRows).  At the
// model's vocab 248320 that is 61 blocks, so the SPLIT is what the engine runs and the one-block kernel is
// the FALLBACK (a vocabulary wider than 262144, more than 64 rows, a captured stream, or no scratch).
// The `sampler_split` rows are therefore the DEFAULT's cost; the `sampler_kernel_f32` row is the fallback's.
// Both are ONE workgroup per token over the vocabulary.  The split accumulates its tail in double, so it is
// SKIPPED BY NAME on a device without shaderFloat64.
// =========================================================================================================

void bench_sampler(Ctx& ctx, const std::string& dir, int n_vocab, int reps, int warmups) {
    // THE PARAMETERS THE ENGINE ACTUALLY USES, read off `SamplerParams` (include/strata/kernels/sampler.hpp)
    // and generate.cpp's defaults: `top_k = 20` (and `sampled_k` keeps 1..63 as given, so k = 20) and
    // `penalty_last_n = 0` (the penalty window is DISABLED by default).  The rows below therefore come in
    // pairs: the parameters this harness used before (top_k 64, window 64) and the engine's DEFAULT
    // (top_k 20, window 0), so the two are never confused again.
    const int history_len = 64;                         // the artifact's window
    const int max_tokens = 64;                          // the engine's kSplitMaxRows (its split bound)
    std::vector<float> row((size_t) n_vocab * max_tokens);
    for (size_t i = 0; i < row.size(); ++i) row[i] = 0.25f * rndf();
    for (int t = 0; t < max_tokens; ++t) {              // a shortlist worth filtering, per row
        float* r = row.data() + (size_t) t * n_vocab;
        r[7] = 9.0f; r[1234] = 8.0f; r[99999 % n_vocab] = 7.5f;
    }
    std::vector<int32_t> hist((size_t) history_len, -1);
    for (int i = 0; i < 8; ++i) hist[(size_t) i] = 7;   // a real penalty hit on the head token
    Buf b_l = alloc(ctx, row.size() * 4), b_h = alloc(ctx, (size_t) history_len * 4),
        b_o = alloc(ctx, (size_t) max_tokens * 4);
    ctx.write(b_l, row.data(), row.size() * 4);
    ctx.write(b_h, hist.data(), hist.size() * 4);
    struct Pc {
        int n_vocab, n_tokens, history_len, penalty_last_n, top_k, min_keep;
        float temperature, top_p, min_p, penalty_repeat, penalty_freq, penalty_present;
        uint32_t seed_lo, seed_hi, counter_lo, counter_hi;
    } pc{};
    pc.n_vocab = n_vocab; pc.history_len = history_len; pc.penalty_last_n = history_len;
    pc.top_k = 64; pc.min_keep = 1; pc.temperature = 1.0f; pc.top_p = 0.95f; pc.min_p = 0.0f;
    pc.penalty_repeat = 1.0f; pc.penalty_freq = 0.0f; pc.penalty_present = 0.0f;
    pc.seed_lo = 0x12345678u; pc.seed_hi = 0x9abcdef0u; pc.counter_lo = 0; pc.counter_hi = 0;
    const int sreps = reps < 5 ? reps : 5, swarm = warmups < 2 ? warmups : 2;
    char shape[64];

    // ---- THE ONE-BLOCK f32 FALLBACK.  ONE dispatch per timed batch: its top-k is k rounds over the whole row
    // in ONE workgroup, already long enough that batching would rival a driver timeout.
    pc.n_tokens = 1;
    VkPipeline p_f32 = ctx.pipeline(dir + "/sampler_kernel_f32.spv", 3, (uint32_t) sizeof(Pc));
    Timing tf = time_kernel(ctx, p_f32, {&b_l, &b_h, &b_o}, &pc, (uint32_t) sizeof(pc), 1 /*groups*/, 1 /*groups_y*/,
                            1 /*batch*/, sreps, swarm);
    std::snprintf(shape, sizeof shape, "vocab=%d n_tokens=1", n_vocab);
    report("sampler_kernel_f32", shape, tf, (double) n_vocab, 0.0);

    // ---- THE DEFAULT: the SPLIT, at the single-token decode shape AND at the engine's row bound.
    if (!ctx.info().shader_float64) {
        std::printf("SKIP sampler_split              | device has no shaderFloat64 (its tail accumulates in double)\n");
    } else {
        VkPipeline p_split = ctx.pipeline(dir + "/sampler_split.spv", 3, (uint32_t) sizeof(Pc));
        // (top_k, penalty_last_n, row count, row name).  `top_k = 20, penalty_last_n = 0` IS the engine's
        // default (`SamplerParams`); `top_k = 64, penalty_last_n = 64` is what this harness measured before.
        struct Cfg { int k, win, nt; const char* name; };
        const Cfg cfgs[] = {
            {64, 64, 1, "sampler_split"},          // kept: the row the release note/README quote
            {64, 64, 64, "sampler_split_64"},      // kept: the engine's row bound at the old parameters
            {20, 0, 1, "sampler_split_def"},       // THE ENGINE'S DEFAULT: top_k 20, no penalty window
            {20, 64, 1, "sampler_split_k20w64"},   // the penalty scan's share at the DEFAULT k
            {64, 0, 1, "sampler_split_k64w0"},     // the k-round scan's share with the window OFF
        };
        double split1 = 0.0;
        for (const Cfg& c : cfgs) {
            pc.n_tokens = c.nt;
            pc.top_k = c.k;
            pc.penalty_last_n = c.win;
            Timing ts = time_kernel(ctx, p_split, {&b_l, &b_h, &b_o}, &pc, (uint32_t) sizeof(pc), (uint32_t) c.nt, 1,
                                    1, sreps, swarm);
            std::snprintf(shape, sizeof shape, "vocab=%d n_tokens=%d k=%d win=%d", n_vocab, c.nt, c.k, c.win);
            report(c.name, shape, ts, (double) n_vocab * c.nt, 0.0);
            if (c.nt == 1 && c.k == 64 && c.win == 64) split1 = ts.med;
        }
        std::printf("XPAIR sampler_split/sampler_kernel_f32 vocab=%d  %.4f / %.4f  = %.3f  (the DEFAULT over the fallback)\n",
                    n_vocab, split1, tf.med, tf.med > 0 ? split1 / tf.med : 0.0);
    }
    ctx.free(b_l); ctx.free(b_h); ctx.free(b_o);
}

// =========================================================================================================
// THE QUANTISE FAMILY.  q8_1 and q8_K need only f32/int8/int16 and run everywhere; q8_0's ggml bytes are
// computed from a float64 quotient (the reference's `rint`), so it runs only on a device that reports
// shaderFloat64 - on the Arc it is SKIPPED BY NAME, exactly as the numeric gate skips its case there.
// =========================================================================================================

void bench_quantize_q8_0(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    if (!ctx.info().shader_float64) {
        std::printf("SKIP quantize_q8_0                 | device has no shaderFloat64 (the reference divides in float64)\n");
        return;
    }
    const int n = 65536, nb = n / 32;
    std::vector<float> x = floats((size_t) n);
    Buf bx = alloc(ctx, (size_t) n * 4), b_blocks = alloc(ctx, (size_t) nb * 34 + 64);
    ctx.write(bx, x.data(), (size_t) n * 4);
    VkPipeline p = ctx.pipeline(dir + "/quantize_q8_0.spv", 2, 4);
    struct { int n_blocks; } pc{nb};
    Timing t = time_kernel(ctx, p, {&bx, &b_blocks}, &pc, sizeof(pc), (uint32_t) ((nb + 255) / 256), 1, 16, reps,
                           warmups);
    report("quantize_q8_0", "n=65536 (2048 blocks)", t, (double) n, 0.0);
    ctx.free(bx); ctx.free(b_blocks);
}

void bench_quantize_q8_1(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n_in = 2560, ncols = 8;
    std::vector<float> x((size_t) n_in * ncols);
    for (auto& v : x) v = 0.5f * rndf();
    Buf b_x = alloc(ctx, x.size() * 4), b_y = alloc(ctx, (size_t) ncols * (n_in / 32) * 36 + 64);
    ctx.write(b_x, x.data(), x.size() * 4);
    VkPipeline p = ctx.pipeline(dir + "/quantize_q8_1.spv", 2, 8);
    struct { int n_in; int ncols; } pc{n_in, ncols};
    const uint32_t n_wg = (uint32_t) (ncols * ((n_in + 255) / 256));
    Timing t = time_kernel(ctx, p, {&b_x, &b_y}, &pc, sizeof(pc), n_wg, 1, 16, reps, warmups);
    report("quantize_q8_1", "n_in=2560 ncols=8", t, (double) n_in * ncols, 0.0);
    ctx.free(b_x); ctx.free(b_y);
}

void bench_quantize_q8_K(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n = 65536, nb = n / 256;
    std::vector<float> x = floats((size_t) n);
    Buf bx = alloc(ctx, (size_t) n * 4), b_blocks = alloc(ctx, (size_t) nb * 292 + 64);
    ctx.write(bx, x.data(), (size_t) n * 4);
    VkPipeline p = ctx.pipeline(dir + "/quantize_q8_K.spv", 2, 4);
    struct { int n_blocks; } pc{nb};
    Timing t = time_kernel(ctx, p, {&bx, &b_blocks}, &pc, sizeof(pc), (uint32_t) ((nb + 255) / 256), 1, 16, reps,
                           warmups);
    report("quantize_q8_K", "n=65536 (256 blocks)", t, (double) n, 0.0);
    ctx.free(bx); ctx.free(b_blocks);
}

// THE S-FAMILY SPLIT GEMV over a QUANTIZED activation: the pair `shared_expert`'s canonical path dispatches
// (this batch).  Three rows, the SHARED EXPERT's own projection shapes: the gate/up projection at Q8_K (n_in
// 2560 -> n_out 640, group 64 and group 32) and `ffn_down_shexp` at Q8_0 (n_in 640 -> n_out 2560, group 32 -
// the shape Q8_K is structurally impossible for).  The shader is ONE for both activation kinds; only `q8k` and
// the form change, so these are the honest per-dispatch costs of the shared expert's three GEMVs.
void bench_s_gemv_q8_split(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    if (!ctx.info().storage_buffer_8bit) {
        std::printf("SKIP s_gemv_q8_split              | device lacks storageBuffer8BitAccess\n");
        return;
    }
    struct Case { bool q8k; int bits, bias, group, n_in, n_out; const char* shape; };
    const Case cases[] = {
        {true,  4, 0, 64, 2560, 640, "Q8_K n_in=2560 n_out=640 group=64 (gate/up)"},
        {true,  8, 0, 32, 2560, 640, "Q8_K n_in=2560 n_out=640 group=32 (S8)"},
        {false, 8, 0, 32,  640, 2560, "Q8_0 n_in=640 n_out=2560 group=32 (ffn_down_shexp)"},
    };
    for (const Case& c : cases) {
        const int per_byte = 8 / c.bits, n_groups = c.n_in / c.group;
        const int blk_elems = c.q8k ? 256 : 32, blk_bytes = c.q8k ? 292 : 34;
        const int n_blocks = c.n_in / blk_elems;
        std::vector<uint8_t> act((size_t) n_blocks * blk_bytes, 0);
        for (size_t i = 0; i < act.size(); ++i) act[i] = (uint8_t) ((i * 13 + 7) & 0xFF);
        std::vector<uint8_t> codes((size_t) c.n_out * (c.n_in / per_byte), 0);
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = (uint8_t) ((i * 37 + 11) & 0xFF);
        std::vector<float> scales((size_t) c.n_out * n_groups, 0.0f);
        for (size_t i = 0; i < scales.size(); ++i) scales[i] = 0.01f * (float) (1 + (i % 7));
        Buf ba = alloc(ctx, act.size()), bc = alloc(ctx, codes.size()), bs = alloc(ctx, scales.size() * 4);
        Buf bo = alloc(ctx, scales.size() * 4 + 16), by = alloc(ctx, (size_t) c.n_out * 4 + 64);
        ctx.write(ba, act.data(), act.size());
        ctx.write(bc, codes.data(), codes.size());
        ctx.write(bs, scales.data(), scales.size() * 4);
        ctx.write(bo, scales.data(), scales.size() * 4);   // the OFFSET binding is always bound (never read)
        const int byte_shift = (per_byte == 4) ? 2 : ((per_byte == 2) ? 1 : 0);
        int group_shift = 0; while ((1 << group_shift) < c.group) ++group_shift;
        struct { int n_in, n_out, code_bits, byte_shift, bias, codebook, group_shift, has_offset, q8k; } pc{
            c.n_in, c.n_out, c.bits, byte_shift, c.bias, 0, group_shift, 0, c.q8k ? 1 : 0};
        VkPipeline p = ctx.pipeline(dir + "/s_gemv_q8_split.spv", 5, (int) sizeof(pc));
        Timing t = time_kernel(ctx, p, {&ba, &bc, &bs, &bo, &by}, &pc, sizeof(pc), (uint32_t) c.n_out, 1, 16, reps,
                               warmups);
        report("s_gemv_q8_split", c.shape, t, (double) c.n_out, (double) c.n_in * c.n_out);
        ctx.free(ba); ctx.free(bc); ctx.free(bs); ctx.free(bo); ctx.free(by);
    }
}

// =========================================================================================================
// THE PERFORMANCE TIER, class B: the NATIVE fast paths against the LEGACY kernel each replaces, at the
// SAME shape on the SAME device.  This is the harness's whole purpose - a class-B increment's claim is its
// before/after, and the ratio is the deliverable.  Each pair prints both rows and an "XPAIR" line with
// native/legacy so the direction is unambiguous (a ratio < 1.0 means the native kernel is FASTER).
// A native kernel that is NOT faster is a finding to report, not to tune away.
// =========================================================================================================

void bench_rope_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int rows = 512, head_dim = 256, n_rot = 64, half = 32, max_pos = 256;
    std::vector<float> x = floats((size_t) rows * head_dim);
    std::vector<int> pos((size_t) rows);
    for (int r = 0; r < rows; ++r) pos[(size_t) r] = r % max_pos;
    std::vector<float> cs((size_t) max_pos * half), sn((size_t) max_pos * half);
    for (int p = 0; p < max_pos; ++p)
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(1.0e7, -2.0 * (double) i / (double) n_rot);
            const double a = (double) p * inv;
            cs[(size_t) p * half + i] = (float) std::cos(a);
            sn[(size_t) p * half + i] = (float) std::sin(a);
        }
    Buf bx = alloc(ctx, x.size() * 4), bo = alloc(ctx, x.size() * 4), bp = alloc(ctx, (size_t) rows * 4),
        bc = alloc(ctx, cs.size() * 4), bs = alloc(ctx, sn.size() * 4), bm = alloc(ctx, 4);
    ctx.write(bx, x.data(), x.size() * 4);
    ctx.write(bp, pos.data(), (size_t) rows * 4);
    ctx.write(bc, cs.data(), cs.size() * 4);
    ctx.write(bs, sn.data(), sn.size() * 4);
    const int32_t z = 0;
    ctx.write(bm, &z, 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "rows=%d head_dim=%d n_rot=64", rows, head_dim);
    // legacy: rope_neox, one thread per ROW (the CUDA's own decomposition) - a host-built float64 table.
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/rope_neox.spv", 6, 16);
        struct { int32_t rows, head_dim, n_rot, mrope; } pc{rows, head_dim, n_rot, 0};
        tl = time_kernel(ctx, p, {&bx, &bo, &bc, &bs, &bp, &bm}, &pc, sizeof(pc),
                         (uint32_t) ((rows + 255) / 256), 1, 16, reps, warmups);
        report("rope_neox (legacy)", shape, tl, (double) rows * head_dim, 0.0);
    }
    // native: one thread per (row, PAIR), the angle computed on device in f32.
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_rope_apply.spv", 4, 40);
        struct { int32_t rows, head_dim, n_rot, mrope; float theta_scale, freq_scale, corr_low, corr_high, ext_factor, mscale; } pc;
        pc.rows = rows; pc.head_dim = head_dim; pc.n_rot = n_rot; pc.mrope = 0;
        pc.theta_scale = std::pow(1.0e7f, -2.0f / (float) n_rot);
        pc.freq_scale = 1.0f; pc.corr_low = 0; pc.corr_high = 0; pc.ext_factor = 0; pc.mscale = 1.0f;
        tn = time_kernel(ctx, p, {&bx, &bo, &bp, &bm}, &pc, sizeof(pc),
                         (uint32_t) ((head_dim / 2 + 255) / 256), (uint32_t) rows, 16, reps, warmups);
        report("native_rope_apply", shape, tn, (double) rows * head_dim, 0.0);
    }
    std::printf("XPAIR rope %s | legacy rope_neox | native native_rope_apply | native/legacy %.3f\n", shape,
                tn.med / tl.med);
    ctx.free(bx); ctx.free(bo); ctx.free(bp); ctx.free(bc); ctx.free(bs); ctx.free(bm);
}

void bench_router_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int NE = 512, K = 10, NT = 16;
    std::vector<float> logits = floats((size_t) NT * NE);
    Buf bl = alloc(ctx, logits.size() * 4), bi = alloc(ctx, (size_t) NT * K * 4), bw = alloc(ctx, (size_t) NT * K * 4);
    ctx.write(bl, logits.data(), logits.size() * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "n_tokens=%d n_expert=512 k=10", NT);
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/router_top10_f32.spv", 3, 12);
        struct { int32_t n_tokens, n_expert, k; } pc{NT, NE, K};
        tl = time_kernel(ctx, p, {&bl, &bi, &bw}, &pc, sizeof(pc), (uint32_t) NT, 1, 16, reps, warmups);
        report("router_top10_f32 (legacy)", shape, tl, (double) NT * NE, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_router_top10.spv", 3, 4);
        struct { int32_t n_tokens; } pc{NT};
        tn = time_kernel(ctx, p, {&bl, &bi, &bw}, &pc, sizeof(pc), (uint32_t) NT, 1, 16, reps, warmups);
        report("native_router_top10", shape, tn, (double) NT * NE, 0.0);
    }
    std::printf("XPAIR router %s | legacy router_top10_f32 | native native_router_top10 | native/legacy %.3f\n", shape,
                tn.med / tl.med);
    ctx.free(bl); ctx.free(bi); ctx.free(bw);
}

void bench_moe_combine_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n_embd = 2560, k = 10;
    std::vector<float> parts = floats((size_t) k * n_embd), w = floats((size_t) k), sh = floats((size_t) n_embd);
    Buf bp = alloc(ctx, parts.size() * 4), bw = alloc(ctx, (size_t) k * 4), bs = alloc(ctx, (size_t) n_embd * 4),
        by = alloc(ctx, (size_t) n_embd * 4);
    ctx.write(bp, parts.data(), parts.size() * 4);
    ctx.write(bw, w.data(), (size_t) k * 4);
    ctx.write(bs, sh.data(), (size_t) n_embd * 4);
    struct { int32_t n_embd, k, has_shared; } pc{n_embd, k, 1};
    char shape[64];
    std::snprintf(shape, sizeof shape, "n_embd=2560 k=10 shared=1");
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/moe_combine_f32.spv", 4, 12);
        tl = time_kernel(ctx, p, {&bp, &bw, &bs, &by}, &pc, sizeof(pc), (uint32_t) ((n_embd + 255) / 256), 1, 64,
                         reps, warmups);
        report("moe_combine_f32 (legacy)", shape, tl, (double) n_embd, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_moe_combine.spv", 4, 12);
        tn = time_kernel(ctx, p, {&bp, &bw, &bs, &by}, &pc, sizeof(pc), (uint32_t) ((n_embd + 255) / 256), 1, 64,
                         reps, warmups);
        report("native_moe_combine", shape, tn, (double) n_embd, 0.0);
    }
    std::printf("XPAIR moe_combine %s | legacy moe_combine_f32 | native native_moe_combine | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    ctx.free(bp); ctx.free(bw); ctx.free(bs); ctx.free(by);
}

void bench_rms_norm_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int rows = 128, cols = 2560;
    const uint64_t n = (uint64_t) rows * cols;
    std::vector<float> x = floats(n), g = floats((size_t) cols);
    Buf bx = alloc(ctx, n * 4), bg = alloc(ctx, (size_t) cols * 4), bo = alloc(ctx, n * 4);
    ctx.write(bx, x.data(), n * 4);
    ctx.write(bg, g.data(), (size_t) cols * 4);
    struct { int32_t rows, cols; float eps; } pc{rows, cols, 1e-6f};
    char shape[64];
    std::snprintf(shape, sizeof shape, "rows=%d cols=%d", rows, cols);
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/rms_norm.spv", 2, 12);
        tl = time_kernel(ctx, p, {&bx, &bg}, &pc, sizeof(pc), (uint32_t) rows, 1, 64, reps, warmups);
        report("rms_norm (legacy)", shape, tl, (double) n, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_qsa_rms_norm_weighted.spv", 3, 12);
        tn = time_kernel(ctx, p, {&bx, &bg, &bo}, &pc, sizeof(pc), (uint32_t) rows, 1, 64, reps, warmups);
        report("native_qsa_rms_norm_weighted", shape, tn, (double) n, 0.0);
    }
    std::printf("XPAIR rms_norm %s | legacy rms_norm | native native_qsa_rms_norm_weighted | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    ctx.free(bx); ctx.free(bg); ctx.free(bo);
}

// THE QSA GATE PAIR: `native_qsa_gate_apply` against the legacy `qsa_gate_apply_f32` it replaces at the SAME
// shape.  There is NO dispatch chain here - the layer's QSA gate is ONE dispatch either way
// (`native_qsa_gate_apply` OR `qsa_gate_apply_f32`, layer.cpp:1010-1012), unlike the fused GDN paths - so this
// is a drop-in pair and is EXPECTED to wash: both kernels are one thread per output element, no reduction, no
// shared memory, and (because the target has no shaderFloat64) both compute the sigmoid and the product in f32.
// The pair is printed so the wash is measured rather than assumed.
void bench_qsa_gate_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int nh = 24, hd = 256;                 // the artifact: 24 query heads x head_dim 256
    const size_t n = (size_t) nh * hd;
    std::vector<float> attn = floats(n), qf = floats((size_t) nh * 2 * hd);
    Buf ba = alloc(ctx, n * 4), bq = alloc(ctx, (size_t) nh * 2 * hd * 4), bo = alloc(ctx, n * 4);
    ctx.write(ba, attn.data(), n * 4);
    ctx.write(bq, qf.data(), qf.size() * 4);
    struct { int32_t n_head; int32_t head_dim; } pc{nh, hd};
    const uint32_t groups = (uint32_t) ((n + 255u) / 256u);
    char shape[64];
    std::snprintf(shape, sizeof shape, "n_head=%d head_dim=%d", nh, hd);
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/qsa_gate_apply_f32.spv", 3, 8);
        tl = time_kernel(ctx, p, {&ba, &bq, &bo}, &pc, sizeof(pc), groups, 1, 64, reps, warmups);
        report("qsa_gate_apply_f32 (legacy)", shape, tl, (double) n, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_qsa_gate_apply.spv", 3, 8);
        tn = time_kernel(ctx, p, {&ba, &bq, &bo}, &pc, sizeof(pc), groups, 1, 64, reps, warmups);
        report("native_qsa_gate_apply", shape, tn, (double) n, 0.0);
    }
    std::printf("XPAIR qsa_gate %s | legacy qsa_gate_apply_f32 | native native_qsa_gate_apply | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    ctx.free(ba); ctx.free(bq); ctx.free(bo);
}

// THE DEFAULT QSA DECODE ATTENTION: `qsa_decode_attn_step` (layer.cpp:980) - the KV POOLS read through the PAGE
// TABLE with an online softmax.  This is the port's RE-DERIVATION of the engine's chunk+merge CUDA onto ONE
// WORKGROUP PER QUERY HEAD with barrier-tree reductions (subgroup ops are banned; the CUDA's own header calls
// the chunked decomposition the performance form, and this is the correctness form).  It is a SOLO row: the
// symbol had NO shader in this tree before this batch, so there is no second implementation to pair against.
// Timed at the artifact's geometry (24 query heads, 2 KV heads, head_dim 256, page_size 4) across the engine's
// real selection widths (`qsa_selection_width`, `idx_top_k = 2048`), so the row is the cost of the attention the
// engine actually runs.  The pool is a CONSTANT f16 pattern (1.0 K / 0.5 V) - the arithmetic and the traffic are
// the same as for any data, and a timing row needs no fixture.
void bench_qsa_decode_attn(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int NH = 24, KH = 2, HD = 256, PS = 4, PAGES = 2048 / PS + 8;
    const int rows = PAGES * KH * PS;
    std::vector<uint16_t> kp((size_t) rows * HD, 0x3C00u), vp((size_t) rows * HD, 0x3800u);   // f16 1.0 / 0.5
    std::vector<int32_t> table(PAGES);
    for (int i = 0; i < PAGES; ++i) table[i] = i;
    std::vector<float> q((size_t) NH * HD);
    for (size_t i = 0; i < q.size(); ++i) q[i] = 0.02f * (float) ((int) (i % 13) - 6);
    Buf bq = alloc(ctx, q.size() * 4), bk = alloc(ctx, kp.size() * 2), bv = alloc(ctx, vp.size() * 2);
    Buf bt = alloc(ctx, table.size() * 4), bi = alloc(ctx, (size_t) 2048 * 4), bs = alloc(ctx, 5 * 4);
    Buf bo = alloc(ctx, (size_t) NH * HD * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, kp.data(), kp.size() * 2);
    ctx.write(bv, vp.data(), vp.size() * 2);
    ctx.write(bt, table.data(), table.size() * 4);
    const int widths[] = {256, 1024, 2048};               // the engine's real `n_ids` grows to idx_top_k = 2048
    for (int n_ids : widths) {
        std::vector<int32_t> ids((size_t) n_ids);
        for (int c = 0; c < n_ids; ++c) ids[(size_t) c] = (c * 3) % (PAGES * PS);
        std::vector<int32_t> step = {0, 0, 0, n_ids, 0};
        ctx.write(bi, ids.data(), ids.size() * 4);
        ctx.write(bs, step.data(), step.size() * 4);
        struct { int32_t n_head, kv_heads, head_dim, page_size, mode; } pc{NH, KH, HD, PS, 0};
        VkPipeline p = ctx.pipeline(dir + "/qsa_decode_attn.spv", 11, sizeof(pc));
        const Timing t = time_kernel(ctx, p, {&bq, &bk, &bv, &bt, &bt, &bt, &bt, &bt, &bi, &bs, &bo}, &pc,
                                     sizeof(pc), (uint32_t) NH, 1, 1, reps, warmups);
        char shape[96];
        std::snprintf(shape, sizeof shape, "n_head=%d kv_heads=%d head_dim=%d n_ids=%d page_size=%d", NH, KH, HD,
                      n_ids, PS);
        report("qsa_decode_attn (default)", shape, t, (double) NH * HD, 0.0);
    }
    ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bt); ctx.free(bi); ctx.free(bs); ctx.free(bo);
}

// =========================================================================================================
// THE PERFORMANCE TIER, class B, batch 2: the native GDN / DeltaNet MIXER kernels, each against the legacy
// kernel it replaces at the same shape on the same device.  Same XPAIR convention (native/legacy, < 1.0 faster).
// =========================================================================================================

void bench_gdn_conv_silu_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int C = 2560, dc = 4;                         // n_embd channels, d_conv
    const size_t hist = (size_t) C * 3;
    std::vector<float> cs = floats(hist), x = floats((size_t) C), kw = floats((size_t) C * dc);
    Buf bcs = alloc(ctx, hist * 4), bx = alloc(ctx, (size_t) C * 4), bkw = alloc(ctx, (size_t) C * dc * 4),
        braw = alloc(ctx, (size_t) C * 4), bsilu = alloc(ctx, (size_t) C * 4), bo = alloc(ctx, (size_t) C * 4);
    ctx.write(bcs, cs.data(), hist * 4);
    ctx.write(bx, x.data(), (size_t) C * 4);
    ctx.write(bkw, kw.data(), (size_t) C * dc * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "C=%d d_conv=4", C);
    struct { int32_t channels; int32_t d_conv; } pcc{C, dc};
    // legacy: gdn_conv_step, one thread per channel, one output.
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/gdn_conv_step.spv", 4, 8);
        tl = time_kernel(ctx, p, {&bcs, &bx, &bkw, &bo}, &pcc, sizeof(pcc), (uint32_t) ((C + 255) / 256), 1, 64,
                         reps, warmups);
        report("gdn_conv_step (legacy)", shape, tl, (double) C, 0.0);
    }
    // native: native_gdn_conv_silu, one thread per channel, TWO outputs (raw + fused SiLU).
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_conv_silu.spv", 5, 8);
        tn = time_kernel(ctx, p, {&bcs, &bx, &bkw, &braw, &bsilu}, &pcc, sizeof(pcc), (uint32_t) ((C + 255) / 256),
                         1, 64, reps, warmups);
        report("native_gdn_conv_silu", shape, tn, (double) C, 0.0);
    }
    std::printf("XPAIR gdn_conv %s | legacy gdn_conv_step | native native_gdn_conv_silu | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    // THE FUSION, MEASURED.  Per DISPATCH the native writes one extra output and computes the SiLU, so the row
    // above can read slower - but the legacy branch is conv_step -> a D2D copy -> silu_inplace (layer.cpp:255-257),
    // i.e. it runs a SECOND dispatch the native removes.  The copy cannot be timed as a kernel; the two KERNELS
    // can, so this row times conv_step + silu_f32 in one recorded batch and reports native over that chain.
    {
        VkPipeline pc_ = ctx.pipeline(dir + "/gdn_conv_step.spv", 4, 8);
        VkPipeline ps_ = ctx.pipeline(dir + "/silu_f32.spv", 1, 4);
        struct { int32_t n; } pcs{C};
        Timing tc = time_two(ctx, pc_, {&bcs, &bx, &bkw, &bo}, &pcc, sizeof(pcc), (uint32_t) ((C + 255) / 256), 1,
                             ps_, {&bo}, &pcs, sizeof(pcs), (uint32_t) ((C + 255) / 256), 1, 64, reps, warmups);
        report("gdn_conv_step+silu_f32 (legacy chain)", shape, tc, (double) C, 0.0);
        std::printf("XPAIR gdn_conv_fused %s | legacy conv_step+silu_f32 (2 dispatches) | native "
                    "native_gdn_conv_silu (1) | native/chain %.3f\n", shape, tn.med / tc.med);
    }
    ctx.free(bcs); ctx.free(bx); ctx.free(bkw); ctx.free(braw); ctx.free(bsilu); ctx.free(bo);
}

void bench_gdn_l2_norm_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int rows = 48, cols = 128;                    // h_v heads, S
    const uint64_t n = (uint64_t) rows * cols;
    std::vector<float> x = floats(n);
    Buf bx = alloc(ctx, n * 4);
    ctx.write(bx, x.data(), n * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "rows=48 cols=128");
    Timing tl;
    {
        struct { int32_t rows, cols; float eps; } pcl{rows, cols, 1e-6f};
        VkPipeline p = ctx.pipeline(dir + "/gdn_l2_norm.spv", 1, 12);
        tl = time_kernel(ctx, p, {&bx}, &pcl, sizeof(pcl), (uint32_t) rows, 1, 64, reps, warmups);
        report("gdn_l2_norm (legacy)", shape, tl, (double) n, 0.0);
    }
    Timing tn;
    {
        const float inv_sqrt_cols = 1.0f / std::sqrt((float) cols);
        struct { int32_t rows, cols; float eps; float inv_sqrt_cols; } pcn{rows, cols, 1e-6f, inv_sqrt_cols};
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_l2_norm.spv", 1, 16);
        tn = time_kernel(ctx, p, {&bx}, &pcn, sizeof(pcn), (uint32_t) rows, 1, 64, reps, warmups);
        report("native_gdn_l2_norm", shape, tn, (double) n, 0.0);
    }
    std::printf("XPAIR gdn_l2_norm %s | legacy gdn_l2_norm | native native_gdn_l2_norm | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    ctx.free(bx);
}

void bench_gdn_beta_gate_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48;
    std::vector<float> b = floats((size_t) h_v);
    Buf bb = alloc(ctx, (size_t) h_v * 4);
    ctx.write(bb, b.data(), (size_t) h_v * 4);
    struct { int32_t n; } pc{h_v};
    const uint32_t groups = (uint32_t) ((h_v + 255) / 256);
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/gdn_beta_gate.spv", 1, 4);
        tl = time_kernel(ctx, p, {&bb}, &pc, sizeof(pc), groups, 1, 128, reps, warmups);
        report("gdn_beta_gate (legacy)", "h_v=48", tl, (double) h_v, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_beta_gate.spv", 1, 4);
        tn = time_kernel(ctx, p, {&bb}, &pc, sizeof(pc), groups, 1, 128, reps, warmups);
        report("native_gdn_beta_gate", "h_v=48", tn, (double) h_v, 0.0);
    }
    std::printf("XPAIR gdn_beta_gate h_v=48 | legacy gdn_beta_gate | native native_gdn_beta_gate | native/legacy %.3f\n",
                tn.med / tl.med);
    ctx.free(bb);
}

// batch 3: the remaining three native GDN / DeltaNet mixer kernels.  `native_gdn_gate` and
// `native_gdn_out_norm` are drop-in replacements (one dispatch each side, the same work per element), so
// their pairs are expected to be washes; `native_gdn_step` is ALSO measured against the two-dispatch legacy
// chain it replaces (`scale_inplace` + `gdn_step`), which is the layer-level comparison.

void bench_gdn_gate_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48;                                 // ssm_v_heads; the native wrapper is per-head, one token
    std::vector<float> alpha = floats((size_t) h_v), dt = floats((size_t) h_v), a = floats((size_t) h_v);
    for (auto& v : a) v = -std::fabs(v) - 0.1f;         // ssm_a = -exp(A_log)
    Buf bA = alloc(ctx, (size_t) h_v * 4), bD = alloc(ctx, (size_t) h_v * 4), bS = alloc(ctx, (size_t) h_v * 4),
        bG = alloc(ctx, (size_t) h_v * 4);
    ctx.write(bA, alpha.data(), (size_t) h_v * 4);
    ctx.write(bD, dt.data(), (size_t) h_v * 4);
    ctx.write(bS, a.data(), (size_t) h_v * 4);
    const uint32_t groups = (uint32_t) ((h_v + 255) / 256);
    Timing tl;
    {
        // legacy gdn_gate at the shape the layer uses it: n_tokens = 1, h_v = 48 (layer.cpp:300).
        VkPipeline p = ctx.pipeline(dir + "/gdn_gate.spv", 4, 8);
        struct { int32_t h_vv; int32_t n_tokens; } pc{h_v, 1};
        tl = time_kernel(ctx, p, {&bA, &bD, &bS, &bG}, &pc, sizeof(pc), groups, 1, 128, reps, warmups);
        report("gdn_gate (legacy)", "h_v=48 n_tokens=1", tl, (double) h_v, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_gate.spv", 4, 4);
        struct { int32_t heads; } pc{h_v};
        tn = time_kernel(ctx, p, {&bA, &bD, &bS, &bG}, &pc, sizeof(pc), groups, 1, 128, reps, warmups);
        report("native_gdn_gate", "heads=48", tn, (double) h_v, 0.0);
    }
    std::printf("XPAIR gdn_gate h_v=48 | legacy gdn_gate | native native_gdn_gate | native/legacy %.3f\n",
                tn.med / tl.med);
    ctx.free(bA); ctx.free(bD); ctx.free(bS); ctx.free(bG);
}

void bench_gdn_out_norm_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int h_v = 48, S = 128;                        // the native wrapper requires cols == 128
    const size_t n = (size_t) h_v * S;
    std::vector<float> o = floats(n), z = floats(n), sn = floats((size_t) S);
    Buf bo = alloc(ctx, n * 4), bz = alloc(ctx, n * 4), bsn = alloc(ctx, (size_t) S * 4), by = alloc(ctx, n * 4);
    ctx.write(bo, o.data(), n * 4);
    ctx.write(bz, z.data(), n * 4);
    ctx.write(bsn, sn.data(), (size_t) S * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "h_v=%d S=%d", h_v, S);
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/gdn_out_norm.spv", 4, 12);
        struct { int32_t h_vv; int32_t S; float eps; } pc{h_v, S, 1e-6f};
        tl = time_kernel(ctx, p, {&bo, &bz, &bsn, &by}, &pc, sizeof(pc), (uint32_t) h_v, 1, 64, reps, warmups);
        report("gdn_out_norm (legacy)", shape, tl, (double) n, 0.0);
    }
    Timing tn;
    {
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_out_norm.spv", 4, 12);
        struct { int32_t heads; int32_t S; float eps; } pc{h_v, S, 1e-6f};
        tn = time_kernel(ctx, p, {&bo, &bz, &bsn, &by}, &pc, sizeof(pc), (uint32_t) h_v, 1, 64, reps, warmups);
        report("native_gdn_out_norm", shape, tn, (double) n, 0.0);
    }
    std::printf("XPAIR gdn_out_norm %s | legacy gdn_out_norm | native native_gdn_out_norm | native/legacy %.3f\n",
                shape, tn.med / tl.med);
    ctx.free(bo); ctx.free(bz); ctx.free(bsn); ctx.free(by);
}

void bench_gdn_step_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;              // the native wrapper requires S == 128
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    Buf bst = alloc(ctx, nstate * 4), bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4),
        bv = alloc(ctx, no * 4), bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4),
        bo = alloc(ctx, no * 4);
    ctx.write(bst, st.data(), nstate * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    const float scale = 1.0f / std::sqrt((float) S);
    char shape[64];
    std::snprintf(shape, sizeof shape, "S=128 h_k=16 h_v=48 (3 MiB state)");
    Timing tl;
    {
        VkPipeline p = ctx.pipeline(dir + "/gdn_step.spv", 7, 12);
        struct { int32_t S; int32_t h_k; int32_t h_v; } pc{S, h_k, h_v};
        tl = time_kernel(ctx, p, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pc, sizeof(pc),
                         (uint32_t) ((no + 255) / 256), 1, 8, reps, warmups);
        report("gdn_step (legacy)", shape, tl, (double) no, (double) no * 3.0 * (double) S);
    }
    Timing tn;
    {
        // one THREAD per column (the coalesced legacy decomposition, carrying the native arithmetic and its
        // fused readout scale); the dispatch is groups_for(h_v * S).
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
        struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pc{S, h_k, h_v, scale};
        tn = time_kernel(ctx, p, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pc, sizeof(pc),
                         (uint32_t) ((no + 255) / 256), 1, 8, reps, warmups);
        report("native_gdn_step", shape, tn, (double) no, (double) no * 3.0 * (double) S);
    }
    std::printf("XPAIR gdn_step %s | legacy gdn_step | native native_gdn_step | native/legacy %.3f\n", shape,
                tn.med / tl.med);
    // THE FUSION, MEASURED.  The native body folds the `1/sqrt(S)` readout scale into the kernel; the legacy
    // branch applies it to q in a SEPARATE `scale_inplace` dispatch (layer.cpp:276).  This row times
    // scale_inplace + gdn_step in one recorded batch and reports native over that two-dispatch chain.  NOTE:
    // scale is applied to q in place once per batch iteration, so q decays toward zero across the timed
    // replays; neither kernel's cost depends on the values (no data-dependent branch), so the timing is
    // unaffected - stated rather than hidden.
    {
        VkPipeline psc = ctx.pipeline(dir + "/scale.spv", 1, 8);
        VkPipeline pst = ctx.pipeline(dir + "/gdn_step.spv", 7, 12);
        struct { int32_t n; float s; } pcs{(int32_t) (h_k * S), scale};
        struct { int32_t S; int32_t h_k; int32_t h_v; } pct{S, h_k, h_v};
        Timing tc = time_two(ctx, psc, {&bq}, &pcs, sizeof(pcs), (uint32_t) ((h_k * S + 255) / 256), 1,
                             pst, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pct, sizeof(pct),
                             (uint32_t) ((no + 255) / 256), 1, 8, reps, warmups);
        report("scale+gdn_step (legacy chain)", shape, tc, (double) no, (double) no * 3.0 * (double) S);
        std::printf("XPAIR gdn_step_fused %s | legacy scale+gdn_step (2 dispatches) | native native_gdn_step (1) | "
                    "native/chain %.3f\n", shape, tn.med / tc.med);
    }
    ctx.free(bst); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2); ctx.free(bo);
}

// =========================================================================================================
// THE PERFORMANCE TIER, class B, batch 4: the THREE FUSED GDN PATHS.  Each is measured against (a) the
// multi-dispatch chain it replaces - where a fusion shows, by removing dispatches - and (b) the non-fused
// native kernel(s) it also replaces, where a comparison exists.  Same XPAIR convention (fused/legacy or
// fused/chain, < 1.0 means the fused path is faster).
// =========================================================================================================

void bench_fused_gdn_conv_l2_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int C = 10240, dc = 4, qk = 32, k_heads = 16;      // the model: ssm_conv_channels, 2*ssm_k_heads
    const size_t hist = (size_t) C * 3;
    std::vector<float> cs = floats(hist), x = floats((size_t) C), kw = floats((size_t) C * dc);
    Buf bcs = alloc(ctx, hist * 4), bx = alloc(ctx, (size_t) C * 4), bkw = alloc(ctx, (size_t) C * dc * 4),
        braw = alloc(ctx, (size_t) C * 4), bh = alloc(ctx, (size_t) C * 4);
    ctx.write(bcs, cs.data(), hist * 4);
    ctx.write(bx, x.data(), (size_t) C * 4);
    ctx.write(bkw, kw.data(), (size_t) C * dc * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "C=%d qk_heads=%d d_conv=4", C, qk);
    const uint32_t cg = (uint32_t) ((C + 255) / 256);
    Timing tcs;
    {   // the single native kernel it also replaces
        struct { int32_t channels; int32_t d_conv; } pc{C, dc};
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_conv_silu.spv", 5, 8);
        tcs = time_kernel(ctx, p, {&bcs, &bx, &bkw, &braw, &bh}, &pc, sizeof(pc), cg, 1, 64, reps, warmups);
        report("native_gdn_conv_silu", shape, tcs, (double) C, 0.0);
    }
    Timing tf;
    {   // the fused path: conv + SiLU + both L2 norms, ONE dispatch
        struct { int32_t channels; int32_t qk_heads; float eps; } pc{C, qk, 1e-6f};
        VkPipeline p = ctx.pipeline(dir + "/fused_gdn_conv_l2.spv", 4, 12);
        tf = time_kernel(ctx, p, {&bcs, &bx, &bkw, &bh}, &pc, sizeof(pc), cg, 1, 64, reps, warmups);
        report("fused_gdn_conv_l2", shape, tf, (double) C, 0.0);
    }
    std::printf("XPAIR gdn_conv_l2 %s | native native_gdn_conv_silu (1 dispatch) | fused fused_gdn_conv_l2 (1) | "
                "fused/native %.3f\n", shape, tf.med / tcs.med);
    {   // the chain the fused path replaces: conv_silu -> l2_norm(q) -> l2_norm(k), THREE dispatches
        struct { int32_t channels; int32_t d_conv; } pcc{C, dc};
        struct { int32_t rows; int32_t cols; float eps; float inv_sqrt_cols; } pcl{2 * k_heads, 128, 1e-6f,
                                                                                  1.0f / std::sqrt(128.0f)};
        VkPipeline pc_ = ctx.pipeline(dir + "/native_gdn_conv_silu.spv", 5, 8);
        VkPipeline pl_ = ctx.pipeline(dir + "/native_gdn_l2_norm.spv", 1, 16);
        const uint32_t lg = (uint32_t) (2 * k_heads);
        Timing tc = time_chain(ctx,
                               {{pc_, {&bcs, &bx, &bkw, &braw, &bh}, &pcc, sizeof(pcc), cg, 1},
                                {pl_, {&bh}, &pcl, sizeof(pcl), lg, 1},
                                {pl_, {&bh}, &pcl, sizeof(pcl), lg, 1}},
                               64, reps, warmups);
        report("native_gdn_conv_silu+2x l2_norm (chain)", shape, tc, (double) C, 0.0);
        std::printf("XPAIR gdn_conv_l2_fused %s | native conv_silu+2x l2_norm (3 dispatches) | fused "
                    "fused_gdn_conv_l2 (1) | fused/chain %.3f\n", shape, tf.med / tc.med);
    }
    ctx.free(bcs); ctx.free(bx); ctx.free(bkw); ctx.free(braw); ctx.free(bh);
}

void bench_fused_gdn_ab_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n = 2560, h_v = 48;                            // n_embd, ssm_v_heads
    auto bf16_of = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) (u >> 16); };
    std::vector<float> x = floats((size_t) n);
    std::vector<uint16_t> wa((size_t) h_v * n), wb((size_t) h_v * n);
    for (size_t i = 0; i < wa.size(); ++i) {
        wa[i] = bf16_of(rndf() * 0.1f);
        wb[i] = bf16_of(rndf() * 0.1f);
    }
    std::vector<float> dt = floats((size_t) h_v), a = floats((size_t) h_v);
    for (auto& v : a) v = -std::fabs(v) - 0.1f;              // ssm_a = -exp(A_log)
    Buf bx = alloc(ctx, (size_t) n * 4), bwa = alloc(ctx, (size_t) h_v * n * 2), bwb = alloc(ctx, (size_t) h_v * n * 2),
        bdt = alloc(ctx, (size_t) h_v * 4), bssm = alloc(ctx, (size_t) h_v * 4), balpha = alloc(ctx, (size_t) h_v * 4),
        bbeta = alloc(ctx, (size_t) h_v * 4), bg = alloc(ctx, (size_t) h_v * 4);
    ctx.write(bx, x.data(), (size_t) n * 4);
    ctx.write(bwa, wa.data(), (size_t) h_v * n * 2);
    ctx.write(bwb, wb.data(), (size_t) h_v * n * 2);
    ctx.write(bdt, dt.data(), (size_t) h_v * 4);
    ctx.write(bssm, a.data(), (size_t) h_v * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "h_v=%d n=%d", h_v, n);
    Timing tm;
    {   // the bf16 mat-vec it also replaces (one of the two)
        struct { int32_t n_in; int32_t n_out; } pc{n, h_v};
        VkPipeline p = ctx.pipeline(dir + "/bf16_mmvf_f32.spv", 3, 8);
        tm = time_kernel(ctx, p, {&bx, &bwa, &balpha}, &pc, sizeof(pc), (uint32_t) h_v, 1, 64, reps, warmups);
        report("bf16_mmvf_f32 (alpha)", shape, tm, (double) h_v, 0.0);
    }
    Timing tf;
    {   // the fused path: BOTH bf16 mat-vecs + BOTH epilogues, ONE dispatch
        struct { int32_t n; int32_t h_v; } pc{n, h_v};
        VkPipeline p = ctx.pipeline(dir + "/fused_gdn_ab.spv", 7, 8);
        tf = time_kernel(ctx, p, {&bx, &bwa, &bwb, &bdt, &bssm, &bg, &bbeta}, &pc, sizeof(pc), (uint32_t) (2 * h_v),
                         1, 64, reps, warmups);
        report("fused_gdn_ab", shape, tf, (double) (2 * h_v), 0.0);
    }
    std::printf("XPAIR gdn_ab %s | bf16 bf16_mmvf_f32 (1 dispatch, one row set) | fused fused_gdn_ab (1, both) | "
                "fused/mmvf %.3f\n", shape, tf.med / tm.med);
    {   // the chain the fused path replaces: mmvf(alpha) -> mmvf(beta) -> beta_gate -> gate, FOUR dispatches
        struct { int32_t n_in; int32_t n_out; } pcm{n, h_v};
        struct { int32_t n; } pcb{h_v};
        struct { int32_t heads; } pcg{h_v};
        VkPipeline pm_ = ctx.pipeline(dir + "/bf16_mmvf_f32.spv", 3, 8);
        VkPipeline pgb = ctx.pipeline(dir + "/native_gdn_beta_gate.spv", 1, 4);
        VkPipeline pgn = ctx.pipeline(dir + "/native_gdn_gate.spv", 4, 4);
        const uint32_t g1 = (uint32_t) h_v, ge = (uint32_t) ((h_v + 255) / 256);
        Timing tc = time_chain(ctx,
                               {{pm_, {&bx, &bwa, &balpha}, &pcm, sizeof(pcm), g1, 1},
                                {pm_, {&bx, &bwb, &bbeta}, &pcm, sizeof(pcm), g1, 1},
                                {pgb, {&bbeta}, &pcb, sizeof(pcb), ge, 1},
                                {pgn, {&balpha, &bdt, &bssm, &bg}, &pcg, sizeof(pcg), ge, 1}},
                               64, reps, warmups);
        report("2x bf16_mmvf + beta_gate + gate (chain)", shape, tc, (double) (2 * h_v), 0.0);
        std::printf("XPAIR gdn_ab_fused %s | native 2x bf16_mmvf+beta_gate+gate (4 dispatches) | fused "
                    "fused_gdn_ab (1) | fused/chain %.3f\n", shape, tf.med / tc.med);
    }
    ctx.free(bx); ctx.free(bwa); ctx.free(bwb); ctx.free(bdt); ctx.free(bssm); ctx.free(balpha); ctx.free(bbeta);
    ctx.free(bg);
}

void bench_fused_gdn_step_norm_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;                   // the native wrapper requires S == 128
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v), z = floats(no),
                        gamma = floats((size_t) S);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    Buf bst = alloc(ctx, nstate * 4), bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4),
        bv = alloc(ctx, no * 4), bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4),
        bo = alloc(ctx, no * 4), bz = alloc(ctx, no * 4), bgm = alloc(ctx, (size_t) S * 4), by = alloc(ctx, no * 4);
    ctx.write(bst, st.data(), nstate * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    ctx.write(bz, z.data(), no * 4);
    ctx.write(bgm, gamma.data(), (size_t) S * 4);
    char shape[64];
    std::snprintf(shape, sizeof shape, "S=128 h_k=16 h_v=48 (3 MiB state)");
    const uint32_t sg = (uint32_t) ((no + 255) / 256);
    const float scale = 1.0f / std::sqrt((float) S);
    Timing ts;
    {   // native_gdn_step (one of the two kernels the fused path replaces)
        struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pc{S, h_k, h_v, scale};
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
        ts = time_kernel(ctx, p, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pc, sizeof(pc), sg, 1, 8, reps, warmups);
        report("native_gdn_step", shape, ts, (double) no, (double) no * 3.0 * (double) S);
    }
    Timing tn;
    {   // native_gdn_out_norm (the other one)
        struct { int32_t heads; int32_t S; float eps; } pc{h_v, S, 1e-6f};
        VkPipeline p = ctx.pipeline(dir + "/native_gdn_out_norm.spv", 4, 12);
        tn = time_kernel(ctx, p, {&bo, &bz, &bgm, &by}, &pc, sizeof(pc), (uint32_t) h_v, 1, 64, reps, warmups);
        report("native_gdn_out_norm", shape, tn, (double) no, 0.0);
    }
    Timing tf;
    {   // the fused path: step + closing norm, ONE dispatch
        struct { int32_t S; int32_t h_k; int32_t h_v; float eps; } pc{S, h_k, h_v, 1e-6f};
        VkPipeline p = ctx.pipeline(dir + "/fused_gdn_step_norm.spv", 9, 16);
        tf = time_kernel(ctx, p, {&bst, &bq, &bk, &bv, &bg, &bb2, &bz, &bgm, &by}, &pc, sizeof(pc), sg, 1, 8, reps,
                         warmups);
        report("fused_gdn_step_norm", shape, tf, (double) no, (double) no * 3.0 * (double) S);
    }
    std::printf("XPAIR gdn_step_norm_step %s | native native_gdn_step (1 dispatch) | fused fused_gdn_step_norm "
                "(1) | fused/native %.3f\n", shape, tf.med / ts.med);
    std::printf("XPAIR gdn_step_norm_out %s | native native_gdn_out_norm (1 dispatch) | fused "
                "fused_gdn_step_norm (1) | fused/native %.3f\n", shape, tf.med / tn.med);
    {   // the chain the fused path replaces: native_gdn_step -> native_gdn_out_norm, TWO dispatches
        struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pcs{S, h_k, h_v, scale};
        struct { int32_t heads; int32_t S; float eps; } pcn{h_v, S, 1e-6f};
        VkPipeline ps_ = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
        VkPipeline pn_ = ctx.pipeline(dir + "/native_gdn_out_norm.spv", 4, 12);
        Timing tc = time_chain(ctx,
                               {{ps_, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pcs, sizeof(pcs), sg, 1},
                                {pn_, {&bo, &bz, &bgm, &by}, &pcn, sizeof(pcn), (uint32_t) h_v, 1}},
                               8, reps, warmups);
        report("native_gdn_step+native_gdn_out_norm (chain)", shape, tc, (double) no, (double) no * 3.0 * (double) S);
        std::printf("XPAIR gdn_step_norm_fused %s | native step+out_norm (2 dispatches) | fused "
                    "fused_gdn_step_norm (1) | fused/chain %.3f\n", shape, tf.med / tc.med);
    }
    ctx.free(bst); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2);
    ctx.free(bo); ctx.free(bz); ctx.free(bgm); ctx.free(by);
}

// =========================================================================================================
// THE RECURRENCE'S BATCH SWEEP.  Every other GDN row in this file is measured at ONE batch size, and a row
// measured at batch 8 cannot say what the SAME chain costs per dispatch at the batch the engine actually
// submits: `kLiveBatchMax` = 128 dispatches per command buffer (`vulkan/src/device/vk_compute.cpp`), i.e. 64
// `native_gdn_step` + `native_gdn_out_norm` pairs.  `time_chain` divides by the batch, so its figure is the
// per-ITERATION (per pair) cost; this sweeps the batch with the identical fixture and the identical
// `encode_dispatch` path, and prints the per-DISPATCH cost beside it.  The claim it can FALSIFY: "a batch-8
// row is representative".  If the per-dispatch cost is flat in the batch, it is; if it falls as the batch
// grows, the small-batch rows under-measure the engine's units and a dispatch-count change is worth more than
// the small-batch ratio suggests.
void bench_gdn_rec_batch_sweep(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;                   // the native wrapper requires S == 128
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v), z = floats(no),
                        gamma = floats((size_t) S);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    Buf bst = alloc(ctx, nstate * 4), bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4),
        bv = alloc(ctx, no * 4), bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4),
        bo = alloc(ctx, no * 4), bz = alloc(ctx, no * 4), bgm = alloc(ctx, (size_t) S * 4), by = alloc(ctx, no * 4);
    ctx.write(bst, st.data(), nstate * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    ctx.write(bz, z.data(), no * 4);
    ctx.write(bgm, gamma.data(), (size_t) S * 4);
    const uint32_t sg = (uint32_t) ((no + 255) / 256);
    const float scale = 1.0f / std::sqrt((float) S);
    struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pcs{S, h_k, h_v, scale};
    struct { int32_t heads; int32_t S; float eps; } pcn{h_v, S, 1e-6f};
    struct { int32_t S; int32_t h_k; int32_t h_v; float eps; } pcf{S, h_k, h_v, 1e-6f};
    VkPipeline ps_ = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
    VkPipeline pn_ = ctx.pipeline(dir + "/native_gdn_out_norm.spv", 4, 12);
    VkPipeline pf_ = ctx.pipeline(dir + "/fused_gdn_step_norm.spv", 9, 16);
    // 64 pairs = 128 dispatches = exactly `kLiveBatchMax`, the engine's own submission unit.
    const int batches[] = {1, 8, 32, 64, 128, 199};
    for (int b : batches) {
        const Timing tc = time_chain(ctx,
                                     {{ps_, {&bst, &bq, &bk, &bv, &bg, &bb2, &bo}, &pcs, sizeof(pcs), sg, 1},
                                      {pn_, {&bo, &bz, &bgm, &by}, &pcn, sizeof(pcn), (uint32_t) h_v, 1}},
                                     b, reps, warmups);
        const Timing tf = time_kernel(ctx, pf_, {&bst, &bq, &bk, &bv, &bg, &bb2, &bz, &bgm, &by}, &pcf,
                                      sizeof(pcf), sg, 1, b, reps, warmups);
        std::printf("SWEEP gdn_rec pair batch=%-4d (%3d dispatches/replay) | step+out_norm %8.4f ms/pair "
                    "(%8.4f ms/dispatch) | fused %8.4f ms/pair (%8.4f ms/dispatch) | fused/chain %.3f\n",
                    b, 2 * b, tc.med, tc.med / 2.0, tf.med, tf.med, tf.med / tc.med);
    }
    ctx.free(bst); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2);
    ctx.free(bo); ctx.free(bz); ctx.free(bgm); ctx.free(by);
}

// =========================================================================================================
// THE STEP KERNEL'S OWN PROBE - parallelism in flight, the engine's batch, and the STATE FOOTPRINT.
// Three questions the port has NOT answered for `native_gdn_step`, each with its own row:
//   (1) SHAPE.  The dispatch is `groups_for(h_v*S)` = 24 workgroups x 256 lanes = 6,144 threads on a card
//       whose device reports 4,096 fp32 lanes (32 Xe2 cores x 128) - so the grid offers under two waves and,
//       at one workgroup per core, leaves 8 of 32 cores idle.  Printed so thread_count / lanes is a number.
//   (2) THE ENGINE'S BATCH.  The hot single-buffer row at batch 1/8/64/128, with the STATE RE-UPLOADED before
//       every row so a state that decays across a long timed region cannot confound the comparison (the
//       port's `gdn_step_pair` native row read 0.1072 ms where the batch sweep read 0.0427 - factor 2.5 -
//       and the one structural difference is how many times that buffer had already run when it was timed).
//   (3) STATE FOOTPRINT.  The engine owns 36 distinct 3 MiB GDN states (one per GDN layer = 108 MiB, over the
//       24 MB L2) and runs ONE dispatch per (token, layer), so a harness that replays a single buffer is
//       L2-hot in a way the engine's never is.  The `cold` rows cycle through 36 states, same kernel, same
//       grid - the direct test of the port's "8.5x engine-vs-bench" gap, and of whether the state must be
//       DEVICE_LOCAL (the engine's arena type) rather than host-visible for the number to be comparable.
// A row here is an IN-STREAM MARGINAL COST (the documented limit of this harness), NOT a bandwidth figure.
// =========================================================================================================
Timing time_cycle(Ctx& ctx, VkPipeline p, const std::vector<Buf>& states, const std::vector<const Buf*>& fixed_,
                  const void* pc, uint32_t pc_bytes, uint32_t gx, uint32_t gy, int batch, int reps, int warmups) {
    std::vector<const Buf*> bufs;
    ctx.record_begin();
    for (int i = 0; i < batch; ++i) {
        bufs.clear();
        bufs.push_back(&states[(size_t) i % states.size()]);
        for (const Buf* b : fixed_) bufs.push_back(b);
        ctx.record_dispatch(p, bufs, pc, pc_bytes, gx, gy);
    }
    ctx.record_end_and_submit();
    for (int w = 0; w < warmups; ++w) ctx.replay_recorded();
    std::vector<double> ms;
    ms.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx.replay_recorded();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) batch);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.reps = reps; t.batch = batch; t.med = ms[ms.size() / 2]; t.lo = ms.front(); t.hi = ms.back();
    return t;
}

void bench_gdn_step_probe(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;                  // the native wrapper requires S == 128
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    const float scale = 1.0f / std::sqrt((float) S);
    const uint32_t sg = (uint32_t) ((no + 255) / 256);
    struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pcn{S, h_k, h_v, scale};
    struct { int32_t S; int32_t h_k; int32_t h_v; } pcl{S, h_k, h_v};
    const char* shape = "S=128 h_k=16 h_v=48 (3 MiB state)";

    std::printf("PROBE gdn_step shape | %s | grid = %u workgroups x 256 lanes = %u threads | card = 32 Xe2 cores "
                "x 128 = 4096 fp32 lanes (B70/BMG-G31)\n",
                shape, sg, sg * 256u);

    Buf bst = alloc(ctx, nstate * 4), bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4),
        bv = alloc(ctx, no * 4), bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4),
        bo = alloc(ctx, no * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    VkPipeline p = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
    const std::vector<const Buf*> fix = {&bq, &bk, &bv, &bg, &bb2, &bo};

    // (2) the hot single-buffer sweep, state RE-UPLOADED before each row (decay cannot confound it).
    double hot128 = 0.0;
    const int hot_batches[] = {1, 8, 64, 128};
    for (int b : hot_batches) {
        ctx.write(bst, st.data(), nstate * 4);
        std::vector<const Buf*> bf = {&bst, &bq, &bk, &bv, &bg, &bb2, &bo};
        Timing t = time_kernel(ctx, p, bf, &pcn, sizeof(pcn), sg, 1, b, reps, warmups);
        report("native_gdn_step hot", shape, t, (double) no, (double) no * 3.0 * (double) S);
        if (b == 128) hot128 = t.med;
    }
    // the ORDER confound, measured: the SAME batch-8 row run twice more in the same process.  If they differ,
    // a row's position in the process decides its value.
    for (int r2 = 0; r2 < 2; ++r2) {
        ctx.write(bst, st.data(), nstate * 4);
        std::vector<const Buf*> bf = {&bst, &bq, &bk, &bv, &bg, &bb2, &bo};
        Timing t = time_kernel(ctx, p, bf, &pcn, sizeof(pcn), sg, 1, 8, reps, warmups);
        report("native_gdn_step hot8_repeat", shape, t, (double) no, (double) no * 3.0 * (double) S);
    }

    // (3) the STATE FOOTPRINT: 36 distinct 3 MiB states (108 MiB, over the 24 MB L2), cycled at the engine's
    // batch of 128 dispatches - the same kernel and grid the hot rows above ran.
    const int nlay = 36;
    std::vector<Buf> cold, colddev;
    for (int i = 0; i < nlay; ++i) {
        Buf s = alloc(ctx, nstate * 4);
        std::vector<float> si = st;                         // a DISTINCT state per layer, so it is a real 108 MiB
        for (size_t z = (size_t) i; z < si.size(); z += (size_t) nlay) si[z] += (float) i;
        ctx.write(s, si.data(), nstate * 4);
        cold.push_back(s);
    }
    Timing tcold = time_cycle(ctx, p, cold, fix, &pcn, sizeof(pcn), sg, 1, 128, reps, warmups);
    report("native_gdn_step cold36host", "36x3MiB host-visible, batch 128", tcold, (double) no,
           (double) no * 3.0 * (double) S);

    // the same footprint in DEVICE_LOCAL memory (the engine's arena type is not host-visible)
    for (int i = 0; i < nlay; ++i) {
        Buf s = ctx.alloc_device(nstate * 4);
        ctx.write(s, st.data(), nstate * 4);
        colddev.push_back(s);
    }
    Timing tcoldd = time_cycle(ctx, p, colddev, fix, &pcn, sizeof(pcn), sg, 1, 128, reps, warmups);
    report("native_gdn_step cold36dev", "36x3MiB device-local, batch 128", tcoldd, (double) no,
           (double) no * 3.0 * (double) S);

    // the LEGACY kernel through the identical instrument (hot 128 vs cold 36 host), so the pair is ranked
    VkPipeline pl = ctx.pipeline(dir + "/gdn_step.spv", 7, 12);
    ctx.write(bst, st.data(), nstate * 4);
    std::vector<const Buf*> bl = {&bst, &bq, &bk, &bv, &bg, &bb2, &bo};
    Timing tlh = time_kernel(ctx, pl, bl, &pcl, sizeof(pcl), sg, 1, 128, reps, warmups);
    report("gdn_step legacy hot", shape, tlh, (double) no, (double) no * 3.0 * (double) S);
    Timing tlc = time_cycle(ctx, pl, cold, fix, &pcl, sizeof(pcl), sg, 1, 128, reps, warmups);
    report("gdn_step legacy cold36host", "36x3MiB host-visible, batch 128", tlc, (double) no,
           (double) no * 3.0 * (double) S);
    std::printf("XPAIR gdn_step_probe | hot128 native %.4f vs cold36host %.4f = %.3f | legacy hot128 %.4f vs "
                "cold36host %.4f = %.3f | cold36dev/native-hot128 %.3f\n",
                hot128, tcold.med, tcold.med / (hot128 > 0 ? hot128 : 1e-9), tlh.med, tlc.med,
                tlc.med / (tlh.med > 0 ? tlh.med : 1e-9), tcoldd.med / (hot128 > 0 ? hot128 : 1e-9));

    ctx.free(bst); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2); ctx.free(bo);
    for (Buf& s : cold) ctx.free(s);
    for (Buf& s : colddev) ctx.free(s);
}

// =========================================================================================================
// THE STEP KERNEL'S UNROLL A/B - and the bit-exactness check that has to pass before any timing matters.
// The unroll (`native_gdn_step.comp`, KU) is a MEMORY-LEVEL-PARALLELISM change and must not touch a single
// float: the same single accumulator, the same ascending `i` order.  This arm DISPATCHES the KU=1 (rolled)
// .spv, the SHIPPED KU and a KU=8 variant onto IDENTICAL inputs and compares STATE and O BITWISE, naming the
// first mismatching index when they differ - so "the same summation order" is a measurement, not a claim in a
// comment.  Then it times all three at batch 128 hot and through the 36-state footprint.  All three .spv are
// built from the SAME source by `run_bench.sh` (`-DKU=1/8/<shipped>`), so the arm is reproducible.
// =========================================================================================================
static constexpr int KU_SHIPPED = 16;
void bench_gdn_step_unroll(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int S = 128, h_k = 16, h_v = 48;
    const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
    std::vector<float> st = floats(nstate), q = floats((size_t) h_k * S), k = floats((size_t) h_k * S),
                        v = floats(no), gate = floats((size_t) h_v), beta = floats((size_t) h_v);
    for (auto& g : gate) g = -std::fabs(g) - 0.5f;
    const float scale = 1.0f / std::sqrt((float) S);
    const uint32_t sg = (uint32_t) ((no + 255) / 256);
    struct { int32_t S; int32_t h_k; int32_t h_v; float scale; } pc{S, h_k, h_v, scale};
    Buf bq = alloc(ctx, (size_t) h_k * S * 4), bk = alloc(ctx, (size_t) h_k * S * 4), bv = alloc(ctx, no * 4),
        bg = alloc(ctx, (size_t) h_v * 4), bb2 = alloc(ctx, (size_t) h_v * 4);
    ctx.write(bq, q.data(), q.size() * 4);
    ctx.write(bk, k.data(), k.size() * 4);
    ctx.write(bv, v.data(), no * 4);
    ctx.write(bg, gate.data(), (size_t) h_v * 4);
    ctx.write(bb2, beta.data(), (size_t) h_v * 4);
    const std::vector<const Buf*> fix = {&bq, &bk, &bv, &bg, &bb2};

    VkPipeline pbase = ctx.pipeline(dir + "/native_gdn_step_u1.spv", 7, 16);
    VkPipeline pun = ctx.pipeline(dir + "/native_gdn_step.spv", 7, 16);
    VkPipeline pu16 = ctx.pipeline(dir + "/native_gdn_step_u8.spv", 7, 16);

    // ---- BITWISE: baseline vs unrolled on identical inputs, ONE dispatch each ----
    Buf stA = alloc(ctx, nstate * 4), stB = alloc(ctx, nstate * 4), stC = alloc(ctx, nstate * 4),
        oA = alloc(ctx, no * 4), oB = alloc(ctx, no * 4), oC = alloc(ctx, no * 4);
    ctx.write(stA, st.data(), nstate * 4);
    ctx.write(stB, st.data(), nstate * 4);
    ctx.write(stC, st.data(), nstate * 4);
    auto one = [&](VkPipeline p, Buf& s, Buf& o) {
        ctx.record_begin();
        std::vector<const Buf*> b = {&s, &bq, &bk, &bv, &bg, &bb2, &o};
        ctx.record_dispatch(p, b, &pc, sizeof(pc), sg, 1);
        ctx.record_end_and_submit();
    };
    one(pbase, stA, oA);
    one(pun, stB, oB);
    one(pu16, stC, oC);
    std::vector<float> rsA(nstate), rsB(nstate), rsC(nstate), roA(no), roB(no), roC(no);
    ctx.read(stA, rsA.data(), nstate * 4);
    ctx.read(stB, rsB.data(), nstate * 4);
    ctx.read(stC, rsC.data(), nstate * 4);
    ctx.read(oA, roA.data(), no * 4);
    ctx.read(oB, roB.data(), no * 4);
    ctx.read(oC, roC.data(), no * 4);
    auto cmp = [](const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
        size_t bad = 0;
        for (size_t z = 0; z < a.size(); ++z)
            if (a[z] != b[z]) { if (bad == 0) std::printf("   BITEXACT first mismatch %s: idx %zu base %.9g new %.9g\n", tag, z, a[z], b[z]); ++bad; }
        std::printf("   BITEXACT %s: %zu/%zu differ%s\n", tag, bad, a.size(), bad == 0 ? "" : "  <-- NOT BIT-EXACT");
        return bad == 0;
    };
    bool ok8 = cmp("state shipped", rsA, rsB) & cmp("o shipped", roA, roB);
    bool ok16 = cmp("state u8", rsA, rsC) & cmp("o u8", roA, roC);
    std::printf("PROBE gdn_step_unroll | the ROLLED order (KU=1) vs the SHIPPED KU=%d bit-exact: %s | vs KU=8: %s | "
                "shaders u1(rolled) native_gdn_step_u1.spv shipped native_gdn_step.spv u8 native_gdn_step_u8.spv\n",
                KU_SHIPPED, ok8 ? "YES" : "NO", ok16 ? "YES" : "NO");
    ctx.free(stA); ctx.free(stB); ctx.free(stC); ctx.free(oA); ctx.free(oB); ctx.free(oC);

    // ---- TIMING: hot batch 128 and the 36-state footprint, each through the identical instrument ----
    Buf bst = alloc(ctx, nstate * 4), bo = alloc(ctx, no * 4);
    ctx.write(bst, st.data(), nstate * 4);
    const int nlay = 36;
    std::vector<Buf> cold;
    for (int i = 0; i < nlay; ++i) {
        Buf s = alloc(ctx, nstate * 4);
        std::vector<float> si = st;
        for (size_t z = (size_t) i; z < si.size(); z += (size_t) nlay) si[z] += (float) i;
        ctx.write(s, si.data(), nstate * 4);
        cold.push_back(s);
    }
    struct V { const char* name; VkPipeline p; };
    const V vs[] = {{"u1(rolled)", pbase}, {"shipped", pun}, {"u8", pu16}};
    for (const V& v : vs) {
        ctx.write(bst, st.data(), nstate * 4);
        std::vector<const Buf*> b = {&bst, &bq, &bk, &bv, &bg, &bb2, &bo};
        Timing th = time_kernel(ctx, v.p, b, &pc, sizeof(pc), sg, 1, 128, reps, warmups);
        std::string tag = std::string("hot128 ") + v.name;
        report(("native_gdn_step " + std::string(v.name)).c_str(), tag, th, (double) no, (double) no * 3.0 * S);
        Timing tc = time_cycle(ctx, v.p, cold, fix, &pc, sizeof(pc), sg, 1, 128, reps, warmups);
        tag = std::string("cold36host ") + v.name;
        report(("native_gdn_step " + std::string(v.name)).c_str(), tag, tc, (double) no, (double) no * 3.0 * S);
    }
    ctx.free(bst); ctx.free(bo); ctx.free(bq); ctx.free(bk); ctx.free(bv); ctx.free(bg); ctx.free(bb2);
    for (Buf& s : cold) ctx.free(s);
}

// =========================================================================================================
// THE BF16-PROJECTION PAIR (`bf16_gemv` / `bf16_gemv_split`, ONE shared shader) - the DEFAULT side of the
// `native_bf16_projections` setting, ported so the setting cannot route the engine at an unported symbol.  There
// is NO legacy sibling to compare against (this IS the non-native branch), so the pair is measured against the
// PORTED native sibling `bf16_gemv_fp32_mmvf` (`bf16_mmvf_f32`), which the setting selects when it is ON: same
// workgroup-per-row decomposition, the only difference being the activation's precision (bf16 vs f32).  XPAIR
// convention is `<row>/<baseline>`; < 1.0 means the first row is faster, and a same-shape drop-in is EXPECTED to
// wash.  The CUDA's one-thread-per-row naive decomposition was ALSO measured and is NOT shipped: at the engine's
// shapes it is 14-18x slower on the GPUs (Arc 0.073x, Ryzen iGPU 0.103x of the workgroup form at n_out=512) and
// the CUDA does not take it above n_out=64 anyway.
// Forward declaration: `time_iters` (the per-dispatch instrument) is defined further down the file.
template <class RecordIter>
Timing time_iters(Ctx& ctx, int iters, int disp_per_iter, int reps, int warmups, RecordIter rec);

// =========================================================================================================
// (A) THE VERIFY WINDOW'S GROUPED EXPERT GEMV (`native_gu_any` / `native_down_any`), AT THE ENGINE'S SHAPE.
//
// The mixture model prices the window's 2,304 `native_gu_any`/`native_down_any` dispatches at the
// `iq1m_mmvq` n_out=1280 ncols=1 row (111 us).  That is a DIFFERENT shader, a DIFFERENT grid (`n_out`
// workgroups vs `2*n_ff x gy`) and a DIFFERENT memory type - so this arm prices the kernel that actually
// fills the window, and moves ONE property at a time so the lever is named rather than assumed:
//
//   gu_1win      ONE dispatch, grid (2*n_ff, 30) - all 30 groups in one window (the CUDA's shape)
//   gu_port8     EIGHT dispatches, one per window, the 30 groups spread over the windows (the port's shape:
//                a 4 GiB storage-buffer index limit forces one launch per arena window, so the launch count
//                is multiplied by nwin and ~7/8 of each launch has no matching group)
//   gu_1grp30    30 entries on ONE expert, ONE dispatch (the "batch tokens per weight read" hypothesis)
//   dn_1win / dn_port8   the same two shapes for the down projection
//
// The two mechanisms the task names (tokens per dispatch, and dispatch count) therefore have separate rows.
// =========================================================================================================
void bench_native_grouped_engine(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n_embd = 2560, n_ff = 1280;
    const int n_tok = 3, K = 10, cap = n_tok * K;              // 30 entries - the engine's --spec 2 window
    const int ty_gu = 18;                                      // IQ3_XXS (layer 0's gate/up): 98 B / 256
    const int ty_dn = 20;                                      // IQ4_NL (layer 0's down): 18 B / 32
    const int nwin = 8;                                        // the 27 GiB arena in 4 GiB-64 MiB windows

    const int nb_gu = n_embd / 256, gu_row = nb_gu * 98, gu_blob = 2 * n_ff * gu_row;
    std::vector<uint8_t> wgu((size_t) cap * (size_t) gu_blob);
    for (size_t i = 0; i < wgu.size(); ++i) wgu[i] = (uint8_t) (i * 29 + 7);
    std::vector<uint32_t> off_gu((size_t) cap);
    for (int g = 0; g < cap; ++g) off_gu[(size_t) g] = (uint32_t) ((size_t) g * (size_t) gu_blob);
    std::vector<uint8_t> agu((size_t) cap * (size_t) (n_embd / 32) * 36u);
    for (size_t i = 0; i < agu.size(); ++i) agu[i] = (uint8_t) (i * 11 + 3);

    Buf b_w = ctx.alloc_device(wgu.size()), b_a = ctx.alloc_device(agu.size()), b_off = ctx.alloc((size_t) cap * 4);
    Buf b_win = ctx.alloc((size_t) cap * 4), b_s = ctx.alloc(((size_t) cap + 1) * 4), b_ng = ctx.alloc(4);
    Buf b_e = ctx.alloc((size_t) cap * 4), b_g1 = ctx.alloc(sizeof(strata::vkport::kIq2sGrid));
    Buf b_g2 = ctx.alloc(sizeof(strata::vkport::kIq3xxsGrid)), b_g3 = ctx.alloc(sizeof(strata::vkport::kIq3sGrid));
    Buf b_og = ctx.alloc_device((size_t) cap * n_ff * 4u + 256u), b_ou = ctx.alloc_device((size_t) cap * n_ff * 4u + 256u);
    ctx.write(b_w, wgu.data(), wgu.size()); ctx.write(b_a, agu.data(), agu.size());
    ctx.write(b_off, off_gu.data(), off_gu.size() * 4);
    ctx.write(b_g1, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    ctx.write(b_g2, strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid));
    ctx.write(b_g3, strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid));
    VkPipeline pgu = ctx.pipeline(dir + "/native_gu_any.spv", 12, 16);
    const std::string gshape = "n_embd=2560 n_ff=1280 ty=IQ3_XXS entries=30 device-local";

    auto set_layout = [&](int mode) {                          // 0: 30 groups x1 win0 | 1: 30 groups x1 spread | 2: 1 group x30
        std::vector<int32_t> s((size_t) cap + 1), e((size_t) cap), ng(1, cap);
        std::vector<uint32_t> win((size_t) cap, 0u);
        if (mode == 2) {
            s[0] = 0; s[1] = cap;
            ng[0] = 1;
        } else {
            for (int g = 0; g <= cap; ++g) s[(size_t) g] = g;
            if (mode == 1) for (int g = 0; g < cap; ++g) win[(size_t) g] = (uint32_t) (g % nwin);
        }
        for (int g = 0; g < cap; ++g) e[(size_t) g] = g % n_tok;
        ctx.write(b_s, s.data(), s.size() * 4); ctx.write(b_e, e.data(), e.size() * 4);
        ctx.write(b_win, win.data(), win.size() * 4); ctx.write(b_ng, ng.data(), 4);
    };
    auto gu_disp = [&](uint32_t ngy, int win_id) {
        struct { int n_embd, n_ff, ty, win_id; } pc{n_embd, n_ff, ty_gu, win_id};
        ctx.record_dispatch(pgu, {&b_w, &b_a, &b_g1, &b_g2, &b_g3, &b_off, &b_win, &b_s, &b_ng, &b_e, &b_og, &b_ou},
                            &pc, sizeof(pc), (uint32_t) (2 * n_ff), ngy);
    };

    set_layout(0);
    report("gu_1win", gshape + " | 1 dispatch, all 30 groups in one window",
           time_iters(ctx, 8, 1, reps, warmups, [&](int) { gu_disp((uint32_t) cap, 0); }), (double) cap * n_ff, 0.0);
    set_layout(1);
    report("gu_port8", gshape + " | 8 dispatches (one per arena window), groups spread over the windows",
           time_iters(ctx, 1, nwin, reps, warmups, [&](int) {
               for (int w = 0; w < nwin; ++w) gu_disp((uint32_t) cap, w);
           }), (double) cap * n_ff, 0.0);
    set_layout(2);
    report("gu_1grp30", gshape + " | 1 dispatch, 30 entries on ONE expert (tokens batched)",
           time_iters(ctx, 8, 1, reps, warmups, [&](int) { gu_disp((uint32_t) cap, 0); }), (double) cap * n_ff, 0.0);
    // THE ENGINE'S REAL CASE: a layer's 30 groups are CONTIGUOUS in the pack, so they live in ONE window -
    // the port still issues nwin launches, 1 doing all the work and nwin-1 with no matching group at all.
    set_layout(0);
    report("gu_engine1", gshape + " | 8 dispatches, all 30 groups in ONE window (the engine's real case)",
           time_iters(ctx, 1, nwin, reps, warmups, [&](int) {
               for (int w = 0; w < nwin; ++w) gu_disp((uint32_t) cap, w);
           }), (double) cap * n_ff, 0.0);
    // THE COST OF AN EMPTY LAUNCH ON ITS OWN: the same grid, no group in this dispatch's window.
    set_layout(1);
    report("gu_empty", gshape + " | 1 dispatch, NO group in this window (an empty launch of the same grid)",
           time_iters(ctx, 8, 1, reps, warmups, [&](int) { gu_disp((uint32_t) cap, 99); }), (double) cap * n_ff, 0.0);
    // THE Y-GRID SWEEP.  `gy` is the number of groups a launch covers side by side (the port's
    // `grid_groups`); the groups are strided by it, so the RESULTS do not depend on it - only how the work
    // is spread.  A SMALL gy makes each launch fewer workgroups, which is what the nwin-1 empty launches
    // (the engine's real case) pay for; the full launch may pay for it in parallelism.  Per-call cost =
    // full(gy) + (nwin-1) * empty(gy).
    for (uint32_t gy : {1u, 2u, 4u, 8u, 16u, 30u}) {
        char sh[96];
        set_layout(0);
        std::snprintf(sh, sizeof sh, "gy=%u | all 30 groups in one window (the work)", gy);
        Timing tf = time_iters(ctx, 8, 1, reps, warmups, [&](int) { gu_disp(gy, 0); });
        report("gu_gysweep", sh, tf, (double) cap * n_ff, 0.0);
        set_layout(1);
        std::snprintf(sh, sizeof sh, "gy=%u | no group in this window (an empty launch)", gy);
        Timing te = time_iters(ctx, 8, 1, reps, warmups, [&](int) { gu_disp(gy, 99); });
        report("gu_gysweep_empty", sh, te, (double) cap * n_ff, 0.0);
        std::printf("      gy=%u: full %.4f ms + %d empty x %.4f = %.4f ms per call (vs 1 full + 0 empty = %.4f)\n",
                    gy, tf.med, nwin - 1, te.med, tf.med + (nwin - 1) * te.med, tf.med);
    }

    const int d_row = (n_ff / 32) * 18, dn_blob = d_row * n_embd;
    std::vector<uint8_t> wdn((size_t) cap * (size_t) dn_blob);
    for (size_t i = 0; i < wdn.size(); ++i) wdn[i] = (uint8_t) (i * 37 + 11);
    std::vector<uint32_t> off_dn((size_t) cap);
    for (int g = 0; g < cap; ++g) off_dn[(size_t) g] = (uint32_t) ((size_t) g * (size_t) dn_blob);
    std::vector<uint8_t> hq((size_t) cap * (size_t) (n_ff / 32) * 36u);
    for (size_t i = 0; i < hq.size(); ++i) hq[i] = (uint8_t) (i * 13 + 5);
    Buf b_wd = ctx.alloc_device(wdn.size()), b_hq = ctx.alloc_device(hq.size()), b_offd = ctx.alloc((size_t) cap * 4);
    Buf b_wind = ctx.alloc((size_t) cap * 4), b_sd = ctx.alloc(((size_t) cap + 1) * 4), b_ngd = ctx.alloc(4);
    Buf b_ed = ctx.alloc((size_t) cap * 4), b_yd = ctx.alloc_device((size_t) cap * n_embd * 4u + 256u);
    ctx.write(b_wd, wdn.data(), wdn.size()); ctx.write(b_hq, hq.data(), hq.size());
    ctx.write(b_offd, off_dn.data(), off_dn.size() * 4);
    VkPipeline pdn = ctx.pipeline(dir + "/native_down_any.spv", 8, 24);
    const std::string dshape = "n_embd=2560 n_ff=1280 ty=IQ4_NL entries=30 device-local";
    std::vector<int32_t> sd((size_t) cap + 1), ed((size_t) cap);
    for (int g = 0; g <= cap; ++g) sd[(size_t) g] = g;
    for (int g = 0; g < cap; ++g) ed[(size_t) g] = g % n_tok;
    ctx.write(b_sd, sd.data(), sd.size() * 4); ctx.write(b_ed, ed.data(), ed.size() * 4);
    ctx.write(b_ngd, &cap, 4);
    auto dn_disp = [&](uint32_t ngy, int win_id) {
        struct { int n_embd, n_ff, ty, win_id, d_row, down_off; } pc{n_embd, n_ff, ty_dn, win_id, d_row, 0};
        ctx.record_dispatch(pdn, {&b_wd, &b_hq, &b_offd, &b_wind, &b_sd, &b_ngd, &b_ed, &b_yd},
                            &pc, sizeof(pc), (uint32_t) n_embd, ngy);
    };
    std::vector<uint32_t> win0((size_t) cap, 0u), winsp((size_t) cap);
    for (int g = 0; g < cap; ++g) winsp[(size_t) g] = (uint32_t) (g % nwin);
    ctx.write(b_wind, win0.data(), win0.size() * 4);
    report("dn_1win", dshape + " | 1 dispatch, all 30 groups in one window",
           time_iters(ctx, 8, 1, reps, warmups, [&](int) { dn_disp((uint32_t) cap, 0); }), (double) cap * n_embd, 0.0);
    ctx.write(b_wind, winsp.data(), winsp.size() * 4);
    report("dn_port8", dshape + " | 8 dispatches (one per arena window), groups spread over the windows",
           time_iters(ctx, 1, nwin, reps, warmups, [&](int) {
               for (int w = 0; w < nwin; ++w) dn_disp((uint32_t) cap, w);
           }), (double) cap * n_embd, 0.0);
    ctx.write(b_wind, win0.data(), win0.size() * 4);
    report("dn_engine1", dshape + " | 8 dispatches, all 30 groups in ONE window (the engine's real case)",
           time_iters(ctx, 1, nwin, reps, warmups, [&](int) {
               for (int w = 0; w < nwin; ++w) dn_disp((uint32_t) cap, w);
           }), (double) cap * n_embd, 0.0);
    // THE EMPTY DOWN LAUNCH - the same grid with no group in this dispatch's window (the gu side's twin).
    ctx.write(b_wind, winsp.data(), winsp.size() * 4);
    report("dn_empty", dshape + " | 1 dispatch, NO group in this window (an empty launch of the same grid)",
           time_iters(ctx, 8, 1, reps, warmups, [&](int) { dn_disp((uint32_t) cap, 99); }), (double) cap * n_embd, 0.0);
    for (uint32_t gy : {1u, 2u, 4u, 8u, 16u, 30u}) {
        char sh[96];
        ctx.write(b_wind, win0.data(), win0.size() * 4);
        std::snprintf(sh, sizeof sh, "gy=%u | all 30 groups in one window (the work)", gy);
        Timing tf = time_iters(ctx, 8, 1, reps, warmups, [&](int) { dn_disp(gy, 0); });
        report("dn_gysweep", sh, tf, (double) cap * n_embd, 0.0);
        ctx.write(b_wind, winsp.data(), winsp.size() * 4);
        std::snprintf(sh, sizeof sh, "gy=%u | no group in this window (an empty launch)", gy);
        Timing te = time_iters(ctx, 8, 1, reps, warmups, [&](int) { dn_disp(gy, 99); });
        report("dn_empty_gy", sh, te, (double) cap * n_embd, 0.0);
        std::printf("      dn gy=%u: work %.4f + %d empty x %.4f = %.4f ms per call\n", gy, tf.med, nwin - 1,
                    te.med, tf.med + (nwin - 1) * te.med);
    }

    ctx.free(b_w); ctx.free(b_a); ctx.free(b_off); ctx.free(b_win); ctx.free(b_s); ctx.free(b_ng);
    ctx.free(b_e); ctx.free(b_g1); ctx.free(b_g2); ctx.free(b_g3); ctx.free(b_og); ctx.free(b_ou);
    ctx.free(b_wd); ctx.free(b_hq); ctx.free(b_offd); ctx.free(b_wind); ctx.free(b_sd); ctx.free(b_ngd);
    ctx.free(b_ed); ctx.free(b_yd);
}

// =========================================================================================================
// (B) THE `fused_gr_*` GROUP, PRICED - the mixture model's entire 112.7 us residual.
//
// The RECORDED-arm histogram carries 576 dispatches of EACH of `fused_gr_rs` / `_down` / `_mix` / `_inject`
// (2,304 = 21% of the window's count) and the mixture model has NO row for them, so the model closes only by
// making them carry 112.7 us each.  `fused_gr_read_multi` (vulkan/src/kernels/fused_gr_vk.cpp) is a LOOP over
// tokens and each token is FOUR dispatches at the ARTIFACT'S geometry (N=2560 HC=4 LR=320), so this arm
// measures the four kernels singly AND the chain the engine actually issues, at the engine's batch.
// =========================================================================================================
void bench_gr_pricing(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int N = 2560, HC = 4, LR = 320;                       // fused_gr.cu:20-23, the artifact's geometry
    const int64_t hc_dim = (int64_t) HC * N;
    std::vector<float> r = floats((size_t) hc_dim), z = floats((size_t) hc_dim);
    std::vector<float> nrm = floats((size_t) hc_dim), bo = floats((size_t) N), inj = floats((size_t) HC);
    std::vector<float> lof = floats((size_t) LR), rs = floats((size_t) HC), mix = floats((size_t) N);
    Buf b_r = ctx.alloc_device((size_t) hc_dim * 4), b_o = ctx.alloc_device((size_t) hc_dim * 4);
    Buf b_n = ctx.alloc_device((size_t) hc_dim * 4);
    Buf b_d = ctx.alloc_device((size_t) LR * (size_t) (hc_dim / 2) * 4);     // w_down: bf16 pairs
    Buf b_u = ctx.alloc_device((size_t) hc_dim * (size_t) (LR / 2) * 4);     // w_up:   bf16 pairs
    Buf b_l = ctx.alloc_device((size_t) LR * 4), b_s = ctx.alloc_device((size_t) HC * 4);
    Buf b_m = ctx.alloc_device((size_t) N * 4), b_b = ctx.alloc_device((size_t) N * 4);
    Buf b_i = ctx.alloc_device((size_t) HC * 4), b_w = ctx.alloc_device((size_t) HC * (size_t) (hc_dim / 2) * 4);
    Buf b_j = ctx.alloc_device((size_t) HC * 4);
    ctx.write(b_r, r.data(), r.size() * 4); ctx.write(b_o, z.data(), z.size() * 4);
    ctx.write(b_n, nrm.data(), nrm.size() * 4); ctx.write(b_b, bo.data(), bo.size() * 4);
    ctx.write(b_i, inj.data(), inj.size() * 4); ctx.write(b_l, lof.data(), lof.size() * 4);
    ctx.write(b_s, rs.data(), rs.size() * 4); ctx.write(b_m, mix.data(), mix.size() * 4);
    ctx.write(b_j, inj.data(), inj.size() * 4);
    {
        std::vector<uint8_t> wd((size_t) b_d.bytes, 0), wu((size_t) b_u.bytes, 0), wi((size_t) b_w.bytes, 0);
        for (size_t i = 0; i < wd.size(); ++i) wd[i] = (uint8_t) (i * 7 + 1);
        for (size_t i = 0; i < wu.size(); ++i) wu[i] = (uint8_t) (i * 7 + 3);
        for (size_t i = 0; i < wi.size(); ++i) wi[i] = (uint8_t) (i * 7 + 5);
        ctx.write(b_d, wd.data(), wd.size()); ctx.write(b_u, wu.data(), wu.size()); ctx.write(b_w, wi.data(), wi.size());
    }
    const std::string shape = "N=2560 HC=4 LR=320 (the artifact's geometry)";
    const uint32_t g_rs = HC, g_dn = LR, g_mx = N, g_in = HC;
    VkPipeline p_rs = ctx.pipeline(dir + "/fused_gr_rs.spv", 5, 20);
    VkPipeline p_dn = ctx.pipeline(dir + "/fused_gr_down.spv", 5, 12);
    VkPipeline p_mx = ctx.pipeline(dir + "/fused_gr_mix.spv", 6, 12);
    VkPipeline p_in = ctx.pipeline(dir + "/fused_gr_inject.spv", 5, 8);
    struct { int n_embd, hc; float eps; int apply; } pcrs{N, HC, 1e-6f, 1};
    struct { int n_embd, hc, hc_lr; } pcdn{N, HC, LR}, pcmx{N, HC, LR};
    struct { int n_embd, hc; } pcin{N, HC};
    auto one = [&](int) {
        ctx.record_dispatch(p_rs, {&b_r, &b_b, &b_i, &b_o, &b_s}, &pcrs, sizeof(pcrs), g_rs, 1);
        ctx.record_dispatch(p_dn, {&b_o, &b_n, &b_s, &b_d, &b_l}, &pcdn, sizeof(pcdn), g_dn, 1);
        ctx.record_dispatch(p_mx, {&b_o, &b_n, &b_s, &b_l, &b_u, &b_m}, &pcmx, sizeof(pcmx), g_mx, 1);
        ctx.record_dispatch(p_in, {&b_o, &b_n, &b_s, &b_w, &b_j}, &pcin, sizeof(pcin), g_in, 1);
    };
    report("fused_gr_rs", shape, time_kernel(ctx, p_rs, {&b_r, &b_b, &b_i, &b_o, &b_s}, &pcrs, sizeof(pcrs), g_rs, 1, 8,
                                             reps, warmups), (double) HC, 0.0);
    report("fused_gr_down", shape, time_kernel(ctx, p_dn, {&b_o, &b_n, &b_s, &b_d, &b_l}, &pcdn, sizeof(pcdn), g_dn, 1,
                                               8, reps, warmups), (double) LR, 0.0);
    report("fused_gr_mix", shape, time_kernel(ctx, p_mx, {&b_o, &b_n, &b_s, &b_l, &b_u, &b_m}, &pcmx, sizeof(pcmx), g_mx,
                                              1, 8, reps, warmups), (double) N, 0.0);
    report("fused_gr_inject", shape, time_kernel(ctx, p_in, {&b_o, &b_n, &b_s, &b_w, &b_j}, &pcin, sizeof(pcin), g_in, 1,
                                                 8, reps, warmups), (double) HC, 0.0);
    for (int nt : {3, 8}) {
        char sh[96];
        std::snprintf(sh, sizeof sh, "%s | fused_gr_read_multi T=%d (%d dispatches)", shape.c_str(), nt, 4 * nt);
        Timing t = time_iters(ctx, nt, 4, reps, warmups, one);
        report("fused_gr_multi", sh, t, (double) N, 0.0);
        std::printf("      fused_gr_multi T=%d: %.4f ms/token (%d dispatches), %.4f ms/dispatch\n", nt, t.med * 4.0,
                    4 * nt, t.med);
    }
    ctx.free(b_r); ctx.free(b_o); ctx.free(b_n); ctx.free(b_d); ctx.free(b_u); ctx.free(b_l); ctx.free(b_s);
    ctx.free(b_m); ctx.free(b_b); ctx.free(b_i); ctx.free(b_w); ctx.free(b_j);
}

// =========================================================================================================
void bench_bf16_gemv_pair(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int shapes[][2] = {{2560, 512}, {2560, 48}};   // the router/indexer projection; the GDN alpha/beta projection
    for (const auto& s : shapes) {
        const int n_in = s[0], n_out = s[1];
        const size_t xw = (size_t) n_in / 2;             // bf16 activation, 2 per 32-bit word
        const size_t ww = (size_t) n_out * n_in / 2;     // bf16 weights, 2 per 32-bit word
        std::vector<uint32_t> xbuf(xw, 0), wbuf(ww, 0);
        for (auto& w : xbuf) w = next_rand();
        for (auto& w : wbuf) w = next_rand();
        std::vector<float> xf = floats((size_t) n_in);
        Buf bx = alloc(ctx, xw * 4), bx32 = alloc(ctx, (size_t) n_in * 4), bw = alloc(ctx, ww * 4),
            by = alloc(ctx, (size_t) n_out * 4 + 64);
        ctx.write(bx, xbuf.data(), xw * 4);
        ctx.write(bx32, xf.data(), (size_t) n_in * 4);
        ctx.write(bw, wbuf.data(), ww * 4);
        struct { int32_t n_in; int32_t n_out; } pc{n_in, n_out};
        char shape[64];
        std::snprintf(shape, sizeof shape, "n_in=%d n_out=%d", n_in, n_out);
        Timing tv, tm;
        {
            VkPipeline p = ctx.pipeline(dir + "/bf16_gemv.spv", 3, (int) sizeof(pc));
            tv = time_kernel(ctx, p, {&bx, &bw, &by}, &pc, sizeof(pc), (uint32_t) n_out, 1, 64, reps, warmups);
            report("bf16_gemv_wg_row", shape, tv, (double) n_out, (double) n_out * n_in);
        }
        {
            VkPipeline p = ctx.pipeline(dir + "/bf16_mmvf_f32.spv", 3, (int) sizeof(pc));
            tm = time_kernel(ctx, p, {&bx32, &bw, &by}, &pc, sizeof(pc), (uint32_t) n_out, 1, 64, reps, warmups);
            report("bf16_gemv_fp32_mmvf", shape, tm, (double) n_out, (double) n_out * n_in);
        }
        std::printf("XPAIR bf16_gemv %s | ported bf16_gemv (bf16 act) | native bf16_gemv_fp32_mmvf (f32 act) | "
                    "native/bf16_gemv %.3f\n", shape, tm.med / tv.med);
        ctx.free(bx); ctx.free(bx32); ctx.free(bw); ctx.free(by);
    }
}

// =========================================================================================================
// THE PER-DISPATCH GAP, ATTRIBUTED: the engine's recorded replay (64.0 us per executed dispatch) and its
// live prefill (152.5 us) against this harness's in-stream marginal (5-20 us) - the largest single
// unexplained number in the port's record (PERFORMANCE-B70-2026-10-06.md, section 3).
//
// EVERY existing row in this file is K copies of ONE kernel, on the SAME few buffers, warm.  The engine's
// verify window is a LONG chain (~1,400 dispatches per segment, 4,222 per round at the 199-token arm) of
// SMALL, DIFFERENT, dependent dispatches over a 27 GiB arena, replayed from a recorded command buffer.
// This arm holds the WORK per dispatch constant and moves ONE structural property at a time, so the gap
// can be ATTRIBUTED rather than asserted:
//
//   gap_uniform      K copies of `scale`, ONE mapped buffer        (this harness's normal shape)
//   gap_altshader    `scale` <-> `add`, same grid, same buffer     (a DIFFERENT pipeline EVERY dispatch)
//   gap_devlocal     K copies of `scale`, a DEVICE_LOCAL buffer    (the engine's arena memory type)
//   gap_altbuffer    `scale` on K spread views of ONE 1 GiB buffer (the descriptor's target moves)
//   gap_engineshape  alt-shader + device-local + rotating views    (all three at once)
//   gap_batch        `scale` at K = 1 .. 1408 in ONE buffer        (the fixed/marginal split: F and c)
//
// `scale` and `add` are chosen because they do the SAME work at the SAME grid (one f32 element per thread,
// `n` of them) and differ ONLY in their shader binary and binding count - so alt-shader changes the
// pipeline and nothing else about the work.  Every row is the SAME instrument as the rest of the file
// (wall clock around a recorded-batch fence, median/batch) and carries the same documented limit: a
// per-dispatch cost, NOT a bandwidth figure.
// =========================================================================================================
template <class RecordIter>
Timing time_iters(Ctx& ctx, int iters, int disp_per_iter, int reps, int warmups, RecordIter rec) {
    ctx.record_begin();
    for (int i = 0; i < iters; ++i) rec(i);
    ctx.record_end_and_submit();
    const int total = iters * disp_per_iter;
    for (int w = 0; w < warmups; ++w) ctx.replay_recorded();
    std::vector<double> ms;
    ms.reserve((size_t) reps);
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        ctx.replay_recorded();
        const auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / (double) total);
    }
    std::sort(ms.begin(), ms.end());
    Timing t;
    t.reps = reps;
    t.batch = total;
    t.med = ms[ms.size() / 2];
    t.lo = ms.front();
    t.hi = ms.back();
    return t;
}

void bench_dispatch_gap(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const uint32_t n = 16384;                                // 64 workgroups x 256 lanes, a per-token activation
    const uint32_t gx = (n + 255) / 256;
    struct { int32_t n; float s; } pcs{n, 1.0009765625f};    // `scale`: x[i] *= s (s != 1, so it is real work)
    struct { int32_t n; } pca{n};                            // `add`:   dst[i] += src[i]
    std::vector<float> x = floats(n), y = floats(n);
    const std::string shape = "scale/add n=16384, grid 64x256";

    VkPipeline ps = ctx.pipeline(dir + "/scale.spv", 1, (int) sizeof(pcs));
    VkPipeline pa = ctx.pipeline(dir + "/add.spv", 2, (int) sizeof(pca));

    Buf ms_ = alloc(ctx, (size_t) n * 4), md_ = alloc(ctx, (size_t) n * 4);            // mapped (the bench type)
    Buf ds_ = ctx.alloc_device((size_t) n * 4), dd_ = ctx.alloc_device((size_t) n * 4); // VRAM (the arena type)
    ctx.write(ms_, x.data(), (size_t) n * 4);
    ctx.write(md_, y.data(), (size_t) n * 4);
    ctx.write(ds_, x.data(), (size_t) n * 4);
    ctx.write(dd_, y.data(), (size_t) n * 4);

    // the harness's normal shape: K copies of ONE kernel, ONE warm buffer
    report("gap_uniform", shape, time_iters(ctx, 128, 1, reps, warmups, [&](int) {
        ctx.record_dispatch(ps, {&ms_}, &pcs, sizeof(pcs), gx, 1);
    }), (double) n, 0.0);
    // a DIFFERENT pipeline every dispatch (identical work, identical grid, identical buffer)
    report("gap_altshader", shape, time_iters(ctx, 128, 2, reps, warmups, [&](int) {
        ctx.record_dispatch(ps, {&ms_}, &pcs, sizeof(pcs), gx, 1);
        ctx.record_dispatch(pa, {&md_, &ms_}, &pca, sizeof(pca), gx, 1);
    }), (double) n, 0.0);
    // the engine's arena memory type, one buffer, no rotation
    report("gap_devlocal", shape, time_iters(ctx, 128, 1, reps, warmups, [&](int) {
        ctx.record_dispatch(ps, {&ds_}, &pcs, sizeof(pcs), gx, 1);
    }), (double) n, 0.0);

    // THE DESCRIPTOR'S TARGET MOVES: K spread views of ONE device-local buffer, 8 MiB apart, so the 128
    // views cover 1 GiB - the address spread the engine's 27 GiB arena has and a 4-buffer bench does not.
    const int nview = 128;
    const uint64_t stride = 8ull << 20;
    Buf big = ctx.alloc_device(stride * (uint64_t) nview);
    for (int i = 0; i < nview; ++i) {
        Buf v = view(big, stride * (uint64_t) i);
        ctx.write(v, x.data(), (size_t) n * 4);
    }
    report("gap_altbuffer", "scale on 128 views, 8 MiB stride over 1 GiB, device-local",
           time_iters(ctx, 128, 1, reps, warmups, [&](int i) {
               Buf v = view(big, stride * (uint64_t) (i % nview));
               ctx.record_dispatch(ps, {&v}, &pcs, sizeof(pcs), gx, 1);
           }), (double) n, 0.0);

    // ALL THREE AT ONCE: a different pipeline, a moving descriptor target, and device-local memory
    report("gap_engineshape", "scale<->add on 128 rotating 1 GiB views, device-local",
           time_iters(ctx, 128, 2, reps, warmups, [&](int i) {
               Buf v = view(big, stride * (uint64_t) (i % nview));
               ctx.record_dispatch(ps, {&v}, &pcs, sizeof(pcs), gx, 1);
               ctx.record_dispatch(pa, {&v, &v}, &pca, sizeof(pca), gx, 1);
           }), (double) n, 0.0);

    // THE FIXED/MARGINAL SPLIT.  per-dispatch = F/K + c for ONE command buffer of K dispatches, so two
    // rows give F and c - and the engine's ~1,400-dispatch segments can be read against BOTH.
    const int bs[] = {1, 2, 8, 32, 128, 512, 1408};
    for (int K : bs) {
        char sh[80];
        std::snprintf(sh, sizeof sh, "scale n=16384 K=%d (ONE command buffer)", K);
        report("gap_batch", sh, time_iters(ctx, K, 1, reps, warmups, [&](int) {
            ctx.record_dispatch(ps, {&ms_}, &pcs, sizeof(pcs), gx, 1);
        }), (double) n, 0.0);
    }

    ctx.free(ms_); ctx.free(md_); ctx.free(ds_); ctx.free(dd_); ctx.free(big);
}

// =========================================================================================================
// TASK 1 AUDIT (a): `native_k_mmvq` - the "biggest dispatch producer" claim, checked, AND its launch shape.
//
// The brief sized the prefill's dispatch pressure as "native_k_mmvq 41,580 dispatches in one prefill
// histogram, ~50% of the whole dispatch layer".  That figure PREDATES the batched native GEMM
// (`prefill_vk.cpp::Gemm::native`, the `beta == 0 && ldy == N` arm): the per-token loop it describes pays
// three dispatches PER TOKEN (f16_to_f32, quantize_q8_1, native_k_mmvq) for each of the 210 K-quant dense
// tensors, hence 210 x 198 = 41,580; the batched arm pays three dispatches per PROJECTION, so 210.  The
// engine's OWN current histogram agrees (base199): `native_k_mmvq.spv` is ABSENT from the top-14 of the
// run's 59,640 live dispatches, whose 14th entry is 1,152 - so the whole 198-token prefill plus 32 decoded
// tokens uses < 1,152.  This arm prices both shapes, and shows why there is no empty launch here to cap:
// the host passes a ONE-DIMENSIONAL grid (`n_out`), there is no window loop (the only two
// `for (uint32_t w < nwin)` loops in the tree are `native_expert_grouped` and `verify::fetch_blobs`), and
// the shader reads no `gl_WorkGroupID.y` - `gy` is not a knob this launcher HAS.
// =========================================================================================================
void bench_native_k_mmvq_engine(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n_in = 2560, n_out = 1280, ty = 14;            // Q6_K: the commonest dense K-quant in coder-iq1_m
    const int row_bytes = (n_in / 256) * 210;                // Q6_K: 256 values in 210 B, 10 blocks per row
    const int ncols_big = 198;                               // the engine's own --prefill chunk
    VkPipeline p = ctx.pipeline(dir + "/native_k_mmvq.spv", 3, 20);
    std::vector<uint8_t> wbuf((size_t) n_out * row_bytes);
    for (size_t i = 0; i < wbuf.size(); ++i) wbuf[i] = (uint8_t) (i * 7 + 1);
    Buf bw = ctx.alloc_device(wbuf.size());
    ctx.write(bw, wbuf.data(), wbuf.size());
    const std::string shape = "Q6_K n_in=2560 n_out=1280 device-local, 1-D grid (n_out,1)";

    auto q8buf = [&](int nc) {
        std::vector<uint8_t> a((size_t) nc * (n_in / 32) * 36);
        for (size_t i = 0; i < a.size(); ++i) a[i] = (uint8_t) (i * 11 + 3);
        return a;
    };
    {   // THE BATCHED FORM - one dispatch for all 198 tokens (the shipped prefill path)
        const int nc = ncols_big;
        std::vector<uint8_t> a = q8buf(nc);
        Buf ba = ctx.alloc_device(a.size()), bo = ctx.alloc_device((size_t) n_out * nc * 4);
        ctx.write(ba, a.data(), a.size());
        struct { int32_t n_in, n_out, row_bytes, ncols, ty; } pc{n_in, n_out, row_bytes, nc, ty};
        report("k_mmvq_batched", shape + " | ncols=198: 1 dispatch for 198 tokens",
               time_kernel(ctx, p, {&bw, &ba, &bo}, &pc, sizeof(pc), (uint32_t) n_out, 1, 8, reps, warmups),
               (double) n_out * nc, 0.0);
        ctx.free(ba); ctx.free(bo);
    }
    {   // THE PER-TOKEN FORM - one dispatch per token (the form the 41,580 figure counted)
        std::vector<uint8_t> a = q8buf(1);
        Buf ba = ctx.alloc_device(a.size()), bo = ctx.alloc_device((size_t) n_out * 4);
        ctx.write(ba, a.data(), a.size());
        struct { int32_t n_in, n_out, row_bytes, ncols, ty; } pc{n_in, n_out, row_bytes, 1, ty};
        report("k_mmvq_pertoken", shape + " | ncols=1: 1 dispatch PER TOKEN (the 41,580 form)",
               time_kernel(ctx, p, {&bw, &ba, &bo}, &pc, sizeof(pc), (uint32_t) n_out, 1, 8, reps, warmups),
               (double) n_out, 0.0);
        ctx.free(ba); ctx.free(bo);
    }
    {   // THE LAUNCH SHAPE, in the gu/down sweep's own shape: a y-grid multiplies WORKGROUPS, never "empty"
        // ones - there is no guard to fail, so a `gy` sweep has no floor to find.  8x the grid, ~8x the time.
        std::vector<uint8_t> a = q8buf(1);
        Buf ba = ctx.alloc_device(a.size()), bo = ctx.alloc_device((size_t) n_out * 4);
        ctx.write(ba, a.data(), a.size());
        struct { int32_t n_in, n_out, row_bytes, ncols, ty; } pc{n_in, n_out, row_bytes, 1, ty};
        report("k_mmvq_gy8_spurious", shape + " | gy=8: the shader reads NO WorkGroupID.y, so it is 8x WORK",
               time_kernel(ctx, p, {&bw, &ba, &bo}, &pc, sizeof(pc), (uint32_t) n_out, 8, 8, reps, warmups),
               (double) n_out, 0.0);
        ctx.free(ba); ctx.free(bo);
    }
    ctx.free(bw);
}

// =========================================================================================================
// TASK 1 AUDIT (b): the grouped launcher's SECOND per-window waste - the WINDOW-INDEPENDENT stages.
//
// The window loop exists ONLY for the weight read (a 32-bit storage-buffer index reaches one 4 GiB window).
// TWO of the launcher's four stages read NO weights: the SwiGLU over the gate/up scratch, and the q8_1
// quantise of its result.  Both sat INSIDE the window loop, so a call ran `nwin`=8 SwiGLU launches and 8
// q8_1 launches - 7 of each recomputing the byte-identical result over the identical scratch.  Same species
// as the empty launch, one layer up.  At the engine's shape (30 groups = the --spec 2 window's 3 tokens x
// top-10, n_embd 2560, n_ff 1280, the arena's 8 windows, device-local):
//   grp_call_current   8 x (gu + swiglu + q8_1 + down)     <- the shipped form
//   grp_call_hoist     8 x gu + swiglu + q8_1 + 8 x down   <- the fix (gu and down are window-bound only)
// =========================================================================================================
void bench_native_grouped_hoist(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int n_embd = 2560, n_ff = 1280, cap = 30, nwin = 8;
    const int ty_gu = 18, ty_dn = 20;                        // IQ3_XXS gu / IQ4_NL down (layer 0's formats)
    const int gu_row = (n_embd / 256) * 98, gu_blob = 2 * n_ff * gu_row;
    const int dn_row = (n_ff / 32) * 18, dn_blob = dn_row * n_embd;
    std::vector<uint8_t> wgu((size_t) cap * gu_blob), wdn((size_t) cap * dn_blob);
    for (size_t i = 0; i < wgu.size(); ++i) wgu[i] = (uint8_t) (i * 29 + 7);
    for (size_t i = 0; i < wdn.size(); ++i) wdn[i] = (uint8_t) (i * 37 + 11);
    std::vector<uint32_t> off_gu((size_t) cap), off_dn((size_t) cap);
    for (int g = 0; g < cap; ++g) {
        off_gu[(size_t) g] = (uint32_t) ((size_t) g * gu_blob);
        off_dn[(size_t) g] = (uint32_t) ((size_t) g * dn_blob);
    }
    std::vector<uint8_t> act((size_t) cap * (n_embd / 32) * 36);
    for (size_t i = 0; i < act.size(); ++i) act[i] = (uint8_t) (i * 11 + 3);

    Buf b_w = ctx.alloc_device(wgu.size()), b_wd = ctx.alloc_device(wdn.size());
    Buf b_act = ctx.alloc_device(act.size());
    Buf b_off = ctx.alloc((size_t) cap * 4), b_offd = ctx.alloc((size_t) cap * 4);
    Buf b_win = ctx.alloc((size_t) cap * 4), b_start = ctx.alloc(((size_t) cap + 1) * 4), b_ng = ctx.alloc(4);
    Buf b_tok = ctx.alloc((size_t) cap * 4), b_sd = ctx.alloc(((size_t) cap + 1) * 4), b_ngd = ctx.alloc(4);
    Buf b_g1 = ctx.alloc(sizeof(strata::vkport::kIq2sGrid)), b_g2 = ctx.alloc(sizeof(strata::vkport::kIq3xxsGrid));
    Buf b_g3 = ctx.alloc(sizeof(strata::vkport::kIq3sGrid));
    Buf b_gate = ctx.alloc_device((size_t) cap * n_ff * 4u + 256u);
    Buf b_up = ctx.alloc_device((size_t) cap * n_ff * 4u + 256u);
    Buf b_h = ctx.alloc_device((size_t) cap * n_ff * 4u + 256u);
    Buf b_hq = ctx.alloc_device((size_t) cap * (n_ff / 32) * 36u);
    Buf b_out = ctx.alloc_device((size_t) cap * n_embd * 4u + 256u);
    ctx.write(b_w, wgu.data(), wgu.size()); ctx.write(b_wd, wdn.data(), wdn.size());
    ctx.write(b_act, act.data(), act.size());
    ctx.write(b_off, off_gu.data(), off_gu.size() * 4);
    ctx.write(b_offd, off_dn.data(), off_dn.size() * 4);
    std::vector<uint32_t> win0((size_t) cap, 0u);            // every group in ONE window (the engine's real case)
    ctx.write(b_win, win0.data(), win0.size() * 4);
    std::vector<int32_t> st((size_t) cap + 1), tk((size_t) cap);
    for (int g = 0; g <= cap; ++g) st[(size_t) g] = g;
    for (int g = 0; g < cap; ++g) tk[(size_t) g] = g % 3;
    ctx.write(b_start, st.data(), st.size() * 4); ctx.write(b_sd, st.data(), st.size() * 4);
    ctx.write(b_tok, tk.data(), tk.size() * 4); ctx.write(b_ng, &cap, 4); ctx.write(b_ngd, &cap, 4);
    ctx.write(b_g1, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    ctx.write(b_g2, strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid));
    ctx.write(b_g3, strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid));

    VkPipeline pgu = ctx.pipeline(dir + "/native_gu_any.spv", 12, 16);
    VkPipeline psw = ctx.pipeline(dir + "/swiglu_f32.spv", 3, 4);
    VkPipeline pq8 = ctx.pipeline(dir + "/quantize_q8_1.spv", 2, 8);
    VkPipeline pdn = ctx.pipeline(dir + "/native_down_any.spv", 8, 24);
    struct { int n_embd, n_ff, ty, win_id; } pcg{n_embd, n_ff, ty_gu, 0};
    struct { int n; } pcs{cap * n_ff};
    struct { int n_in, ncols; } pcq{n_ff, cap};
    struct { int n_embd, n_ff, ty, win_id, d_row, down_off; } pcd{n_embd, n_ff, ty_dn, 0, dn_row, 0};
    auto gu = [&](int w) { pcg.win_id = w;
        ctx.record_dispatch(pgu, {&b_w, &b_act, &b_g1, &b_g2, &b_g3, &b_off, &b_win, &b_start, &b_ng, &b_tok,
                                  &b_gate, &b_up}, &pcg, sizeof(pcg), (uint32_t) (2 * n_ff), 1u); };
    auto sw = [&] { ctx.record_dispatch(psw, {&b_gate, &b_up, &b_h}, &pcs, sizeof(pcs),
                                        (uint32_t) ((cap * n_ff + 255) / 256), 1u); };
    auto q8 = [&] { ctx.record_dispatch(pq8, {&b_h, &b_hq}, &pcq, sizeof(pcq),
                                        (uint32_t) ((n_ff + 255) / 256 * cap), 1u); };
    auto dn = [&](int w) { pcd.win_id = w;
        ctx.record_dispatch(pdn, {&b_wd, &b_hq, &b_offd, &b_win, &b_sd, &b_ngd, &b_tok, &b_out}, &pcd, sizeof(pcd),
                            (uint32_t) n_embd, 1u); };
    const std::string shape = "30 groups (3 tok x 10), n_embd 2560 n_ff 1280, nwin 8, device-local";
    // the two ingredients on their own, so the delta is decomposed rather than attributed
    report("grp_swiglu1", shape + " | ONE swiglu over cap*n_ff (window-independent stage)",
           time_iters(ctx, 4, 1, reps, warmups, [&](int) { sw(); }), (double) cap * n_ff, 0.0);
    report("grp_q8_1", shape + " | ONE q8_1 quantise of the intermediate (window-independent stage)",
           time_iters(ctx, 4, 1, reps, warmups, [&](int) { q8(); }), (double) cap * (n_ff / 32) * 36, 0.0);
    const Timing cur = time_iters(ctx, 4, 8 * 4, reps, warmups, [&](int) {
        for (int w = 0; w < nwin; ++w) { gu(w); sw(); q8(); dn(w); }
    });
    const Timing ho = time_iters(ctx, 4, 8 + 1 + 1 + 8, reps, warmups, [&](int) {
        for (int w = 0; w < nwin; ++w) gu(w);
        sw(); q8();
        for (int w = 0; w < nwin; ++w) dn(w);
    });
    report("grp_call_current", shape + " | 8 x (gu + swiglu + q8_1 + down) = 32 dispatches/call", cur, 32.0, 0.0);
    report("grp_call_hoist", shape + " | 8 x gu + swiglu + q8_1 + 8 x down = 18 dispatches/call", ho, 18.0, 0.0);
    const double c = cur.med * 32.0, h = ho.med * 18.0;
    std::printf("      grouped per CALL: current %.4f ms (32 disp) | hoist %.4f ms (18 disp) | delta %.4f ms (%.1f%%)\n",
                c, h, c - h, 100.0 * (c - h) / c);
    ctx.free(b_w); ctx.free(b_wd); ctx.free(b_act); ctx.free(b_off); ctx.free(b_offd); ctx.free(b_win);
    ctx.free(b_start); ctx.free(b_ng); ctx.free(b_tok); ctx.free(b_sd); ctx.free(b_ngd);
    ctx.free(b_g1); ctx.free(b_g2); ctx.free(b_g3);
    ctx.free(b_gate); ctx.free(b_up); ctx.free(b_h); ctx.free(b_hq); ctx.free(b_out);
}

// =========================================================================================================
// TASK 2: `fused_gr_mix` (68.3 us, the largest priced single kernel) - ONE PASS AT WHETHER IT CAN BE CUT.
//
// Per workgroup (one output column d) the kernel loops hc=4 streams; each stream's gate is a 320-long dot
// reduced by the WORKGROUP BARRIER TREE (`wg_sum`, 8 rounds of 2 barriers).  Its arithmetic is 5 MACs per
// thread, so the reductions are the whole cost.  This arm varies `hc` on the SAME shader to price one
// reduction: if the time tracks the number of `wg_sum` calls, the kernel is reduction-bound, and the only
// safe lever - fewer lanes or fewer rounds - changes the summation TREE, which changes the last bits and
// moves the ids the engine's contract pins.  That is the "no" this arm has to price.
// =========================================================================================================
void bench_fused_gr_mix_pass(Ctx& ctx, const std::string& dir, int reps, int warmups) {
    const int N = 2560, LR = 320;
    const int64_t hc_dim = (int64_t) 4 * N;
    std::vector<float> r = floats((size_t) hc_dim), nrm = floats((size_t) hc_dim), lof = floats((size_t) LR);
    std::vector<float> rs = floats(4), mix = floats((size_t) N);
    Buf b_r = ctx.alloc_device((size_t) hc_dim * 4), b_n = ctx.alloc_device((size_t) hc_dim * 4);
    Buf b_l = ctx.alloc_device((size_t) LR * 4), b_s = ctx.alloc_device((size_t) 4 * 4);
    Buf b_u = ctx.alloc_device((size_t) hc_dim * (size_t) (LR / 2) * 4);
    Buf b_m = ctx.alloc_device((size_t) N * 4);
    ctx.write(b_r, r.data(), r.size() * 4); ctx.write(b_n, nrm.data(), nrm.size() * 4);
    ctx.write(b_l, lof.data(), lof.size() * 4); ctx.write(b_s, rs.data(), rs.size() * 4);
    ctx.write(b_m, mix.data(), mix.size() * 4);
    {
        std::vector<uint8_t> wu((size_t) b_u.bytes, 0);
        for (size_t i = 0; i < wu.size(); ++i) wu[i] = (uint8_t) (i * 7 + 3);
        ctx.write(b_u, wu.data(), wu.size());
    }
    VkPipeline p_mx = ctx.pipeline(dir + "/fused_gr_mix.spv", 6, 12);
    const std::string shape = "N=2560 LR=320 (the artifact's geometry), 1 workgroup per output column";
    for (int hc : {4, 2, 1}) {   // the artifact's hc is 4; the other two price ONE barrier reduction each
        struct { int n_embd, hc, hc_lr; } pc{N, hc, LR};
        char tag[48];
        std::snprintf(tag, sizeof tag, "mix_hc%d (%d barrier reductions/col)", hc, hc);
        report(tag, shape + " | the SAME shader, `hc` varied: the reduction cost, isolated",
               time_kernel(ctx, p_mx, {&b_r, &b_n, &b_s, &b_l, &b_u, &b_m}, &pc, sizeof(pc), (uint32_t) N, 1, 8,
                           reps, warmups), (double) N, (double) N * (double) hc * LR);
    }
    ctx.free(b_r); ctx.free(b_n); ctx.free(b_l); ctx.free(b_s); ctx.free(b_u); ctx.free(b_m);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "shaders";
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR")) dir = e;
    int dev = -1, reps = 9, warmups = 3, sampler_vocab = 248320;
    bool list = false;
    std::string only;              // --only <arm>: run exactly one named arm (see the ledger in main)
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") list = true;
        else if (a == "--spv-dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--warmups" && i + 1 < argc) warmups = std::atoi(argv[++i]);
        else if (a == "--only" && i + 1 < argc) only = argv[++i];
        else if (a == "--sampler-vocab" && i + 1 < argc) sampler_vocab = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: vk_bench [--spv-dir D] [--device N] [--reps R] [--warmups W] [--only ARM] [--sampler-vocab N] [--list]\n"); return 2; }
    }
    if (list) {
        for (auto& d : Ctx::list_devices()) {
            std::printf("%s (vendor 0x%04x device 0x%04x, subgroup %u, 8bit-storage %d, fp64 %d)\n", d.name.c_str(),
                        d.vendor_id, d.device_id, d.subgroup_size, (int) d.storage_buffer_8bit, (int) d.shader_float64);
        }
        return 0;
    }

    // The display contract, stated in the bench's own output as well as by the layer (which prints its budget
    // line to stderr).  The numbers below are what configure_display_reserve() reads.
    std::printf("== vk_bench: reserve MIB=%s floor MIB=%s max-budget MIB=%s | spv-dir %s | reps %d warmups %d | "
                "timing: wall clock around the recorded-batch fence, per-dispatch = median of the timed replays / batch\n",
                std::getenv("STRATA_VK_DESKTOP_RESERVE_MIB") ? std::getenv("STRATA_VK_DESKTOP_RESERVE_MIB") : "(default 1024)",
                std::getenv("STRATA_VK_RESERVE_FLOOR_MIB") ? std::getenv("STRATA_VK_RESERVE_FLOOR_MIB") : "(default 512)",
                std::getenv("STRATA_VK_MAX_BUDGET_MIB") ? std::getenv("STRATA_VK_MAX_BUDGET_MIB") : "(unset)",
                dir.c_str(), reps, warmups);

    Ctx ctx(dev, false);
    ctx.configure_display_reserve();
    const auto& di = ctx.info();
    std::printf("== device %d \"%s\" (vendor 0x%04x) subgroup %u | 8bit-storage %d 16bit-storage %d fp64 %d | "
                "usable %.2f GiB\n",
                ctx.device_index(), di.name.c_str(), di.vendor_id, di.subgroup_size, (int) di.storage_buffer_8bit,
                (int) di.storage_buffer_16bit, (int) di.shader_float64, (double) ctx.usable_bytes() / 1073741824.0);
    std::printf("== %-20s | %-30s | %-15s | %-18s | %-18s | %-8s | %-9s | %-12s | %-11s | %-13s\n",
                "kernel", "shape", "median", "min", "max", "reps", "batch", "work", "elems/s", "macs/s");
    std::fflush(stdout);

    // ---- THE ARM LEDGER -----------------------------------------------------------------------------------
    // A sweep that dies mid-way must NAME the arm it died in, and an arm that cannot run must be a NAMED skip
    // rather than a silent return - otherwise a missing row is indistinguishable from a pass.  Each arm prints
    // `-- arm <name>` to stderr (flushed) BEFORE it runs and `-- arm <name> OK (<n> rows)` after; an arm that
    // produces no ROW is a named failure, not an absence.  The run ends with a ledger line and exits 1 if any
    // arm failed or was empty.  `--only <name>` runs exactly one arm so a failure is bisected without a
    // 60-row sweep; an unknown name lists the arms and exits 2.
    size_t ran = 0, skipped = 0, norow = 0;
    bool matched = false;
    auto arm = [&](const char* name, bool needs_8bit, const std::function<void()>& fn) {
        if (!only.empty() && only != name) return;
        matched = true;
        if (needs_8bit && !ctx.info().storage_buffer_8bit) {
            std::printf("SKIP %-22s | device lacks storageBuffer8BitAccess\n", name);
            ++skipped;
            return;
        }
        const uint64_t rows_before = g_rows;
        std::fflush(stdout);
        const long pos_before = std::ftell(stdout);
        std::fprintf(stderr, "-- arm %s ...\n", name);
        std::fflush(stderr);
        fn();
        std::fflush(stdout);
        const long pos_after = std::ftell(stdout);
        const uint64_t got = g_rows - rows_before;
        // EVIDENCE = BYTES ON STDOUT, not ROW count.  The first version of this ledger counted only `report()`
        // rows and therefore called `gdn_rec_batch_sweep` (which emits SWEEP lines) an empty arm - a check that
        // fails on a good arm is as bad as one that cannot fail.  ftell is only meaningful when stdout is a
        // regular file or a pipe; on a terminal it returns -1 and the ROW delta is the fallback.
        const bool measured = pos_before >= 0 && pos_after >= 0;
        const long bytes = measured ? (pos_after - pos_before) : -1;
        const bool printed = measured ? (bytes > 0) : (got > 0);
        if (!printed) {
            std::fprintf(stderr, "ARM %s FAILED: it printed nothing at all (a missing .spv, a device the shape "
                                 "rejects, or an allocation failure - re-run with --only %s for the detail)\n",
                         name, name);
            ++norow;
        } else if (measured && bytes > 0 && bytes < 8) {
            // A single SKIP line is evidence that the arm NAMED its own reason; anything shorter than a line is
            // not.  Counted as run, but the byte count is printed so a skip-only arm cannot hide.
            std::fprintf(stderr, "-- arm %s OK (%llu rows, %ld bytes - short: a named skip?)\n", name,
                         (unsigned long long) got, bytes);
            ++ran;
        } else {
            std::fprintf(stderr, "-- arm %s OK (%llu rows, %ld bytes)\n", name, (unsigned long long) got, bytes);
            ++ran;
        }
        std::fflush(stderr);
    };

    arm("gdn_conv_step", false, [&] { bench_gdn_conv_step(ctx, dir, reps, warmups); });
    arm("gdn_l2_norm", false, [&] { bench_gdn_l2_norm(ctx, dir, reps, warmups); });
    arm("gdn_beta_gate", false, [&] { bench_gdn_beta_gate(ctx, dir, reps, warmups); });
    arm("gdn_gate", false, [&] { bench_gdn_gate(ctx, dir, reps, warmups); });
    arm("gdn_step", false, [&] { bench_gdn_step(ctx, dir, reps, warmups); });
    arm("gdn_out_norm", false, [&] { bench_gdn_out_norm(ctx, dir, reps, warmups); });

    arm("iq_dequant_bf16_256", true, [&] { bench_iq_dequant(ctx, dir, 30, "BF16", 2, 256, reps, warmups); });
    arm("iq_dequant_iq4nl_256", true, [&] { bench_iq_dequant(ctx, dir, 20, "IQ4_NL", 18, 256, reps, warmups); });
    arm("iq_dequant_iq2s_256", true, [&] { bench_iq_dequant(ctx, dir, 11, "IQ2_S", 82, 256, reps, warmups); });
    // the SIZE-SCALING arm (4x)
    arm("iq_dequant_iq2s_1024", true, [&] { bench_iq_dequant(ctx, dir, 11, "IQ2_S", 82, 1024, reps, warmups); });
    // THE ENGINE'S OWN SHAPE: one expert's gate/up (or down) projection is n_ff*n_embd / 256 = 1280*2560/256
    // = 12,800 superblocks per `iq_dequant_f32` call, 3 such calls per expert (gate, up, down).  The 256/1024
    // arms above are DISPATCH-BOUND (fixed per-launch cost dominates), so they cannot see an occupancy change;
    // these are where the port's dequant work actually lives.
    arm("iq_dequant_iq2s_12800", true, [&] { bench_iq_dequant(ctx, dir, 11, "IQ2_S", 82, 12800, reps, warmups); });
    arm("iq_dequant_iq4nl_12800", true, [&] { bench_iq_dequant(ctx, dir, 20, "IQ4_NL", 18, 12800, reps, warmups); });

    arm("iq2s_mmvq_512", true, [&] { bench_iq2s(ctx, dir, 512, reps, warmups); });
    // the SIZE-SCALING arm (4x)
    arm("iq2s_mmvq_2048", true, [&] { bench_iq2s(ctx, dir, 2048, reps, warmups); });
    // THE WINDOW'S OWN KERNEL FAMILY: IQ1_M MMVQ at the engine's gate/up (1280) and down (2560) shapes,
    // one column (one token) and three, mapped vs the arena's DEVICE_LOCAL type.
    arm("iq1m_mmvq_512", false, [&] { bench_iq1m(ctx, dir, 512, 1, false, reps, warmups); });
    arm("iq1m_mmvq_1280", false, [&] { bench_iq1m(ctx, dir, 1280, 1, false, reps, warmups); });
    arm("iq1m_mmvq_2560", false, [&] { bench_iq1m(ctx, dir, 2560, 1, false, reps, warmups); });
    arm("iq1m_mmvq_2560_dev", false, [&] { bench_iq1m(ctx, dir, 2560, 1, true, reps, warmups); });
    arm("iq1m_mmvq_1280_n3_dev", false, [&] { bench_iq1m(ctx, dir, 1280, 3, true, reps, warmups); });
    // THE CONTROLLED COMPARISON the ncols lever rests on: the SAME four rows (ncols 1/3 x mapped/dev) in ONE
    // process, interleaved and repeated, so a memory-type or process-POSITION effect cannot be read as an
    // ncols effect (the port's own "a row's position in the process decides its value" artifact).
    arm("iq1m_controlled", false, [&] {
        bench_iq1m(ctx, dir, 1280, 1, false, reps, warmups);
        bench_iq1m(ctx, dir, 1280, 3, false, reps, warmups);
        bench_iq1m(ctx, dir, 1280, 1, true, reps, warmups);
        bench_iq1m(ctx, dir, 1280, 3, true, reps, warmups);
        bench_iq1m(ctx, dir, 1280, 3, false, reps, warmups);
        bench_iq1m(ctx, dir, 1280, 1, true, reps, warmups);
    });

    arm("quantize_q8_0", false, [&] { bench_quantize_q8_0(ctx, dir, reps, warmups); });
    arm("quantize_q8_1", true, [&] { bench_quantize_q8_1(ctx, dir, reps, warmups); });
    arm("quantize_q8_K", false, [&] { bench_quantize_q8_K(ctx, dir, reps, warmups); });
    // this batch: the shared expert's split-GEMV pair
    arm("s_gemv_q8_split", false, [&] { bench_s_gemv_q8_split(ctx, dir, reps, warmups); });

    // THE PERFORMANCE TIER, class B: native vs legacy, same shape, same device (a native/legacy ratio < 1 is
    // faster).  Before the sampler, which is the heavy one.
    arm("rope_pair", false, [&] { bench_rope_pair(ctx, dir, reps, warmups); });
    arm("router_pair", false, [&] { bench_router_pair(ctx, dir, reps, warmups); });
    arm("moe_combine_pair", false, [&] { bench_moe_combine_pair(ctx, dir, reps, warmups); });
    arm("rms_norm_pair", false, [&] { bench_rms_norm_pair(ctx, dir, reps, warmups); });
    // class B: native_qsa_gate_apply <- qsa_gate_apply_f32
    arm("qsa_gate_pair", false, [&] { bench_qsa_gate_pair(ctx, dir, reps, warmups); });
    // the DEFAULT QSA decode attention (qsa_decode_attn_step)
    arm("qsa_decode_attn", false, [&] { bench_qsa_decode_attn(ctx, dir, reps, warmups); });
    // batch 2: the native GDN / DeltaNet mixer kernels (conv+SiLU, l2_norm, beta_gate).
    arm("gdn_conv_silu_pair", false, [&] { bench_gdn_conv_silu_pair(ctx, dir, reps, warmups); });
    arm("gdn_l2_norm_pair", false, [&] { bench_gdn_l2_norm_pair(ctx, dir, reps, warmups); });
    arm("gdn_beta_gate_pair", false, [&] { bench_gdn_beta_gate_pair(ctx, dir, reps, warmups); });
    // batch 3: the remaining three native GDN / DeltaNet mixer kernels (gate, out_norm, step).
    arm("gdn_gate_pair", false, [&] { bench_gdn_gate_pair(ctx, dir, reps, warmups); });
    arm("gdn_out_norm_pair", false, [&] { bench_gdn_out_norm_pair(ctx, dir, reps, warmups); });
    arm("gdn_step_pair", false, [&] { bench_gdn_step_pair(ctx, dir, reps, warmups); });
    // PERFORMANCE TIER, batch 4: the THREE FUSED GDN PATHS, each against the multi-dispatch chain it replaces and
    // against the non-fused native kernel(s) where a comparison exists.
    arm("fused_gdn_conv_l2_pair", false, [&] { bench_fused_gdn_conv_l2_pair(ctx, dir, reps, warmups); });
    arm("fused_gdn_ab_pair", false, [&] { bench_fused_gdn_ab_pair(ctx, dir, reps, warmups); });
    arm("fused_gdn_step_norm_pair", false, [&] { bench_fused_gdn_step_norm_pair(ctx, dir, reps, warmups); });
    // THE RECURRENCE'S BATCH SWEEP - see the function's header for why a batch-8 row cannot answer it.
    arm("gdn_rec_batch_sweep", false, [&] { bench_gdn_rec_batch_sweep(ctx, dir, reps, warmups); });
    // THE STEP KERNEL'S OWN PROBE: shape (workgroups vs the card's lanes), the engine's batch with a fresh
    // state, and the 36-state (108 MiB) footprint the engine actually runs against.
    arm("gdn_step_probe", false, [&] { bench_gdn_step_probe(ctx, dir, reps, warmups); });
    // THE UNROLL A/B: the baseline .spv vs the unrolled one, bitwise first and then timed hot/cold.
    arm("gdn_step_unroll", false, [&] { bench_gdn_step_unroll(ctx, dir, reps, warmups); });
    // THE BF16-PROJECTION PAIR (`bf16_gemv` / `bf16_gemv_split`): the DEFAULT side of `native_bf16_projections`.
    arm("bf16_gemv_pair", false, [&] { bench_bf16_gemv_pair(ctx, dir, reps, warmups); });
    // THE PREFILL GEMM: the prompt path's own deep kernel, both cooperative-matrix schedules and both FMA
    // paths, at the engine's shapes.
    arm("gemm_prefill", false, [&] { bench_gemm_prefill(ctx, dir, reps, warmups); });
    // THE PER-DISPATCH GAP: the engine's replay (64.0 us) / live prefill (152.5 us) against this harness's
    // in-stream marginal (5-20 us).  Holds the work per dispatch fixed and moves ONE structural property at
    // a time (pipeline diversity, memory type, descriptor-target spread, command-buffer length).
    // THE WINDOW'S GROUPED EXPERT GEMV, AT THE ENGINE'S OWN SHAPE: `native_gu_any`/`native_down_any`, the
    // kernel the mixture model priced with a PROXY row.  One property at a time: group spread over the arena
    // windows (the port's shape) vs one window (the CUDA's), and 30 entries on one expert vs 30 groups.
    arm("native_grouped_engine", true, [&] { bench_native_grouped_engine(ctx, dir, reps, warmups); });
    // THE `fused_gr_*` GROUP PRICED - 576 dispatches of each of the four, the mixture model's whole residual.
    arm("gr_pricing", false, [&] { bench_gr_pricing(ctx, dir, reps, warmups); });
    // THIS BATCH - TASK 1: the "biggest dispatch producer" claim checked, and the grouped launcher's
    // WINDOW-INDEPENDENT stages (SwiGLU + q8_1) priced inside vs outside the window loop.
    arm("native_k_mmvq_engine", true, [&] { bench_native_k_mmvq_engine(ctx, dir, reps, warmups); });
    arm("native_grouped_hoist", true, [&] { bench_native_grouped_hoist(ctx, dir, reps, warmups); });
    // THIS BATCH - TASK 2: `fused_gr_mix`'s reduction cost isolated (hc varied on the same shader).
    arm("fused_gr_mix_pass", false, [&] { bench_fused_gr_mix_pass(ctx, dir, reps, warmups); });
    arm("dispatch_gap", false, [&] { bench_dispatch_gap(ctx, dir, reps, warmups); });

    // THE SAMPLER LAST, ON PURPOSE.  Its one-block top-k over the whole vocabulary is the port's heaviest
    // single dispatch, and on the Ryzen iGPU (RADV) the full-vocabulary shape was measured to trigger a
    // driver "hard recovery" (context lost).  Running it last means a driver reset on one ICD loses only the
    // sampler there, and the rows above are already printed.  --sampler-vocab bounds the row for a weaker
    // device; the default is the artifact's real vocabulary.
    arm("sampler", false, [&] { bench_sampler(ctx, dir, sampler_vocab, reps, warmups); });

    if (!only.empty() && !matched) {
        std::fprintf(stderr, "vk_bench: --only '%s' names no arm.  If it is not in the list below, it does not "
                             "exist - an unknown name is an error, never a silent no-op.\n", only.c_str());
        std::fprintf(stderr, "  valid --only names: gdn_conv_step gdn_l2_norm gdn_beta_gate gdn_gate gdn_step "
                             "gdn_out_norm iq_dequant_bf16_256 iq_dequant_iq4nl_256 iq_dequant_iq2s_256 "
                             "iq_dequant_iq2s_1024 iq_dequant_iq2s_12800 iq_dequant_iq4nl_12800 iq2s_mmvq_512 "
                             "iq2s_mmvq_2048 quantize_q8_0 quantize_q8_1 quantize_q8_K s_gemv_q8_split rope_pair "
                             "router_pair moe_combine_pair rms_norm_pair qsa_gate_pair qsa_decode_attn "
                             "gdn_conv_silu_pair gdn_l2_norm_pair gdn_beta_gate_pair gdn_gate_pair "
                             "gdn_out_norm_pair gdn_step_pair fused_gdn_conv_l2_pair fused_gdn_ab_pair "
                             "fused_gdn_step_norm_pair gdn_rec_batch_sweep gdn_step_probe gdn_step_unroll bf16_gemv_pair gemm_prefill native_grouped_engine gr_pricing native_k_mmvq_engine native_grouped_hoist fused_gr_mix_pass dispatch_gap sampler\n");
        return 2;
    }

    std::printf("== arms: %zu ran | %zu skipped | %zu failed (produced no row) | %llu rows total\n", ran, skipped,
                norow, (unsigned long long) g_rows);
    if (norow > 0) {
        std::printf("== vk_bench FAILED: %zu arm(s) printed nothing - the sweep is INCOMPLETE, not a pass\n", norow);
        return 1;
    }
    std::printf("== vk_bench done\n");
    return 0;
}
