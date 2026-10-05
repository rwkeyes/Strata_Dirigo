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
void report(const char* kernel, const std::string& shape, const Timing& t, double elems, double macs) {
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
// superblock; the grid is the superblock count.  bf16 (ty 30), IQ4_NL (ty 20) and IQ2_S (ty 11, the format
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
// THE SAMPLER - sampler_kernel_f32 (the portable path for devices without shaderFloat64; the target
// hardware).  One workgroup per token over the full vocabulary.
// =========================================================================================================

void bench_sampler(Ctx& ctx, const std::string& dir, int n_vocab, int reps, int warmups) {
    const int history_len = 64;                         // the artifact's window
    std::vector<float> row((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) row[(size_t) i] = 0.25f * rndf();
    row[7] = 9.0f; row[1234] = 8.0f; row[99999 % n_vocab] = 7.5f;  // a shortlist worth filtering
    std::vector<int32_t> hist((size_t) history_len, -1);
    for (int i = 0; i < 8; ++i) hist[(size_t) i] = 7;
    Buf b_l = alloc(ctx, (size_t) n_vocab * 4), b_h = alloc(ctx, (size_t) history_len * 4), b_o = alloc(ctx, 4);
    ctx.write(b_l, row.data(), (size_t) n_vocab * 4);
    ctx.write(b_h, hist.data(), (size_t) history_len * 4);
    struct Pc {
        int n_vocab, n_tokens, history_len, penalty_last_n, top_k, min_keep;
        float temperature, top_p, min_p, penalty_repeat, penalty_freq, penalty_present;
        uint32_t seed_lo, seed_hi, counter_lo, counter_hi;
    } pc{};
    pc.n_vocab = n_vocab; pc.n_tokens = 1; pc.history_len = history_len; pc.penalty_last_n = history_len;
    pc.top_k = 64; pc.min_keep = 1; pc.temperature = 1.0f; pc.top_p = 0.95f; pc.min_p = 0.0f;
    pc.penalty_repeat = 1.0f; pc.penalty_freq = 0.0f; pc.penalty_present = 0.0f;
    pc.seed_lo = 0x12345678u; pc.seed_hi = 0x9abcdef0u; pc.counter_lo = 0; pc.counter_hi = 0;
    VkPipeline p = ctx.pipeline(dir + "/sampler_kernel_f32.spv", 3, (uint32_t) sizeof(Pc));
    // ONE dispatch per timed batch: this kernel's top-k is k rounds over the whole row in ONE workgroup, so a
    // single dispatch is already long; batching them would just make one command buffer rival a driver timeout.
    Timing t = time_kernel(ctx, p, {&b_l, &b_h, &b_o}, &pc, (uint32_t) sizeof(pc), 1 /*groups*/, 1 /*groups_y*/,
                           1 /*batch*/, reps < 5 ? reps : 5, warmups < 2 ? warmups : 2);
    char shape[64];
    std::snprintf(shape, sizeof shape, "vocab=%d n_tokens=1", n_vocab);
    report("sampler_kernel_f32", shape, t, (double) n_vocab, 0.0);
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
// THE BF16-PROJECTION PAIR (`bf16_gemv` / `bf16_gemv_split`, ONE shared shader) - the DEFAULT side of the
// `native_bf16_projections` setting, ported so the setting cannot route the engine at an unported symbol.  There
// is NO legacy sibling to compare against (this IS the non-native branch), so the pair is measured against the
// PORTED native sibling `bf16_gemv_fp32_mmvf` (`bf16_mmvf_f32`), which the setting selects when it is ON: same
// workgroup-per-row decomposition, the only difference being the activation's precision (bf16 vs f32).  XPAIR
// convention is `<row>/<baseline>`; < 1.0 means the first row is faster, and a same-shape drop-in is EXPECTED to
// wash.  The CUDA's one-thread-per-row naive decomposition was ALSO measured and is NOT shipped: at the engine's
// shapes it is 14-18x slower on the GPUs (Arc 0.073x, Ryzen iGPU 0.103x of the workgroup form at n_out=512) and
// the CUDA does not take it above n_out=64 anyway.
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

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "shaders";
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR")) dir = e;
    int dev = -1, reps = 9, warmups = 3, sampler_vocab = 248320;
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") list = true;
        else if (a == "--spv-dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--warmups" && i + 1 < argc) warmups = std::atoi(argv[++i]);
        else if (a == "--sampler-vocab" && i + 1 < argc) sampler_vocab = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "usage: vk_bench [--spv-dir D] [--device N] [--reps R] [--warmups W] [--sampler-vocab N] [--list]\n"); return 2; }
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

    bench_gdn_conv_step(ctx, dir, reps, warmups);
    bench_gdn_l2_norm(ctx, dir, reps, warmups);
    bench_gdn_beta_gate(ctx, dir, reps, warmups);
    bench_gdn_gate(ctx, dir, reps, warmups);
    bench_gdn_step(ctx, dir, reps, warmups);
    bench_gdn_out_norm(ctx, dir, reps, warmups);

    if (ctx.info().storage_buffer_8bit) {
        bench_iq_dequant(ctx, dir, 30, "BF16", 2, 256, reps, warmups);
        bench_iq_dequant(ctx, dir, 20, "IQ4_NL", 18, 256, reps, warmups);
        bench_iq_dequant(ctx, dir, 11, "IQ2_S", 82, 256, reps, warmups);
        bench_iq_dequant(ctx, dir, 11, "IQ2_S", 82, 1024, reps, warmups);   // the SIZE-SCALING arm (4x)
    } else {
        std::printf("SKIP iq_dequant_f32            | device lacks storageBuffer8BitAccess\n");
    }

    if (ctx.info().storage_buffer_8bit) {
        bench_iq2s(ctx, dir, 512, reps, warmups);
        bench_iq2s(ctx, dir, 2048, reps, warmups);                         // the SIZE-SCALING arm (4x)
    } else {
        std::printf("SKIP iq2s_mmvq                 | device lacks storageBuffer8BitAccess\n");
    }

    bench_quantize_q8_0(ctx, dir, reps, warmups);
    if (ctx.info().storage_buffer_8bit) {
        bench_quantize_q8_1(ctx, dir, reps, warmups);
    } else {
        std::printf("SKIP quantize_q8_1             | device lacks storageBuffer8BitAccess\n");
    }
    bench_quantize_q8_K(ctx, dir, reps, warmups);

    // THE PERFORMANCE TIER, class B: native vs legacy, same shape, same device (a native/legacy ratio < 1 is
    // faster).  Before the sampler, which is the heavy one.
    bench_rope_pair(ctx, dir, reps, warmups);
    bench_router_pair(ctx, dir, reps, warmups);
    bench_moe_combine_pair(ctx, dir, reps, warmups);
    bench_rms_norm_pair(ctx, dir, reps, warmups);
    bench_qsa_gate_pair(ctx, dir, reps, warmups);   // class B: native_qsa_gate_apply <- qsa_gate_apply_f32
    // batch 2: the native GDN / DeltaNet mixer kernels (conv+SiLU, l2_norm, beta_gate).
    bench_gdn_conv_silu_pair(ctx, dir, reps, warmups);
    bench_gdn_l2_norm_pair(ctx, dir, reps, warmups);
    bench_gdn_beta_gate_pair(ctx, dir, reps, warmups);
    // batch 3: the remaining three native GDN / DeltaNet mixer kernels (gate, out_norm, step).
    bench_gdn_gate_pair(ctx, dir, reps, warmups);
    bench_gdn_out_norm_pair(ctx, dir, reps, warmups);
    bench_gdn_step_pair(ctx, dir, reps, warmups);
    // PERFORMANCE TIER, batch 4: the THREE FUSED GDN PATHS, each against the multi-dispatch chain it replaces and
    // against the non-fused native kernel(s) where a comparison exists.
    bench_fused_gdn_conv_l2_pair(ctx, dir, reps, warmups);
    bench_fused_gdn_ab_pair(ctx, dir, reps, warmups);
    bench_fused_gdn_step_norm_pair(ctx, dir, reps, warmups);
    // THE BF16-PROJECTION PAIR (`bf16_gemv` / `bf16_gemv_split`): the DEFAULT side of `native_bf16_projections`.
    bench_bf16_gemv_pair(ctx, dir, reps, warmups);

    // THE SAMPLER LAST, ON PURPOSE.  Its one-block top-k over the whole vocabulary is the port's heaviest
    // single dispatch, and on the Ryzen iGPU (RADV) the full-vocabulary shape was measured to trigger a
    // driver "hard recovery" (context lost).  Running it last means a driver reset on one ICD loses only the
    // sampler there, and the rows above are already printed.  --sampler-vocab bounds the row for a weaker
    // device; the default is the artifact's real vocabulary.
    bench_sampler(ctx, dir, sampler_vocab, reps, warmups);

    std::printf("== vk_bench done\n");
    return 0;
}
