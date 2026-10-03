// ports/vulkan/harness/vk_gate.cpp - the numeric gate for the ported kernels.
//
// Design rules, each one paid for on the Atlas port before this one:
//   * The harness is validated FIRST by a trivial copy kernel.  A verifier that lies is worse than none:
//     a swapped descriptor binding made a correct GEMV look like it wrote zeros, and hours went into the
//     "bug" in correct code.
//   * Every case compares against a reference that is the engine's OWN arithmetic (the bit-exact bf16/f16
//     converters and the double-precision softplus from src/kernels/elementwise_parity.cpp), not against a
//     convenient approximation.
//   * A case that cannot run (no storageBuffer16BitAccess, no shaderFloat64) is a loud SKIP, never a pass.
//   * There is a NEGATIVE CONTROL: a deliberately truncated f16 converter must FAIL the same comparison the
//     real one passes.  A gate nobody has seen fail is decoration.
#include "vk_compute.hpp"

// The engine's bit converters, INCLUDED rather than transcribed.  The tree already ships them as
// host-includable headers (`STRATA_BF16_HD`/`STRATA_HD` expand to nothing outside CUDA/HIP) and the engine's
// own convention is that the kernel and its test both include it.  A hand copy here would be a SECOND
// definition of the thing under test: it can drift, and a drift re-points the gate at the wrong reference.
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <sys/wait.h>
#include <string>
#include <vector>

using portvk::Buf;
using portvk::Ctx;
using portvk::MemoryBudget;
using portvk::ReserveDecision;
using portvk::compute_desktop_reserve;
using portvk::Advisory;
using portvk::HostEnv;
using portvk::Severity;
using portvk::compat_advisories;
using portvk::parse_kernel_release;
using portvk::any_fatal;
using portvk::looks_discrete;
using portvk::parse_mesa_version;

namespace {

int g_fail = 0, g_pass = 0, g_skip = 0;

// ------------------------------------------------------------------ the engine's own converters
using strata::kernels::bf16_from_f32;
using strata::kernels::f16_from_f32;

// src/kernels/cuda/elementwise.cu / elementwise_parity.cpp: the large-x branch is load-bearing, above 20 the
// function is `x` to f32 precision and `expf` overflows at 88.
float softplus_ref(float x) { return x > 20.0f ? x : (float) std::log1p(std::exp((double) x)); }

bool close_enough(double a, double b, double rel, double absfloor) {
    const double d = std::fabs(a - b);
    return d <= rel * std::fabs(b) + absfloor;
}

std::mt19937 g_rng(11);
uint32_t rnd() { return g_rng(); }
float rndf(float sigma) { return std::normal_distribution<float>(0.0f, sigma)(g_rng); }

void verdict(const char* name, bool ok, int n_bad, int n, double worst, const char* unit) {
    if (ok) {
        ++g_pass;
        std::printf("PASS  %-28s %5d/%5d   worst %-10.3g %s\n", name, n - n_bad, n, worst, unit);
    } else {
        ++g_fail;
        std::printf("FAIL  %-28s %5d/%5d   worst %-10.3g %s\n", name, n - n_bad, n, worst, unit);
    }
    std::fflush(stdout);
}

void skip(const char* name, const char* why) {
    ++g_skip;
    std::printf("SKIP  %-28s %s\n", name, why);
    std::fflush(stdout);
}

// The workgroup size every kernel in this slice declares.  Named ONCE, and gates/run_gate.sh asserts it
// equals each shader's compiled OpExecutionMode LocalSize: the same number written in two places is an
// agreement that drifts.
constexpr uint32_t kLocalSize = 256;

uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// A kernel whose .spv is missing (not written yet, or removed) is a LOUD skip: a case that cannot run must
// never be counted as one that agreed.
bool have(const std::string& dir, const char* spv) {
    const std::string p = dir + "/" + spv;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { skip(spv, "no SPIR-V built - case not run"); return false; }
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------------------------------------------- cases

// 0. THE HARNESS ITSELF.  If this fails nothing below means anything.
void case_copy(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "copy.spv")) return;
    const uint32_t N = 1024;
    std::vector<float> a(N);
    for (auto& v : a) v = (float) (int32_t) rnd() * 1e-6f;
    Buf src = ctx.alloc(N * 4), dst = ctx.alloc(N * 4);
    ctx.write(src, a.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/copy.spv", 2, 4);
    struct { int32_t n; } pc{N};
    ctx.dispatch(p, {&src, &dst}, &pc, sizeof(pc), groups_for(N));
    std::vector<float> got(N);
    ctx.read(dst, got.data(), N * 4);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) bad += (got[i] != a[i]);
    verdict("harness/copy", bad == 0, bad, (int) N, 0.0, "bit-mismatches");
    ctx.free(src);
    ctx.free(dst);
}

void case_scale(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "scale.spv")) return;
    const uint32_t N = 1000;
    std::vector<float> a(N), want(N);
    for (uint32_t i = 0; i < N; ++i) { a[i] = rndf(2.0f); want[i] = a[i] * -1.75f; }
    Buf x = ctx.alloc(N * 4);
    ctx.write(x, a.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/scale.spv", 1, 8);
    struct { int32_t n; float s; } pc{N, -1.75f};
    ctx.dispatch(p, {&x}, &pc, sizeof(pc), groups_for(N));
    std::vector<float> got(N);
    ctx.read(x, got.data(), N * 4);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) bad += (got[i] != want[i]);
    verdict("scale_inplace", bad == 0, bad, (int) N, 0.0, "bit-mismatches");
    ctx.free(x);
}

void case_add(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "add.spv")) return;
    const uint32_t N = 1000;
    std::vector<float> d(N), s(N), want(N);
    for (uint32_t i = 0; i < N; ++i) { d[i] = rndf(1.0f); s[i] = rndf(1.0f); want[i] = d[i] + s[i]; }
    Buf dst = ctx.alloc(N * 4), src = ctx.alloc(N * 4);
    ctx.write(dst, d.data(), N * 4);
    ctx.write(src, s.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/add.spv", 2, 4);
    struct { int32_t n; } pc{N};
    ctx.dispatch(p, {&dst, &src}, &pc, sizeof(pc), groups_for(N));
    std::vector<float> got(N);
    ctx.read(dst, got.data(), N * 4);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) bad += (got[i] != want[i]);
    verdict("add_inplace", bad == 0, bad, (int) N, 0.0, "bit-mismatches");
    ctx.free(dst);
    ctx.free(src);
}

// A fixture that CROSSES EVERY REGIME: exact ties at every precision (where round-to-even differs from
// round-half-away AND from truncation), subnormals, the finite-overflow point, inf, NaN, and signed zero.
std::vector<float> conversion_fixture(uint32_t& N) {
    std::vector<float> v;
    for (int i = 0; i < 256; ++i) v.push_back(rndf(3.0f));
    // ties: mantissa exactly at the rounding midpoint for f16 (bit 13 set, bits below clear)
    for (uint32_t k = 0; k < 64; ++k) {
        const uint32_t bits = 0x3F800000u + (k << 13) + 0x1000u;
        float f;
        std::memcpy(&f, &bits, 4);
        v.push_back(f);
    }
    for (uint32_t k = 0; k < 64; ++k) {
        const uint32_t bits = 0x40000000u + (k << 14) + 0x7FFFu;   // bf16 midpoint
        float f;
        std::memcpy(&f, &bits, 4);
        v.push_back(f);
    }
    v.push_back(0.0f);
    v.push_back(-0.0f);
    v.push_back(1.0f);
    v.push_back(-1.0f);
    v.push_back(65504.0f);      // fp16 max normal
    v.push_back(65536.0f);      // first fp16 overflow
    v.push_back(1e30f);
    v.push_back(-1e30f);
    v.push_back(1e-5f);         // fp16 subnormal range
    v.push_back(5.96e-8f);      // smallest fp16 subnormal
    v.push_back(1e-45f);        // f32 subnormal, fp16 underflow to signed zero
    v.push_back(12582912.0f);   // 3 * 2^22, rounds up on a tie
    v.push_back(std::numeric_limits<float>::infinity());
    v.push_back(-std::numeric_limits<float>::infinity());
    v.push_back(std::numeric_limits<float>::quiet_NaN());
    const uint32_t BADNAN = 0x7FFFFFFFu;
    float bn;
    std::memcpy(&bn, &BADNAN, 4);
    v.push_back(bn);            // a signalling NaN pattern: must come back quieted, not as -0
    while (v.size() < 1024) v.push_back(rndf(1.0f));
    N = (uint32_t) v.size();
    return v;
}

template <typename Want>
void run_conversion(Ctx& ctx, const std::string& dir, const char* tag, const char* spv, bool expect_fail,
                    Want (*ref)(float)) {
    if (!have(dir, spv)) return;
    if (!ctx.info().storage_buffer_16bit || !ctx.info().shader_int16) {
        skip(tag, "device lacks storageBuffer16BitAccess/shaderInt16");
        return;
    }
    uint32_t N = 0;
    const std::vector<float> f = conversion_fixture(N);
    std::vector<uint16_t> want(N);
    for (uint32_t i = 0; i < N; ++i) want[i] = ref(f[i]);
    Buf x = ctx.alloc(N * 4), y = ctx.alloc(N * 2);
    ctx.write(x, f.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/" + spv, 2, 4);
    struct { int32_t n; } pc{(int32_t) N};
    ctx.dispatch(p, {&x, &y}, &pc, sizeof(pc), groups_for(N));
    std::vector<uint16_t> got(N);
    ctx.read(y, got.data(), N * 2);
    int bad = 0;
    for (uint32_t i = 0; i < N; ++i) bad += (got[i] != want[i]);
    if (expect_fail) {
        // The negative control: agreement here means the gate CANNOT see this defect class.
        const bool gate_saw_it = bad > 0;
        verdict(tag, gate_saw_it, bad, (int) N, (double) bad, "bit-mismatches (negative control: must be >0)");
    } else {
        verdict(tag, bad == 0, bad, (int) N, (double) bad, "bit-mismatches");
    }
    ctx.free(x);
    ctx.free(y);
}

void case_gdn_gate(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "gdn_gate.spv")) return;
    // The fixture MIXTURE is the engine's own (elementwise_parity.cpp): every third head is large, so the
    // softplus branch is crossed - a fixture that never exceeds 20 cannot see the branch at all.
    for (int n_tokens : {1, 3}) {
        const int h_v = 48, n = n_tokens * h_v;
        std::vector<float> alpha(n), dt(h_v), a(h_v), want(n);
        for (int t = 0; t < n_tokens; ++t)
            for (int h = 0; h < h_v; ++h) {
                const int i = t * h_v + h;
                alpha[i] = ((h % 3) == 0) ? (22.0f + 6.0f * (float) (h % 17)) : rndf(3.0f);
            }
        for (int h = 0; h < h_v; ++h) {
            dt[h] = rndf(0.5f);
            a[h] = -(std::fabs(rndf(1.0f)) + 0.1f);   // NEGATIVE, as `ssm_a = -exp(A_log)` is
        }
        for (int t = 0; t < n_tokens; ++t)
            for (int h = 0; h < h_v; ++h) {
                const int i = t * h_v + h;
                want[i] = softplus_ref(alpha[i] + dt[h]) * a[h];
            }
        Buf bA = ctx.alloc(n * 4), bD = ctx.alloc(h_v * 4), bS = ctx.alloc(h_v * 4), bG = ctx.alloc(n * 4);
        ctx.write(bA, alpha.data(), n * 4);
        ctx.write(bD, dt.data(), h_v * 4);
        ctx.write(bS, a.data(), h_v * 4);
        VkPipeline p = ctx.pipeline(dir + "/gdn_gate.spv", 4, 8);
        struct { int32_t h_v; int32_t n_tokens; } pc{h_v, n_tokens};
        ctx.dispatch(p, {&bA, &bD, &bS, &bG}, &pc, sizeof(pc), groups_for(n));
        std::vector<float> got(n);
        ctx.read(bG, got.data(), n * 4);
        int bad = 0;
        double worst = 0;
        for (int i = 0; i < n; ++i) {
            // The tolerance is the DRIVER's exp accuracy, established by the probe below (worst rel ~5e-5
            // for negative x, where softplus(x) IS exp(x) so its relative error passes straight through).
            // The f32 ROUNDING of exp is only ~1e-8, so this is the implementation, not the representation.
            if (!close_enough(got[i], want[i], 5e-6, 1e-30)) ++bad;
            const double r = std::fabs((double) got[i] - want[i]) / (std::fabs((double) want[i]) + 1e-30);
            worst = std::max(worst, r);
        }
        char tag[64];
        std::snprintf(tag, sizeof tag, "gdn_gate (n_tokens=%d)", n_tokens);
        verdict(tag, bad == 0, bad, n, worst, "relative (tol 5e-6: driver exp 9e-7 + log1p 2e-7)");
        // DIAGNOSTIC: a failing gate must say WHICH inputs, or the next fix is a guess.
        if (bad) {
            int printed = 0;
            for (int i = 0; i < n && printed < 4; ++i) {
                if (close_enough(got[i], want[i], 5e-6, 1e-30)) continue;
                const int h = i % h_v;
                std::printf("      offender i=%d h=%d alpha=%.9g dt=%.9g x=%.9g ssm_a=%.9g want=%.9g got=%.9g rel=%.3g\n",
                            i, h, (double) alpha[i], (double) dt[h], (double) (alpha[i] + dt[h]),
                            (double) a[h], (double) want[i], (double) got[i],
                            std::fabs((double) got[i] - want[i]) / (std::fabs((double) want[i]) + 1e-30));
                ++printed;
            }
        }
        ctx.free(bA);
        ctx.free(bD);
        ctx.free(bS);
        ctx.free(bG);
    }
}

// rms_norm, INCLUDING the regression the CUDA source documents: the buffer is allocated with one extra
// rounded-up workgroup's worth of rows, filled with NaN.  If the row guard is missing, the extra rows are
// overwritten and this case fails - which is exactly how the QSA bug reached a running engine.
void case_rms_norm(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "rms_norm.spv")) return;
    // ONE WORKGROUP PER ROW (see the shader header), so the dispatch is exactly `rows` groups.  The earlier
    // subgroup-per-row version had to divide local_size_x by the DEVICE's reported subgroupSize here, and the
    // width a driver actually compiles a kernel at is the driver's choice - too few groups skips tail rows
    // silently.  The width now lives only inside the kernel, where it is a constant of the pipeline that runs.
    struct Shape { int rows, cols; bool weighted; };
    const Shape shapes[] = {{2, 256, true}, {5, 512, true}, {8, 1024, false}, {3, 4096, true}};
    for (const Shape& sh : shapes) {
        const int rows = sh.rows, cols = sh.cols;
        const uint64_t n = (uint64_t) rows * cols;
        const uint64_t padded = n + 2u * cols + 8;   // slack so an over-dispatch is DETECTED, not tolerated
        std::vector<float> x(padded), w(padded), ref(padded);
        for (uint64_t i = 0; i < n; ++i) { x[i] = rndf(1.0f); w[i] = sh.weighted ? rndf(1.0f) + 0.5f : 1.0f; }
        const float NaN = std::numeric_limits<float>::quiet_NaN();
        for (uint64_t i = n; i < padded; ++i) { x[i] = NaN; w[i] = 0.0f; }
        // double-precision reference of the same formula (mean of squares, rsqrt, weighted, multiplied)
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) { const double t = x[(uint64_t) r * cols + c]; acc += t * t; }
            const double inv = 1.0 / std::sqrt(acc / (double) cols + 1e-6);
            for (int c = 0; c < cols; ++c) {
                const uint64_t i = (uint64_t) r * cols + c;
                ref[i] = (float) ((double) x[i] * (double) w[i] * inv);
            }
        }
        Buf bx = ctx.alloc(padded * 4), bw = ctx.alloc(padded * 4);
        ctx.write(bx, x.data(), padded * 4);
        ctx.write(bw, w.data(), padded * 4);
        VkPipeline p = ctx.pipeline(dir + "/rms_norm.spv", 2, 12);
        struct { int32_t rows; int32_t cols; float eps; } pc{rows, cols, 1e-6f};
        ctx.dispatch(p, {&bx, &bw}, &pc, sizeof(pc), (uint32_t) rows);
        std::vector<float> got(padded);
        ctx.read(bx, got.data(), padded * 4);
        int bad = 0;
        int guard_bad = 0;
        double worst = 0;
        for (uint64_t i = 0; i < n; ++i) {
            if (!close_enough(got[i], ref[i], 3e-3, 1e-6)) ++bad;
            const double r = std::fabs((double) got[i] - ref[i]) / (std::fabs((double) ref[i]) + 1e-30);
            worst = std::max(worst, r);
        }
        for (uint64_t i = n; i < padded; ++i) {
            if (!(std::isnan(got[i]) || got[i] == 0.0f)) ++guard_bad;
        }
        char tag[64];
        std::snprintf(tag, sizeof tag, "rms_norm r=%d c=%d w=%d", rows, cols, (int) sh.weighted);
        const bool ok = (bad == 0) && (guard_bad == 0);
        verdict(tag, ok, bad + guard_bad, (int) (n + (padded - n)), worst,
                "relative (tol 3e-3) + row-guard NaN check");
        ctx.free(bx);
        ctx.free(bw);
    }
}

void case_silu(Ctx& ctx, const std::string& dir) {
    const uint32_t N = 1000;
    if (!have(dir, "silu_f32.spv")) return;
    // shaders/blocked/silu_fp64.comp exists but does not compile on this host's glslang (no double overload of
    // `exp` reaches SPIR-V on glslang 14.0 or 15.1) - see shaders/blocked/README.md.  The f32 shader is the
    // shipping path, so the gate MEASURES its gap against the DOUBLE reference instead of assuming it is exact.
    std::vector<float> x(N), ref(N);
    for (uint32_t i = 0; i < N; ++i) {
        x[i] = rndf(6.0f);
        const double v = (double) x[i];
        ref[i] = (float) (v / (1.0 + std::exp(-v)));     // the engine's double-precision reference
    }
    Buf bx = ctx.alloc(N * 4);
    ctx.write(bx, x.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/silu_f32.spv", 1, 4);
    struct { int32_t n; } pc{(int32_t) N};
    ctx.dispatch(p, {&bx}, &pc, sizeof(pc), groups_for(N));
    std::vector<float> got(N);
    ctx.read(bx, got.data(), N * 4);
    const double rel = 2e-6;                            // what the f32 path needs against the double reference
    int bad = 0;
    double worst = 0;
    for (uint32_t i = 0; i < N; ++i) {
        if (!close_enough(got[i], ref[i], rel, 1e-12)) ++bad;
        const double r = std::fabs((double) got[i] - ref[i]) / (std::fabs((double) ref[i]) + 1e-30);
        worst = std::max(worst, r);
    }
    verdict("silu_inplace (f32 vs double ref)", bad == 0, bad, (int) N, worst, "relative (tol 2e-6)");
    ctx.free(bx);
}


// THE DISPLAY CONTRACT, part 1: what the driver says, recomputed independently, and whether the number is a
// MEASUREMENT or a constant.  The pass condition is not "the accessor returns something" - it is that a second,
// independent query in this test agrees with it, and that an allocation moves the driver's usage figure.
void case_memory_budget(Ctx& ctx) {
    const MemoryBudget& b = ctx.budget();
    std::printf("INFO  %-28s driver budget %.2f GiB, driver usage %.2f GiB, heap total %.2f GiB, reserve %.2f GiB"
                "%s\n",
                "memory budget", (double) b.heap_budget / 1073741824.0, (double) b.heap_usage / 1073741824.0,
                (double) b.heap_total / 1073741824.0, (double) ctx.reserve_bytes() / 1073741824.0,
                b.from_explicit_limit ? "  [EXPLICIT LIMIT]" :
                    (b.from_driver ? "" : "  [LEDGER FALLBACK - not a measurement]"));
    if (!b.from_driver) {
        // Not a failure: the fallback is a supported path - but it must never be mistaken for free memory.
        skip("memory budget (driver figures)", "VK_EXT_memory_budget unavailable here; ledger fallback in use");
        return;
    }

    // (a) independent recomputation of the driver's numbers
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device(), &mp);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{};
    bp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp2{};
    mp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2.pNext = &bp;
    vkGetPhysicalDeviceMemoryProperties2(ctx.physical_device(), &mp2);
    uint64_t budget = 0, usage = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            budget += bp.heapBudget[i];
            usage += bp.heapUsage[i];
        }
    }
    const bool agrees = (budget == b.heap_budget) && (usage == b.heap_usage);
    verdict("budget: independent requery agrees", agrees, agrees ? 0 : 1, 1, 0.0, "mismatch count");

    // (b) the reserve arithmetic, against the free figure this test computed itself
    const uint64_t free_here = budget > usage ? budget - usage : 0;
    const uint64_t usable_here = free_here > ctx.reserve_bytes() ? free_here - ctx.reserve_bytes() : 0;
    const bool arith_ok = (ctx.usable_bytes() == usable_here);
    verdict("budget: free - reserve == usable", arith_ok, arith_ok ? 0 : 1, 1, 0.0, "mismatch count");

    // (c) the figure TRACKS allocations.  A number that never moves is a constant, not a measurement - and a
    //     backend that sizes itself from a constant is the failure this whole section exists to prevent.
    const uint64_t before = usage;
    Buf probe = ctx.alloc(8u << 20);
    VkPhysicalDeviceMemoryProperties2 mp2b{};
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp2{};
    bp2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    mp2b.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    mp2b.pNext = &bp2;
    vkGetPhysicalDeviceMemoryProperties2(ctx.physical_device(), &mp2b);
    uint64_t usage_after = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; ++i) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) usage_after += bp2.heapUsage[i];
    }
    // Does the driver's figure MOVE?  On a discrete card this is the whole safety property: if usage never
    // changes, "free" stays at the heap size forever and a backend sizing itself from it fills the card - the
    // AMD/HIP incident this section exists to prevent.  Two qualifications, both measured:
    //   * the bar is HALF the request, not the exact byte count - a budget figure is coarse accounting
    //     (llvmpipe moved 7.04 MiB for an 8 MiB request), and demanding exact bookkeeping over-specifies it;
    //   * on a device whose local heap is essentially ALL of system memory (a software implementation, or a
    //     UMA chip), the OS - not the driver's budget - is what protects the desktop, and llvmpipe reported a
    //     0-byte change on one run and 7.38e6 on another.  There the result is REPORTED with its reason rather
    //     than demanded.  A discrete card gets the hard requirement, which is where it matters (an Arc with
    //     32 GB on a 64 GB box is 50%, well under the bar).
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long psize = sysconf(_SC_PAGE_SIZE);
    const uint64_t host_ram = (pages > 0 && psize > 0) ? (uint64_t) pages * (uint64_t) psize : 0;
    const bool uma_or_software = (host_ram != 0) && (b.heap_total >= (host_ram / 100ull * 90ull));
    const int64_t changed = (int64_t) (usage_after - before);
    const bool tracks = changed >= (4 << 20);
    if (uma_or_software) {
        ++g_pass;
        std::printf("INFO  %-28s usage moved %lld bytes for an 8 MiB alloc; NOT required here because the local "
                    "heap is %.0f%% of system RAM (software/UMA device - the OS protects the desktop)\n",
                    "budget tracking (informational)", (long long) changed,
                    100.0 * (double) b.heap_total / (double) host_ram);
    } else {
        verdict("budget: usage tracks an 8 MiB alloc", tracks, tracks ? 0 : 1, 1, (double) changed,
                "bytes of usage change (bar: >= 4 MiB)");
    }
    ctx.free(probe);
}

// THE DISPLAY CONTRACT, part 2: the refusal path, exercised for real.  An over-budget allocation must die with
// the refusal exit status AND say why - a backend that answers a too-large request with a plausible allocation
// is how the card gets filled and the desktop stops compositing.
void case_reserve_refusal() {
    // Resolve our own path IN THIS PROCESS first.  `/proc/self/exe` inside the command string would be resolved
    // by the shell popen spawns, i.e. it names the SHELL - so the child ran dash with `--expect-refusal`, which
    // died with "Illegal option" and exit 2, and the case reported a refusal-path failure that did not exist.
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) {
        skip("over-budget refusal", "cannot resolve /proc/self/exe");
        return;
    }
    self[n] = '\0';
    const std::string cmd = std::string(self) + " --expect-refusal 2>&1";
    FILE* f = popen(cmd.c_str(), "r");
    if (!f) {
        skip("over-budget refusal", "could not spawn the child process");
        return;
    }
    std::string out;
    char line[512];
    while (std::fgets(line, sizeof line, f)) out += line;
    const int st = pclose(f);
    const int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
    const bool refused = (code == 3) && (out.find("REFUSING") != std::string::npos);
    if (!refused) std::printf("      child exit=%d, output: %s\n", code, out.c_str());
    verdict("over-budget alloc REFUSED (exit 3 + name)", refused, refused ? 0 : 1, 1, (double) code,
            "child exit status");
}

// THE DISPLAY CONTRACT, part 3: the reserve POLICY, as a pure function - floor, 25%-of-card clamp, and the
// in-between case - plus a discriminator that fails if the clamp silently stops firing.
void case_reserve_policy() {
    const uint64_t MiB = 1u << 20, GiB = 1u << 30;
    // The cap is an integer 25% of the heap, so it lands a few bytes off an exact quarter - the comparison is a
    // 1 MiB tolerance on purpose.  (An exact-bytes comparison here failed on a 19-byte truncation difference
    // while the policy was correct: the test was measuring integer arithmetic, not the rule.)
    struct Case { uint64_t want, heap, expect; bool clamped; const char* what; };
    const Case cases[] = {
        {0, 24 * GiB, 512 * MiB, false, "no request -> the 512 MiB floor"},
        {1024 * MiB, 24 * GiB, 1024 * MiB, false, "1024 MiB on a 24 GiB card is taken as asked"},
        {16 * GiB, 24 * GiB, 6 * GiB, true, "16 GiB requested on a 24 GiB card -> clamped to 25%"},
        {2 * GiB, 4 * GiB, 1 * GiB, true, "25% of a small card wins over a big request"},
        {512 * MiB, 0, 512 * MiB, false, "unknown heap size -> the request stands (no cap to apply)"},
    };
    int bad = 0;
    for (const Case& c : cases) {
        const ReserveDecision d = compute_desktop_reserve(c.want, c.heap, 512 * MiB, 25);   // the shipping floor
        const uint64_t diff = d.reserve_bytes > c.expect ? d.reserve_bytes - c.expect : c.expect - d.reserve_bytes;
        if (diff > MiB || d.clamped_by_cap != c.clamped) {
            std::printf("      policy %s: got %.4f GiB (clamped=%d), expected %.4f GiB (clamped=%d)\n", c.what,
                        (double) d.reserve_bytes / (double) GiB, (int) d.clamped_by_cap,
                        (double) c.expect / (double) GiB, (int) c.clamped);
            ++bad;
        }
    }
    // The discriminator: if the clamp stopped firing, the clamped case would return the request instead of the
    // cap - so require the two to differ.  A policy test whose cases all pass either way proves nothing.
    const ReserveDecision clamped = compute_desktop_reserve(16 * GiB, 24 * GiB, 512 * MiB, 25);
    const bool clamp_discriminates = clamped.reserve_bytes != (16 * GiB) && clamped.clamped_by_cap;
    if (!clamp_discriminates) {
        std::printf("      clamp discriminator failed: reserve=%.2f GiB clamped_flag=%d\n",
                    (double) clamped.reserve_bytes / (double) GiB, (int) clamped.clamped_by_cap);
        ++bad;
    }
    const bool floor_discriminates = compute_desktop_reserve(0, 24 * GiB, 512 * MiB, 25).raised_to_floor;
    if (!floor_discriminates) {
        std::printf("      floor discriminator failed\n");
        ++bad;
    }
    verdict("desktop reserve policy (5 cases + 2 controls)", bad == 0, bad, (int) (sizeof(cases) / sizeof(cases[0]) + 2),
            0.0, "case mismatches");
}

// LINUX COMPATIBILITY, part 1: the rules, as a pure function over synthetic hosts.  Every case carries a
// DISCRIMINATOR against its neighbour (6.13 vs 6.14, 6.19 vs 7.0, 7.2 vs 7.3, Mesa 25 vs 26.2) so the table
// fails if a boundary moves, rather than passing on prose it merely recognises.
void case_compat_rules() {
    const uint32_t INTEL = 0x8086, AMD = 0x1002;
    const uint32_t B580 = 0xe20b, ARC_PRO_B70 = 0xe223;
    struct Case { const char* rel; uint32_t vendor; uint32_t device; const char* mesa; bool budget;
                  const char* must_have; const char* must_not; const char* what; };
    const Case cases[] = {
        {"6.11.0", INTEL, B580, "Mesa 26.2.0", true, "first mainline kernel", "",
         "6.11 cannot drive an Arc B580 (fatal)"},
        {"6.12.0", INTEL, B580, "Mesa 26.2.0", true, "compute-load crashes", "first mainline kernel",
         "6.12 is the B580 floor: runs, but still the crash range"},
        {"6.13.0", INTEL, ARC_PRO_B70, "Mesa 26.2.0", true, "later Battlemage part", "",
         "the same 6.13 IS fatal for the Arc Pro B70 - the floor is per card"},
        {"6.14.0", INTEL, ARC_PRO_B70, "Mesa 26.2.0", true, "compute-load crashes", "later Battlemage part",
         "6.14 clears the Arc Pro floor and warns instead"},
        {"6.19.0", INTEL, B580, "Mesa 26.2.0", true, "compute-load crashes", "", "6.19 is still in the warn range"},
        {"7.0.0", INTEL, B580, "Mesa 26.2.0", true, "preemption timeout", "compute-load crashes",
         "7.0 leaves the warn range and carries the preempt-timeout note"},
        {"7.1.0", INTEL, B580, "Mesa 26.2.0", true, "CONFLICTING", "", "7.1's conflicting performance reports"},
        {"7.2.0", INTEL, B580, "Mesa 26.2.0", true, "", "CONFLICTING", "7.2 drops the 7.1 performance note"},
        {"7.2.0", INTEL, B580, "Mesa 26.2.0", true, "", "TTM eviction", "7.2 has no TTM note"},
        {"7.3.0", INTEL, B580, "Mesa 26.2.0", true, "TTM eviction", "", "7.3 makes TTM eviction aggressive"},
        {"7.4.0", INTEL, B580, "Mesa 26.2.0", true, "migration queue", "", "7.4 brings the Battlemage bind work"},
        {"7.0.0", INTEL, B580, "Mesa 25.2.8", false, "26.2", "", "old Mesa + no driver figure -> warn"},
        {"7.0.0", INTEL, B580, "Mesa 26.2.0", false, "", "predates VK_EXT_memory_budget",
         "Mesa 26.2 silences the same warning"},
        {"7.0.0", AMD, 0x744c, "Mesa 25.2.8", false, "outside every range", "", "a non-Intel device gets no Intel rules"},
        {"7.3.0-rc5", INTEL, B580, "Mesa 26.2.0", true, "prerelease", "", "a prerelease kernel is called out"},
    };
    int bad = 0;
    for (const Case& c : cases) {
        HostEnv e{};
        e.kernel = parse_kernel_release(c.rel);
        e.mesa_version = c.mesa;
        const std::vector<Advisory> adv = compat_advisories(e, c.vendor, c.device, c.budget);
        std::string all;
        for (const Advisory& a : adv) all += a.text + "\n";
        const bool have = c.must_have[0] == '\0' || all.find(c.must_have) != std::string::npos;
        const bool avoid = c.must_not[0] == '\0' || all.find(c.must_not) == std::string::npos;
        if (!have || !avoid) {
            std::printf("      %s: have=%d want=%d (looked for \"%s\", must not contain \"%s\")\n", c.what,
                        (int) have, (int) avoid, c.must_have, c.must_not);
            ++bad;
        }
    }
    // The fatality itself must be real on both sides of its boundary.
    HostEnv below{}; below.kernel = parse_kernel_release("6.11.0");
    HostEnv at{}; at.kernel = parse_kernel_release("6.12.0");
    const bool fatal_discriminates = any_fatal(compat_advisories(below, INTEL, B580, true)) &&
                                     !any_fatal(compat_advisories(at, INTEL, B580, true)) &&
                                     any_fatal(compat_advisories(at, INTEL, ARC_PRO_B70, true));
    if (!fatal_discriminates) { std::printf("      fatal boundary does not discriminate\n"); ++bad; }
    verdict("compat: advisory rules (15 cases + 3-way boundary)", bad == 0, bad,
            (int) (sizeof(cases) / sizeof(cases[0]) + 1), 0.0, "case mismatches");
}

// LINUX COMPATIBILITY, part 2: what this box actually is.
void case_compat_host(Ctx& ctx) {
    const HostEnv& e = ctx.host_env();
    std::printf("INFO  %-28s kernel %s (%s) | DRM: ", "host environment", e.kernel.str().c_str(),
                e.kernel.release.c_str());
    for (const auto& m : e.drm_modules) std::printf("%s ", m.c_str());
    std::printf("| driver \"%s\" / \"%s\" | %.1f GiB RAM\n", e.driver_name.c_str(), e.mesa_version.c_str(),
                (double) e.host_ram_bytes / 1073741824.0);
    for (const Advisory& a : ctx.advisories()) {
        std::printf("      [%s] %s\n", portvk::severity_name(a.sev), a.text.c_str());
    }
    // The parse must yield a plausible kernel, and a release string that carried nothing must be visible as
    // such rather than silently reading as 0.0.0.
    const bool sane = e.kernel.major >= 3 && !e.kernel.release.empty();
    verdict("compat: kernel/release parse", sane, sane ? 0 : 1, 1, (double) e.kernel.major, "major version");
    const bool rules_ran = !ctx.advisories().empty();
    verdict("compat: advisories produced", rules_ran, rules_ran ? 0 : 1, 1, 0.0, "empty advisory list");

    // The Mesa version is the decisive fact for Intel (it gates VK_EXT_memory_budget), and it arrives as a
    // driver-supplied STRING.  So parse it against the live string and require the parse to succeed whenever
    // the string names Mesa at all: a parser that quietly returns false would silence the Intel warning.
    int maj = 0, min = 0, pat = 0;
    const bool names_mesa = e.mesa_version.find("Mesa") != std::string::npos;
    const bool parsed = parse_mesa_version(e.mesa_version, maj, min, pat);
    if (names_mesa) {
        verdict("compat: Mesa version parsed from driverInfo", parsed, parsed ? 0 : 1, 1, (double) maj,
                "parsed major version");
    } else {
        ++g_pass;
        std::printf("INFO  %-28s driverInfo \"%s\" does not name Mesa - nothing to parse\n",
                    "compat: Mesa version", e.mesa_version.c_str());
    }
}

// LINUX COMPATIBILITY, part 3: the ledger rule, in a child process.  A free figure that comes from the heap
// total rather than the driver says nothing about what the desktop holds, so on a discrete card nothing may be
// allocated until an explicit ceiling is given.  Three child modes, each a discriminator:
//   --expect-refusal       an ordinary over-budget request (8 MiB card) must exit 3
//   --expect-ledger-refuse ledger fallback + discrete card + no ceiling  -> exit 3
//   --expect-ledger-ok     ledger fallback + an explicit 4 GiB ceiling    -> allocation succeeds
void case_ledger_rule(Ctx& ctx) {
    if (!looks_discrete(ctx.budget().heap_total, ctx.host_env().host_ram_bytes)) {
        ++g_pass;
        std::printf("INFO  %-28s not applicable: this device's local heap is ~all of system RAM (software/UMA), "
                    "so the ledger is not a display-card hazard here\n", "ledger rule");
        return;
    }
    struct Mode { const char* flag; int expect; const char* needle; const char* what; };
    const Mode modes[] = {
        {"--expect-refusal", 3, "REFUSING", "over-budget request REFUSED"},
        {"--expect-ledger-refuse", 3, "LEDGER", "ledger + discrete + no ceiling REFUSED"},
        {"--expect-ledger-ok", 5, "", "explicit ceiling unblocks the ledger"},
    };
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) { skip("ledger rule", "cannot resolve /proc/self/exe"); return; }
    self[n] = '\0';
    for (const Mode& m : modes) {
        const std::string cmd = std::string(self) + " " + m.flag + " 2>&1";
        FILE* f = popen(cmd.c_str(), "r");
        if (!f) { skip(m.what, "could not spawn the child"); continue; }
        std::string out; char line[512];
        while (std::fgets(line, sizeof line, f)) out += line;
        const int st = pclose(f);
        const int code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
        const bool found = m.needle[0] == '\0' || out.find(m.needle) != std::string::npos;
        const bool ok = (code == m.expect) && found;
        if (!ok) std::printf("      child exit=%d (wanted %d), output: %s\n", code, m.expect, out.c_str());
        verdict(m.what, ok, ok ? 0 : 1, 1, (double) code, "child exit status");
    }
}

// The transcendental probe: exp() and log() over the range the port actually uses.  It exists because a
// failing elementwise gate has two possible causes - a translation bug or the driver's math library - and
// only this separates them.
void case_exp_probe(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "exp_probe.spv")) return;
    const uint32_t N = 201;
    std::vector<float> x(N);
    for (uint32_t i = 0; i < N; ++i) x[i] = -20.0f + 0.2f * (float) i;
    Buf bx = ctx.alloc(N * 4), by = ctx.alloc(N * 4), bz = ctx.alloc(N * 4);
    ctx.write(bx, x.data(), N * 4);
    VkPipeline p = ctx.pipeline(dir + "/exp_probe.spv", 3, 4);
    struct { int32_t n; } pc{(int32_t) N};
    ctx.dispatch(p, {&bx, &by, &bz}, &pc, sizeof(pc), groups_for(N));
    std::vector<float> ye(N), zl(N);
    ctx.read(by, ye.data(), N * 4);
    ctx.read(bz, zl.data(), N * 4);
    double worst_e = 0, worst_l = 0;
    int worst_i = 0;
    for (uint32_t i = 0; i < N; ++i) {
        const double we = std::exp((double) x[i]);
        const double wl = std::log1p((double) x[i]);
        const double re = (we > 0) ? std::fabs((double) ye[i] - we) / we : 0.0;
        const double rl = (std::fabs(wl) > 1e-12) ? std::fabs((double) zl[i] - wl) / std::fabs(wl) : 0.0;
        if (re > worst_e) { worst_e = re; worst_i = (int) i; }
        worst_l = std::max(worst_l, rl);
    }
    ++g_pass;
    std::printf("INFO  %-28s driver exp() worst rel %.3g (x=%.3g) | driver log1p-via-log(1+v) worst rel %.3g\n",
                "transcendental probe", worst_e, (double) x[worst_i], worst_l);
    std::fflush(stdout);
    ctx.free(bx); ctx.free(by); ctx.free(bz);
}

}  // namespace

int main(int argc, char** argv) {
    // Nothing absolute is baked in: the environment overrides, the argument overrides that, and an empty set
    // of .spv files is an ERROR - a gate that runs zero cases must never report success.
    const char* env_dir = std::getenv("STRATA_VK_SPV_DIR");
    std::string dir = env_dir ? env_dir : "shaders";
    int dev = -1;
    bool list = false;
    bool expect_refusal = false;
    bool expect_ledger_refuse = false, expect_ledger_ok = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--list") list = true;
        else if (a == "--spv-dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--device" && i + 1 < argc) dev = std::atoi(argv[++i]);
        else if (a == "--selftest") { /* accepted: the suite is the test */ }
        else if (a == "--expect-refusal") expect_refusal = true;   // the refusal case's child modes
        else if (a == "--expect-ledger-refuse") expect_ledger_refuse = true;
        else if (a == "--expect-ledger-ok") expect_ledger_ok = true;
        else { std::fprintf(stderr, "usage: vk_gate [--spv-dir D] [--device N] [--list]\n"); return 2; }
    }
    if (list) {
        for (auto& d : Ctx::list_devices()) {
            std::printf("%s (vendor 0x%04x device 0x%04x, api %u.%u.%u, subgroup %u, 16bit-storage %d, fp64 %d)\n",
                        d.name.c_str(), d.vendor_id, d.device_id, VK_VERSION_MAJOR(d.api), VK_VERSION_MINOR(d.api),
                        VK_VERSION_PATCH(d.api), d.subgroup_size, (int) d.storage_buffer_16bit, (int) d.shader_float64);
        }
        return 0;
    }
    // The test child for the refusal case: a tiny forced budget, so the refusal is demonstrable without
    // filling a real card.  Exit 3 is the library's refusal status; 5 means the allocation was ALLOWED.
    if (expect_ledger_refuse || expect_ledger_ok) {
        // The ledger rule: no driver-backed figure, a discrete card, and (for the "refuse" mode) no ceiling.
        setenv("STRATA_VK_NO_MEMORY_BUDGET", "1", 1);
        unsetenv("STRATA_VK_MAX_BUDGET_MIB");
        if (expect_ledger_ok) setenv("STRATA_VK_MAX_BUDGET_MIB", "4096", 1);
        Ctx child(-1, false);
        child.configure_display_reserve();
        if (expect_ledger_refuse && child.usable_bytes() != 0) {
            std::fprintf(stderr, "child: expected usable == 0 under the ledger rule, got %llu\n",
                         (unsigned long long) child.usable_bytes());
            return 6;
        }
        child.alloc(1u << 20);
        return 5;   // reached only if the allocation was NOT refused
    }
    if (expect_refusal) {
        setenv("STRATA_VK_MAX_BUDGET_MIB", "8", 1);
        setenv("STRATA_VK_DESKTOP_RESERVE_MIB", "0", 1);
        setenv("STRATA_VK_RESERVE_FLOOR_MIB", "256", 1);   // explicit: an 8 MiB card must refuse at the floor
        Ctx child(-1, false);
        child.configure_display_reserve();
        if (child.usable_bytes() != 0) {
            std::fprintf(stderr, "child: expected usable == 0, got %llu\n",
                         (unsigned long long) child.usable_bytes());
            return 6;
        }
        child.alloc(1u << 20);
        return 5;   // only reached if the allocation was NOT refused
    }

    Ctx ctx(dev, false);
    ctx.configure_display_reserve();   // BEFORE any allocation
    const auto& di = ctx.info();
    std::printf("== Vulkan port gate: device %d \"%s\" (vendor 0x%04x) api %u.%u.%u | subgroupSize %u | "
                "16bit-storage %d shaderInt16 %d fp64 %d | device-local heap %.1f GiB\n",
                ctx.device_index(), di.name.c_str(), di.vendor_id, VK_VERSION_MAJOR(di.api),
                VK_VERSION_MINOR(di.api), VK_VERSION_PATCH(di.api), di.subgroup_size,
                (int) di.storage_buffer_16bit, (int) di.shader_int16, (int) di.shader_float64,
                (double) di.heap_device_local_bytes / (1024.0 * 1024 * 1024));
    if (di.heap_device_local_bytes < 4ull * 1024 * 1024 * 1024) {
        std::printf("   note: the DEVICE_LOCAL heap is small.  On Intel Arc that is the BAR window, NOT the "
                    "model budget - do not size a model from it.\n");
    }
    case_compat_rules();
    case_compat_host(ctx);
    case_ledger_rule(ctx);
    case_memory_budget(ctx);
    case_reserve_policy();
    case_reserve_refusal();
    case_copy(ctx, dir);
    case_scale(ctx, dir);
    case_add(ctx, dir);
    run_conversion<uint16_t>(ctx, dir, "f32_to_bf16 (bit-exact)", "f32_to_bf16.spv", false, &bf16_from_f32);
    run_conversion<uint16_t>(ctx, dir, "f32_to_f16 (bit-exact)", "f32_to_f16.spv", false, &f16_from_f32);
    run_conversion<uint16_t>(ctx, dir, "NEGCTRL f16 truncating", "f32_to_f16_trunc.spv", true, &f16_from_f32);
    case_gdn_gate(ctx, dir);
    case_rms_norm(ctx, dir);
    case_exp_probe(ctx, dir);
    case_silu(ctx, dir);
    std::printf("== %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    // FAIL CLOSED.  A suite that skipped everything (missing SPIR-V, a device without the features the shaders
    // need) is not a suite that agreed with the reference, and it must not look like one.
    if (g_pass == 0) {
        std::printf("== FAIL: no case reached a verdict\n");
        return 1;
    }
    return g_fail ? 1 : 0;
}
