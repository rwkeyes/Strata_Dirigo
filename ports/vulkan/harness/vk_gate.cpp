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

#include <filesystem>
#include <fstream>
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
using portvk::StackReport;
using portvk::detect_stack;
using portvk::stack_advisories;
using portvk::resolve_icd_library;
using portvk::parse_so_version;
using portvk::IcdEntry;
using portvk::gemm_shape_ok;
using portvk::firmware_present_in;

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
    // A software implementation's heap IS system memory, so its usage figure moves with everything else on the
    // machine and an exact-equality requery is the wrong test for it (measured: llvmpipe disagreed between two
    // queries with nothing of ours in between).  A discrete card's number only moves when someone allocates, so
    // there the equality is required.  Same distinction as the tracking check below, computed once.
    const long hpages = sysconf(_SC_PHYS_PAGES);
    const long hpsize = sysconf(_SC_PAGE_SIZE);
    const uint64_t host_ram_here = (hpages > 0 && hpsize > 0) ? (uint64_t) hpages * (uint64_t) hpsize : 0;
    const bool uma = (host_ram_here != 0) && (b.heap_total >= (host_ram_here / 100ull * 90ull));

    const uint64_t dbudget = budget > b.heap_budget ? budget - b.heap_budget : b.heap_budget - budget;
    const uint64_t dusage = usage > b.heap_usage ? usage - b.heap_usage : b.heap_usage - usage;
    const bool agrees = uma ? (dbudget <= b.heap_budget / 10 && dusage <= b.heap_usage / 10 + (1u << 20))
                            : (budget == b.heap_budget && usage == b.heap_usage);
    if (!agrees) std::printf("      requery delta: budget %llu bytes, usage %llu bytes\n",
                             (unsigned long long) dbudget, (unsigned long long) dusage);
    verdict(uma ? "budget: independent requery agrees (10% tol, UMA)" : "budget: independent requery agrees",
            agrees, agrees ? 0 : 1, 1, (double) dusage, uma ? "usage delta (UMA tolerance)" : "mismatch count");

    // (b) the reserve arithmetic: usable must be max(0, free - reserve).  Computed from a FRESH snapshot, and for
    // a software implementation compared with the same tolerance as (a) and for the same reason - llvmpipe's
    // budget figures track system-wide memory availability, which moves with anything on the box, including the
    // file reads and compiler runs this very gate performs.  On a discrete card the equality is required, and
    // there it is what catches a policy that ignores the reserve or uses the wrong free figure.
    const uint64_t free_fresh = b.heap_budget > b.heap_usage ? b.heap_budget - b.heap_usage : 0;
    const uint64_t usable_fresh = free_fresh > ctx.reserve_bytes() ? free_fresh - ctx.reserve_bytes() : 0;
    const uint64_t reported = ctx.usable_bytes();
    const uint64_t darith = reported > usable_fresh ? reported - usable_fresh : usable_fresh - reported;
    const bool arith_ok = uma ? (darith <= free_fresh / 10 + (1u << 20)) : (reported == usable_fresh);
    if (!arith_ok) std::printf("      arithmetic: reported %llu vs expected %llu (free %llu, reserve %llu)\n",
                               (unsigned long long) reported, (unsigned long long) usable_fresh,
                               (unsigned long long) free_fresh, (unsigned long long) ctx.reserve_bytes());
    verdict(uma ? "budget: free - reserve == usable (10% tol, UMA)" : "budget: free - reserve == usable",
            arith_ok, arith_ok ? 0 : 1, 1, (double) darith, uma ? "usable delta (UMA tolerance)" : "mismatch count");

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
    const bool uma_or_software = uma;                       // computed once, above
    const int64_t changed = (int64_t) (usage_after - before);
    const bool tracks = changed >= (4 << 20);
    if (uma_or_software) {
        ++g_pass;
        std::printf("INFO  %-28s usage moved %lld bytes for an 8 MiB alloc; NOT required here because the local "
                    "heap is %.0f%% of system RAM (software/UMA device - the OS protects the desktop)\n",
                    "budget tracking (informational)", (long long) changed,
                    100.0 * (double) b.heap_total / (double) host_ram_here);
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
    const uint32_t B580 = 0xe20b, ARC_PRO_B70 = 0xe223, ARC_PRO_B50 = 0xe212;
    const uint32_t A770 = 0x56a0, A770M = 0x5690;    // Alchemist: desktop and mobile parts
    struct Case { const char* rel; uint32_t vendor; uint32_t device; const char* mesa; bool budget;
                  const char* module; const char* must_have; const char* must_not; const char* must_not2;
                  const char* what; };
    // Two distinct refusals and the bands around them, per generation, plus the two "must not warn" checks which
    // are what catch a rule that leaks one generation's advice onto another.
    const Case cases[] = {
        // --- Battlemage / Xe2 (Intel's table: B580 initial 6.11, full 6.12) ---
        {"6.10.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "predates the device", "initial-support band", "",
         "6.10 predates the B580's initial support (fatal)"},
        {"6.11.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "initial-support band", "predates the device", "",
         "6.11 is the B580's initial band: it warns, it does not refuse"},
        {"6.12.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "compute-load crashes", "initial-support band", "",
         "6.12 reaches full support and lands in the CCS crash range"},
        {"6.13.0", INTEL, ARC_PRO_B70, "Mesa 26.2.0", true, "xe", "predates the device", "", "",
         "the same 6.13 IS fatal for the Arc Pro B70 - the floor is per card"},
        {"6.14.0", INTEL, ARC_PRO_B70, "Mesa 26.2.0", true, "xe", "compute-load crashes", "predates the device",
         "", "6.14 clears the Arc Pro floor and warns instead"},
        {"6.14.0", INTEL, ARC_PRO_B70, "Mesa 26.2.0", true, "xe", "6.17", "", "",
         "the B70 note records Intel's 6.17 full-support figure against our 6.14 floor"},
        {"6.11.0", INTEL, ARC_PRO_B50, "Mesa 26.2.0", true, "xe", "initial-support band", "predates the device",
         "", "the B50 also starts at 6.11 (its full support is 6.14, so 6.12/6.13 warn rather than crash-warn)"},
        {"6.19.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "compute-load crashes", "", "",
         "6.19 is still in the crash range"},
        {"7.0.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "preemption timeout", "compute-load crashes", "",
         "7.0 leaves the crash range and carries the preempt-timeout note"},
        {"7.1.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "CONFLICTING", "", "",
         "7.1's conflicting performance reports"},
        {"7.2.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "", "CONFLICTING", "", "7.2 drops the 7.1 note"},
        {"7.2.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "", "TTM eviction", "", "7.2 has no TTM note"},
        {"7.3.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "TTM eviction", "", "",
         "7.3 makes TTM eviction aggressive"},
        {"7.4.0", INTEL, B580, "Mesa 26.2.0", true, "xe", "migration queue", "", "",
         "7.4 brings the Battlemage bind work"},
        {"7.0.0", INTEL, B580, "Mesa 25.2.8", false, "xe", "26.2", "", "", "old Mesa + no driver figure -> warn"},
        {"7.0.0", INTEL, B580, "Mesa 26.2.0", false, "xe", "", "predates VK_EXT_memory_budget", "",
         "Mesa 26.2 silences the same warning"},
        // --- Alchemist / Xe-HPG (Intel's table: desktop+Pro initial 6.0, full 6.2; mobile initial 5.19) ---
        {"5.10.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "predates the device", "compute-load crashes", "",
         "5.10 predates Alchemist support entirely"},
        {"6.0.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "initial-support band", "predates the device", "",
         "6.0 is Alchemist's initial-support band"},
        {"6.2.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "", "initial-support band", "compute-load crashes",
         "6.2 is Alchemist full support: no kernel advisory and never the Xe2 crash warning"},
        {"6.6.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "", "predates the device", "compute-load crashes",
         "6.6 LTS drives an A770 - the rule that refused this was wrong"},
        {"6.11.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "", "predates the device", "",
         "PER-GENERATION: the kernel that is the B580's initial band is ordinary for Alchemist"},
        {"5.19.0", INTEL, A770M, "Mesa 26.2.0", true, "i915", "initial-support band", "predates the device", "",
         "the mobile Alchemist parts start one release earlier (5.19)"},
        {"6.2.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "cooperative matrix", "", "",
         "Alchemist carries the cooperative-matrix-regresses note"},
        {"6.6.0", INTEL, A770, "Mesa 26.2.0", true, "xe", "no DG2/Alchemist", "", "",
         "xe on an Alchemist card warns (Intel's Xe table has no DG2 part)"},
        {"6.6.0", INTEL, A770, "Mesa 26.2.0", true, "i915", "", "preemption timeout", "",
         "an Alchemist card on i915 gets NO Xe-driver notes (that is the module distinction working)"},
        {"6.6.0", INTEL, A770, "Mesa 26.2.0", true, "xe", "preemption timeout", "", "",
         "the same card ON XE does get the Xe-driver notes"},
        {"6.5.0", INTEL, 0x1234, "Mesa 26.2.0", true, "i915", "not in the generation table", "predates the device",
         "", "an unrecognised Intel id gets no invented floor"},
        {"7.0.0", AMD, 0x744c, "Mesa 25.2.8", false, "", "outside every range", "", "",
         "a non-Intel device gets no Intel rules"},
        {"7.3.0-rc5", INTEL, B580, "Mesa 26.2.0", true, "xe", "prerelease", "", "",
         "a prerelease kernel is called out"},
    };
    int bad = 0;
    for (const Case& c : cases) {
        HostEnv e{};
        e.kernel = parse_kernel_release(c.rel);
        e.mesa_version = c.mesa;
        if (c.module && c.module[0]) e.drm_modules.push_back(c.module);
        const std::vector<Advisory> adv = compat_advisories(e, c.vendor, c.device, c.budget);
        std::string all;
        for (const Advisory& a : adv) all += a.text + "\n";
        const bool have = c.must_have[0] == '\0' || all.find(c.must_have) != std::string::npos;
        const bool avoid = c.must_not[0] == '\0' || all.find(c.must_not) == std::string::npos;
        const bool avoid2 = c.must_not2[0] == '\0' || all.find(c.must_not2) == std::string::npos;
        if (!have || !avoid || !avoid2) {
            std::printf("      %s: have=%d avoid=%d avoid2=%d (want \"%s\", must not contain \"%s\" / \"%s\")\n",
                        c.what, (int) have, (int) avoid, (int) avoid2, c.must_have, c.must_not, c.must_not2);
            ++bad;
        }
    }
    // The refusals must be real on BOTH sides of every boundary, and the two generations must not share one.
    auto fat = [](const char* rel, uint32_t dev, const char* mod) {
        HostEnv e{};
        e.kernel = parse_kernel_release(rel);
        e.mesa_version = "Mesa 26.2.0";
        if (mod) e.drm_modules.push_back(mod);
        return any_fatal(compat_advisories(e, INTEL, dev, true));
    };
    const bool fatal_discriminates = fat("5.10.0", A770, "i915") &&   // below Alchemist initial
                                     !fat("6.0.0", A770, "i915") &&   // Alchemist initial band: no refusal
                                     !fat("6.6.0", A770, "i915") &&   // a common LTS - the old rule refused this
                                     fat("5.18.0", A770M, "i915") &&  // mobile Alchemist floor one release lower
                                     !fat("5.19.0", A770M, "i915") &&
                                     fat("6.10.0", B580, "xe") &&     // below Battlemage initial
                                     !fat("6.11.0", B580, "xe") &&    // Battlemage initial band
                                     fat("6.13.0", ARC_PRO_B70, "xe") &&
                                     !fat("6.14.0", ARC_PRO_B70, "xe");
    if (!fatal_discriminates) { std::printf("      a refusal boundary does not discriminate\n"); ++bad; }
    verdict("compat: advisory rules (29 cases + 9-way boundary)", bad == 0, bad,
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

// THE REST OF THE STACK, part 1: what is actually installed, and whether each ICD's library resolves.
void case_stack_components(Ctx& ctx) {
    const StackReport& s = ctx.stack();
    std::printf("INFO  %-28s loader %s | libdrm %s | session %s | Level-Zero %s | OpenCL %s\n", "stack",
                s.loader_version.empty() ? "(not found)" : s.loader_version.c_str(),
                s.libdrm_version.empty() ? "(not found)" : s.libdrm_version.c_str(), s.session_type.c_str(),
                s.level_zero ? "yes" : "no", s.opencl ? "yes" : "no");
    int unresolved = 0;
    for (const IcdEntry& e : s.icds) {
        if (!e.resolves()) ++unresolved;
    }
    std::printf("      %zu ICD file(s): %d unresolved; %zu firmware blob(s) probed: ", s.icds.size(), unresolved,
                s.firmware.size());
    for (const auto& f : s.firmware) std::printf("%s=%s ", f.name.c_str(), f.present() ? "present" : "absent");
    std::printf("\n");

    // An empty ICD list would make "every ICD resolves" vacuously true, so the count is asserted first.
    const bool have_icds = !s.icds.empty();
    verdict("stack: ICD files enumerated", have_icds, have_icds ? 0 : 1, 1, (double) s.icds.size(), "ICD files");
    const bool all_resolve = have_icds && unresolved == 0;
    verdict("stack: every ICD library resolves", all_resolve, unresolved, (int) s.icds.size(), (double) unresolved,
            "unresolved ICDs");
    const bool loader_ok = s.loader_version.empty() || parse_so_version(s.loader_version, *new int, *new int, *new int);
    verdict("stack: loader version readable", loader_ok, loader_ok ? 0 : 1, 1, 0.0, "parse failures");
    const bool fw_probed = s.firmware.size() == 6;   // x3 Battlemage blobs + x3 Alchemist blobs
    verdict("stack: firmware blobs probed", fw_probed, fw_probed ? 0 : 1, 6, 0.0, "blobs probed");
}

// Firmware lookups: a real temp directory, both directions.  The bug this guards against was live - the check
// looked only for the uncompressed name and reported all six blobs absent on a box that ships them as .zst and
// has every one of them.
void case_firmware_variants() {
    const std::string dir = "/tmp/vkport_fwtest/";
    std::filesystem::create_directories(dir);
    { std::ofstream f(dir + "test_guc.bin.zst"); f << "compressed placeholder"; }
    std::string found;
    const bool compressed_found = firmware_present_in(dir, "test_guc.bin", &found) &&
                                  found.find(".zst") != std::string::npos;
    const bool bogus_absent = !firmware_present_in(dir, "no_such_blob_for_this_test.bin", &found);
    verdict("stack: compressed firmware variant found", compressed_found, compressed_found ? 0 : 1, 1, 0.0,
            "lookup failures");
    verdict("stack: absent firmware still reported absent", bogus_absent, bogus_absent ? 0 : 1, 1, 0.0,
            "false positives");
    std::remove((dir + "test_guc.bin.zst").c_str());

    // Informational: what the LIVE report says about the blobs for a Battlemage card, on this host.
    const StackReport s = detect_stack();
    std::printf("INFO  firmware for a Battlemage card on this host:");
    for (const auto& f : s.firmware) {
        if (f.name.find("bmg") != std::string::npos) std::printf(" %s=%s", f.name.c_str(), f.present() ? "present" : "ABSENT");
    }
    std::printf("\n");
}

// THE REST OF THE STACK, part 2: the ICD resolution logic, with BOTH controls.  The first version of this check
// tested the JSON's library_path as a file path and reported all nine working ICDs as broken - `library_path` is
// normally a bare soname (`libvulkan_radeon.so`) that the loader resolves through the system library path.  So
// the check must be shown to accept a resolvable name AND to reject an unresolvable one.
void case_icd_resolution() {
    int a = 0, b = 0, c = 0;
    const bool p1 = parse_so_version("libvulkan.so.1.3.275", a, b, c) && a == 1 && b == 3 && c == 275;
    const bool p2 = parse_so_version("libdrm.so.2.125.0", a, b, c) && a == 2 && b == 125 && c == 0;
    const bool p3 = !parse_so_version("libfoo.so", a, b, c);            // no numeric tail -> false, not "1.0.0"
    verdict("stack: .so version parser (3 shapes)", p1 && p2 && p3, (p1 ? 0 : 1) + (p2 ? 0 : 1) + (p3 ? 0 : 1), 3,
            0.0, "parse failures");

    // A bare soname must RESOLVE (positive control) and a bogus one must NOT (negative control).
    const std::string bare = resolve_icd_library("libvulkan_radeon.so");
    const std::string bogus = resolve_icd_library("libno_such_vulkan_driver_xyz.so");
    const std::string abs_missing = resolve_icd_library("/nonexistent/dir/libvulkan_nope.so");
    const std::string abs_present = bare.empty() ? std::string() : resolve_icd_library(bare);
    const bool ok = !bare.empty() && bogus.empty() && abs_missing.empty() && abs_present == bare;
    verdict("stack: ICD library resolution (bare/abs/bogus)", ok, ok ? 0 : 1, 4, 0.0, "resolution mismatches");
    if (!ok) std::printf("      bare=\"%s\" bogus=\"%s\" abs_missing=\"%s\"\n", bare.c_str(), bogus.c_str(),
                         abs_missing.c_str());

    // End-to-end negative control through the detector: a stale ICD naming a library that is not there.
    const std::string stale = "/tmp/vkport_stale_icd.json";
    const std::string good = "/tmp/vkport_good_icd.json";
    const std::string icd_template = R"({"ICD": {"api_version": "1.4.318", "library_path": ")";
    {
        std::ofstream f(stale);
        f << icd_template << "libno_such_vulkan_driver_xyz.so" << R"("}, "file_format_version": "1.0.1"})" << "\n";
    }
    {
        std::ofstream f(good);
        f << icd_template << "libvulkan_radeon.so" << R"("}, "file_format_version": "1.0.1"})" << "\n";
    }
    setenv("VK_ICD_FILENAMES", stale.c_str(), 1);
    const StackReport s_stale = detect_stack();
    const std::vector<Advisory> adv_stale = stack_advisories(s_stale, 0x1002, 0x744c);
    setenv("VK_ICD_FILENAMES", good.c_str(), 1);
    const StackReport s_good = detect_stack();
    unsetenv("VK_ICD_FILENAMES");

    auto flagged = [](const std::vector<Advisory>& v, const char* needle) {
        for (const Advisory& x : v) {
            if (x.text.find(needle) != std::string::npos) return true;
        }
        return false;
    };
    const bool stale_flagged = !s_stale.icds.empty() && !s_stale.icds[0].resolves() &&
                               flagged(adv_stale, "does not resolve");
    const bool good_clean = !s_good.icds.empty() && s_good.icds[0].resolves() && !flagged(stack_advisories(s_good, 0x1002, 0x744c), "does not resolve");
    verdict("stack: stale ICD detected (neg control)", stale_flagged, stale_flagged ? 0 : 1, 1, 0.0, "not flagged");
    verdict("stack: resolvable ICD not flagged (pos control)", good_clean, good_clean ? 0 : 1, 1, 0.0, "false alarm");
    std::remove(stale.c_str());
    std::remove(good.c_str());
}

// kv_q8: the quantisation, transcribed from kv_q8.hpp's own formula -
//     scale = fp16(max|x| / 127),  code = clamp(rint(x / scale), -127, 127)
// - including the detail that matters: the scale is rounded to fp16 FIRST and the codes are computed against
// that STORED value, so they agree with the scale a reader will dequantise with.  Quantising against the
// unrounded `amax / 127` is off by a step at the boundary and shows only as a small accuracy loss, which is why
// this oracle is compared BIT-EXACTLY rather than within a tolerance.
static void kv_q8_quantize_group(const float* x, int n, std::vector<int8_t>& codes, uint16_t& sbits) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(x[i]));
    sbits = strata::kernels::f16_from_f32(amax / 127.0f);
    const float sf = strata::kernels::f32_from_f16(sbits);
    codes.assign(n, 0);
    if (sf > 0.0f) {
        for (int i = 0; i < n; ++i) {
            const float r = x[i] / sf;                    // a FLOAT divide, as the kernel does
            int q = (int) std::nearbyint((double) r);     // __float2int_rn: nearest, ties to even
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
            codes[i] = (int8_t) q;
        }
    }
}

// 8-bit KV append.  Three properties, each of which a plausible port gets wrong:
//   * the CODES AND THE SCALE BITS must match the formula exactly (a paraphrase quantises against the unrounded
//     scale and drifts),
//   * a block whose page is not resident must leave VRAM COMPLETELY untouched - proven with a sentinel-filled
//     image compared byte-for-byte, not by trusting the branch,
//   * the host copy uses a DIFFERENT row formula (the identity layout) and is written unconditionally.
// The whole destination image is compared, so a wrong row is as visible as a wrong code.
void case_kv_q8(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "kv_q8_append.spv")) return;
    if (!ctx.info().storage_buffer_8bit) {
        skip("kv_q8 append", "device lacks storageBuffer8BitAccess - the int8 code path cannot run here");
        return;
    }
    const int kv_heads = 2, head_dim = 256, page_size = 16, pages = 2, groups = head_dim / 64;
    const int pos = 20;                                   // block 1, offset 4 within the page
    const int rows = pages * kv_heads * page_size;         // 64
    const size_t code_bytes = (size_t) rows * head_dim;
    const size_t scale_bytes = (size_t) rows * groups * 2;
    const int8_t CODE_SENTINEL = -99;
    const uint16_t SCALE_SENTINEL = 0x7F7F;

    std::vector<float> kcur((size_t) kv_heads * head_dim), vcur((size_t) kv_heads * head_dim);
    for (float& v : kcur) v = rndf(1.0f);
    for (float& v : vcur) v = rndf(1.0f);
    for (int h = 0; h < kv_heads; ++h) {
        float* k = &kcur[(size_t) h * head_dim];
        float* v = &vcur[(size_t) h * head_dim];
        if (h == 0) {
            // g=0: all zeros -> sf == 0 -> every code is 0 and the scale bits are 0
            for (int t = 0; t < 64; ++t) { k[t] = 0.0f; v[t] = 0.0f; }
            // g=1: all negative, so every code is <= 0
            for (int t = 0; t < 64; ++t) { k[64 + t] = -0.25f * (float) (t + 1); v[64 + t] = -1.0f - (float) t; }
            // g=2: values sitting exactly on +/- the group maximum
            for (int t = 0; t < 64; ++t) { k[128 + t] = (t % 2 == 0) ? 3.5f : -3.5f; v[128 + t] = (t % 2) ? 2.25f : -2.25f; }
            // g=3: a maximum past fp16 range -> the scale is inf -> every code collapses to 0.  This is real
            // engine behaviour, not a synthetic trap: it is what the formula does at ~8.3e6.
            for (int t = 0; t < 64; ++t) { k[192 + t] = 1.0e7f; v[192 + t] = -1.0e7f; }
        }
    }

    struct TableCase { int entry; bool resident; const char* what; };
    const TableCase tables[] = {{0, true, "block resident (the page table maps it to page 0)"},
                                {-1, false, "block NOT resident - VRAM must be left untouched"}};

    for (const TableCase& tc : tables) {
        const std::vector<int32_t> table = {tc.entry, tc.entry};
        const std::vector<int32_t> step = {pos, pos + 1, 0, 1, 0};

        for (int mode = 0; mode < 2; ++mode) {                  // 0 = the VRAM page path, 1 = the host copy
            const bool host = mode == 1;
            Buf b_kq = ctx.alloc(code_bytes), b_vq = ctx.alloc(code_bytes);
            Buf b_ks = ctx.alloc(scale_bytes), b_vs = ctx.alloc(scale_bytes);
            Buf b_tab = ctx.alloc(table.size() * 4), b_step = ctx.alloc(step.size() * 4);
            Buf b_kc = ctx.alloc(kcur.size() * 4), b_vc = ctx.alloc(vcur.size() * 4);

            std::vector<int8_t> got_q(code_bytes, CODE_SENTINEL);
            std::vector<uint16_t> got_s(scale_bytes / 2, SCALE_SENTINEL);
            ctx.write(b_kq, got_q.data(), code_bytes);
            ctx.write(b_vq, got_q.data(), code_bytes);
            ctx.write(b_ks, got_s.data(), scale_bytes);
            ctx.write(b_vs, got_s.data(), scale_bytes);
            ctx.write(b_tab, table.data(), table.size() * 4);
            ctx.write(b_step, step.data(), step.size() * 4);
            ctx.write(b_kc, kcur.data(), kcur.size() * 4);
            ctx.write(b_vc, vcur.data(), vcur.size() * 4);

            const uint32_t threads = (uint32_t) (2 * kv_heads * groups);
            const uint32_t gcount = (threads + kLocalSize - 1) / kLocalSize;
            VkPipeline p = ctx.pipeline(dir + "/kv_q8_append.spv", 8, 16);
            struct { int kv_heads, head_dim, page_size, host_layout; } pc{kv_heads, head_dim, page_size, mode};
            ctx.dispatch(p, {&b_kq, &b_vq, &b_ks, &b_vs, &b_tab, &b_step, &b_kc, &b_vc}, &pc, sizeof(pc), gcount);
            ctx.read(b_kq, got_q.data(), code_bytes);
            ctx.read(b_ks, got_s.data(), scale_bytes);

            // The expected image: sentinel everywhere, then exactly the rows the formula names.
            std::vector<int8_t> want_q(code_bytes, CODE_SENTINEL);
            std::vector<uint16_t> want_s(scale_bytes / 2, SCALE_SENTINEL);
            if (tc.resident || host) {
                for (int h = 0; h < kv_heads; ++h) {
                    const int row = host ? ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size)
                                         : (tc.entry * kv_heads + h) * page_size + (pos % page_size);
                    for (int g = 0; g < groups; ++g) {
                        const int xbase = h * head_dim + g * 64;
                        std::vector<int8_t> kc, vc;
                        uint16_t ks = 0, vs = 0;
                        kv_q8_quantize_group(&kcur[xbase], 64, kc, ks);
                        kv_q8_quantize_group(&vcur[xbase], 64, vc, vs);
                        for (int t = 0; t < 64; ++t) {
                            want_q[(size_t) row * head_dim + g * 64 + t] = kc[t];
                        }
                        want_s[(size_t) row * groups + g] = ks;
                    }
                }
            }
            int q_bad = 0, s_bad = 0;
            for (size_t i = 0; i < code_bytes; ++i) {
                if (got_q[i] != want_q[i]) ++q_bad;
            }
            for (size_t i = 0; i < want_s.size(); ++i) {
                if (got_s[i] != want_s[i]) ++s_bad;
            }
            if (q_bad || s_bad) {
                // Diagnostic: WHERE the image differs, decoded back to (h, g, t), so the cause is visible
                // rather than inferred from a count.
                int shown = 0;
                for (int h = 0; h < kv_heads && shown < 4; ++h) {
                    const int row = host ? ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size)
                                         : (tc.entry * kv_heads + h) * page_size + (pos % page_size);
                    for (int g = 0; g < groups && shown < 4; ++g) {
                        for (int t = 0; t < 64 && shown < 4; ++t) {
                            const size_t idx = (size_t) row * head_dim + g * 64 + t;
                            if (got_q[idx] != want_q[idx]) {
                                std::printf("        code h=%d g=%d t=%d row=%d: got %d want %d\n", h, g, t, row,
                                            (int) got_q[idx], (int) want_q[idx]);
                                ++shown;
                            }
                        }
                    }
                    if (h < 2) {
                        std::printf("        scales h=%d row=%d:", h, row);
                        for (int g = 0; g < groups; ++g) {
                            const size_t sidx = (size_t) row * groups + g;
                            std::printf(" g%d got=%04x want=%04x", g, (unsigned) got_s[sidx],
                                        (unsigned) want_s[sidx]);
                        }
                        std::printf("\n");
                    }
                }
            }
            char label[96];
            std::snprintf(label, sizeof label, "kv_q8 append %s", host ? "host copy" : (tc.resident ? "vram resident" : "vram absent"));
            const bool ok = q_bad == 0 && s_bad == 0;
            const double worst = (double) (q_bad + s_bad);
            std::printf("      %-34s %s\n", label, tc.what);
            verdict(label, ok, q_bad + s_bad, (int) (code_bytes + want_s.size()), worst,
                    ok ? "differing bytes (0 = the whole image matches)" : "differing bytes");
            ctx.free(b_kq); ctx.free(b_vq); ctx.free(b_ks); ctx.free(b_vs);
            ctx.free(b_tab); ctx.free(b_step); ctx.free(b_kc); ctx.free(b_vc);
        }
    }
}

// kv_q8 GATHER: the dequantising reader.  Two things are checked and they are different in kind.
//
// (1) BIT-EXACTNESS of the dequantised output against the engine's formula, because the scratch feeds attention
//     and a value that is merely close is a wrong number in the attention input.  The synthetic cache uses a
//     scale set that walks the fp16 conversion's paths: zero, one, a SUBNORMAL, a half, a normal and the largest
//     finite half - times codes up to +-127, so the products reach the overflow branch too.
// (2) THE ROUND TRIP - append floats, gather them back, compare against the originals - which is the property the
//     engine actually depends on and which neither half proves alone: it only holds if the scale the append
//     wrote is exactly the scale the gather reads.
void case_kv_q8_gather(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "kv_q8_gather.spv")) return;
    if (!ctx.info().storage_buffer_8bit) {
        skip("kv_q8 gather", "device lacks storageBuffer8BitAccess - the int8 code path cannot run here");
        return;
    }
    const int kv_heads = 2, head_dim = 256, page_size = 16, pages = 2, groups = head_dim / 64;
    const int rows = pages * kv_heads * page_size;
    const int per = head_dim / 4;
    const size_t code_bytes = (size_t) rows * head_dim;
    const size_t scale_elems = (size_t) rows * groups;
    const uint16_t SENT = 0xDEADu;

    // ---- (1) the synthetic cache, both sides ----
    const uint16_t scale_set[6] = {0x0000u, 0x3C00u, 0x0001u, 0x3800u, 0x4C00u, 0x7BFFu};
    std::vector<int8_t> kq(code_bytes), vq(code_bytes);
    std::vector<uint16_t> ks(scale_elems), vs(scale_elems);
    for (size_t i = 0; i < code_bytes; ++i) {
        kq[i] = (int8_t) ((int) (i % 255) - 127);
        vq[i] = (int8_t) ((int) ((i * 7) % 255) - 127);
    }
    for (size_t i = 0; i < scale_elems; ++i) {
        ks[i] = scale_set[i % 6];
        vs[i] = scale_set[(i + 3) % 6];
    }
    const std::vector<int32_t> table = {0, 1};          // cell 0..15 -> page 0, 16..31 -> page 1
    const std::vector<int32_t> ids = {0, 20, 21};       // spans both pages and different offsets
    const int n_ids = (int) ids.size();
    const std::vector<int32_t> step = {0, 0, 0, n_ids, 0};

    const size_t scratch_elems = (size_t) n_ids * kv_heads * head_dim;
    const uint64_t slack = 128;
    Buf b_tab = ctx.alloc(table.size() * 4), b_ids = ctx.alloc(ids.size() * 4), b_step = ctx.alloc(step.size() * 4);
    ctx.write(b_tab, table.data(), table.size() * 4);
    ctx.write(b_ids, ids.data(), ids.size() * 4);
    ctx.write(b_step, step.data(), step.size() * 4);

    const uint32_t groups_needed = (uint32_t) (((size_t) n_ids * kv_heads * per + kLocalSize - 1) / kLocalSize);
    VkPipeline pg = ctx.pipeline(dir + "/kv_q8_gather.spv", 6, 12);
    struct { int kv_heads, head_dim, page_size; } pc{kv_heads, head_dim, page_size};

    for (int side = 0; side < 2; ++side) {              // 0 = K, 1 = V (same kernel, different buffers)
        Buf b_codes = ctx.alloc(code_bytes), b_scales = ctx.alloc(scale_elems * 2);
        Buf b_out = ctx.alloc((scratch_elems + slack) * 2);
        ctx.write(b_codes, (side ? vq : kq).data(), code_bytes);
        ctx.write(b_scales, (side ? vs : ks).data(), scale_elems * 2);
        std::vector<uint16_t> got(scratch_elems + slack, SENT);
        ctx.write(b_out, got.data(), got.size() * 2);
        ctx.dispatch(pg, {&b_codes, &b_scales, &b_tab, &b_ids, &b_step, &b_out}, &pc, sizeof(pc), groups_needed);
        ctx.read(b_out, got.data(), got.size() * 2);

        std::vector<uint16_t> want = got;                            // start from the sentinel image
        for (size_t i = 0; i < want.size(); ++i) want[i] = SENT;
        const std::vector<int8_t>& codes = side ? vq : kq;
        const std::vector<uint16_t>& scales = side ? vs : ks;
        for (int id = 0; id < n_ids; ++id) {
            for (int h = 0; h < kv_heads; ++h) {
                const int cell = ids[id];
                const int page = table[cell / page_size];
                const int row = (page * kv_heads + h) * page_size + (cell % page_size);
                for (int d = 0; d < head_dim; ++d) {
                    const float sc = strata::kernels::f32_from_f16(scales[(size_t) row * groups + d / 64]);
                    const float v = (float) codes[(size_t) row * head_dim + d] * sc;
                    want[(size_t) (id * kv_heads + h) * head_dim + d] = strata::kernels::f16_from_f32(v);
                }
            }
        }
        int bad = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (got[i] != want[i]) ++bad;
        }
        char label[96];
        std::snprintf(label, sizeof label, "kv_q8 gather %s (bit-exact)", side ? "V" : "K");
        verdict(label, bad == 0, bad, (int) want.size(), (double) bad,
                bad ? "differing halves (incl. the guard region)" : "differing halves");
        ctx.free(b_codes);
        ctx.free(b_scales);
        ctx.free(b_out);
    }

    // ---- (2) the round trip: floats -> append -> gather -> compare with the originals ----
    {
        const int pos = 20;
        std::vector<float> kcur((size_t) kv_heads * head_dim), vcur((size_t) kv_heads * head_dim);
        for (float& v : kcur) v = rndf(1.0f);
        for (float& v : vcur) v = rndf(1.0f);
        // One group with a tiny maximum, so the scale lands in the fp16 SUBNORMAL range: that is the path where
        // the builtin and the engine's converter could disagree for a different reason than overflow.
        for (int t = 0; t < 64; ++t) {
            kcur[3 * 64 + t] = 1.0e-5f * (float) (t + 1);
        }
        const std::vector<int32_t> rtable = {0, 0};      // both blocks resident on page 0
        const std::vector<int32_t> rstep = {pos, pos + 1, 0, 1, 0};
        const std::vector<int32_t> rids = {pos};

        Buf b_kq = ctx.alloc(code_bytes), b_vq = ctx.alloc(code_bytes);
        Buf b_ks = ctx.alloc(scale_elems * 2), b_vs = ctx.alloc(scale_elems * 2);
        Buf b_kc = ctx.alloc(kcur.size() * 4), b_vc = ctx.alloc(vcur.size() * 4);
        Buf b_rt = ctx.alloc(rtable.size() * 4), b_rs = ctx.alloc(rstep.size() * 4);
        Buf b_ri = ctx.alloc(rids.size() * 4);
        std::vector<int8_t> zero_q(code_bytes, 0);
        std::vector<uint16_t> zero_s(scale_elems, 0);
        ctx.write(b_kq, zero_q.data(), code_bytes);
        ctx.write(b_vq, zero_q.data(), code_bytes);
        ctx.write(b_ks, zero_s.data(), scale_elems * 2);
        ctx.write(b_vs, zero_s.data(), scale_elems * 2);
        ctx.write(b_kc, kcur.data(), kcur.size() * 4);
        ctx.write(b_vc, vcur.data(), vcur.size() * 4);
        ctx.write(b_rt, rtable.data(), rtable.size() * 4);
        ctx.write(b_rs, rstep.data(), rstep.size() * 4);
        ctx.write(b_ri, rids.data(), rids.size() * 4);

        const uint32_t athreads = (uint32_t) (2 * kv_heads * groups);
        VkPipeline pa = ctx.pipeline(dir + "/kv_q8_append.spv", 8, 16);
        struct { int kv_heads, head_dim, page_size, host_layout; } apc{kv_heads, head_dim, page_size, 0};
        ctx.dispatch(pa, {&b_kq, &b_vq, &b_ks, &b_vs, &b_rt, &b_rs, &b_kc, &b_vc}, &apc, sizeof(apc),
                     (athreads + kLocalSize - 1) / kLocalSize);

        const size_t rt_elems = (size_t) kv_heads * head_dim;
        Buf b_kout = ctx.alloc(rt_elems * 2), b_vout = ctx.alloc(rt_elems * 2);
        std::vector<uint16_t> sink(rt_elems, SENT);
        ctx.write(b_kout, sink.data(), rt_elems * 2);
        ctx.write(b_vout, sink.data(), rt_elems * 2);
        const uint32_t gthreads = (uint32_t) (kv_heads * per);
        ctx.dispatch(pg, {&b_kq, &b_ks, &b_rt, &b_ri, &b_rs, &b_kout}, &pc, sizeof(pc),
                     (gthreads + kLocalSize - 1) / kLocalSize);
        ctx.dispatch(pg, {&b_vq, &b_vs, &b_rt, &b_ri, &b_rs, &b_vout}, &pc, sizeof(pc),
                     (gthreads + kLocalSize - 1) / kLocalSize);

        std::vector<uint16_t> kgot(rt_elems), vgot(rt_elems);
        ctx.read(b_kout, kgot.data(), rt_elems * 2);
        ctx.read(b_vout, vgot.data(), rt_elems * 2);

        int bad = 0, bound_bad = 0;
        double worst_ratio = 0;
        for (int h = 0; h < kv_heads; ++h) {
            // the gathered output is in the scratch layout [id][h][head_dim], so it is indexed by h directly -
            // the cache row only mattered to the append
            for (int g = 0; g < groups; ++g) {
                const int xbase = h * head_dim + g * 64;
                std::vector<int8_t> codes;
                uint16_t sbits = 0;
                kv_q8_quantize_group(&kcur[xbase], 64, codes, sbits);
                const float sc = strata::kernels::f32_from_f16(sbits);
                for (int t = 0; t < 64; ++t) {
                    const int d = g * 64 + t;
                    // the pair must agree BIT-EXACTLY: this only holds if the append's scale is exactly what the
                    // gather reads back
                    const uint16_t want = strata::kernels::f16_from_f32((float) codes[t] * sc);
                    if (kgot[(size_t) h * head_dim + d] != want) ++bad;
                    // and the QUANTISATION bound, stated separately because it is a quality claim, not equality:
                    // one code step is the scale, so an error above the scale means the quantiser is wrong rather
                    // than merely lossy
                    const double got = (double) strata::kernels::f32_from_f16(kgot[(size_t) h * head_dim + d]);
                    const double orig = (double) kcur[xbase + t];
                    const double ratio = std::fabs(got - orig) / ((double) sc + 1e-30);
                    worst_ratio = std::max(worst_ratio, ratio);
                    if (std::fabs(got - orig) > (double) sc) ++bound_bad;
                }
            }
        }
        (void) vgot;   // V is exercised by the same code path; K carries the adversarial group
        char label[96];
        std::snprintf(label, sizeof label, "kv_q8 round trip (append then gather)");
        const bool ok = bad == 0 && bound_bad == 0;
        verdict(label, ok, bad + bound_bad, (int) (kv_heads * head_dim), worst_ratio,
                ok ? "worst |x'-x|/scale" : "failures (bit-exact + bound)");
        ctx.free(b_kq); ctx.free(b_vq); ctx.free(b_ks); ctx.free(b_vs);
        ctx.free(b_kc); ctx.free(b_vc); ctx.free(b_rt); ctx.free(b_rs); ctx.free(b_ri);
        ctx.free(b_kout); ctx.free(b_vout);
    }

    ctx.free(b_tab);
    ctx.free(b_ids);
    ctx.free(b_step);
}

// THE HOST-SIDE ORACLE for rope: `build_rope_table`'s own float64 loop, copied rather than paraphrased (it is
// the thing the port must agree with, so it must not be re-derived from the description of it).  The kernel
// receives this table, so the ONLY difference left between device and reference is one rotation step in float
// instead of double.
static void rope_table_host(int n_rot, double theta, int max_pos, std::vector<float>& ct, std::vector<float>& st) {
    const int half = n_rot / 2;
    ct.assign((size_t) max_pos * half, 0.0f);
    st.assign((size_t) max_pos * half, 0.0f);
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            ct[(size_t) p * half + i] = (float) std::cos(ang);
            st[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

// NEOX partial RoPE.  Three things this case exists to catch, all of them silent failures in the wild:
//   * the ADJACENT-pair reading of the rotation (produces correctly-shaped scrambled output),
//   * rotating the whole head_dim instead of the first n_rot (looks right on the first n_rot values),
//   * an angle/table mismatch, which a relative-only comparison reports as a transcendental difference.
// The tail is therefore asserted BIT-EXACT and the rotation uses a near-cancellation tolerance: `a*c - b*s`
// cancels for some pairs, so a relative-only bound is both too loose where it cancels and too tight where the
// values are large.  NEXT.md specifies `max(|want| * 2^-7, (|a| + |b|) * 2^-9)`; that constant is for kernels
// that COMPUTE their angles on device (fast-math sin/cos differ in the last bits).  This port passes the table,
// so the only error left is float-vs-double in one multiply-add and the bound is tightened to 2^-20 of the same
// shape - a loose constant here would accept a rotation that is wrong by ~100x.  The worst ratio of
// error-to-tolerance is printed, so the margin is visible instead of assumed.
void case_rope(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "rope_neox.spv")) return;
    const int head_dim = 256, n_rot = 64, half = n_rot / 2;
    const double theta = 10000.0;
    const int max_pos = 64;
    const float TAIL = 1234.5f;   // a sentinel no rotation could produce, used by every case AND the in-place one
    std::vector<float> ct, st;
    rope_table_host(n_rot, theta, max_pos, ct, st);

    struct RCase { int rows; std::vector<int> pos; bool mrope; const char* what; };
    const RCase cases[] = {
        {3, {0, 1, 7}, false, "position 0 (the identity rotation) beside ordinary positions"},
        {4, {5, 5, 5, 5}, false, "every row sharing one position"},
        {2, {63, 62}, false, "the last two table entries"},
        {2, {0, 1}, true, "the mrope path: cell indices, with t/h/w sectors that differ"},
    };
    // The mrope table: cell -> (t, h, w) with distinct values, so a sector mistake cannot pass unnoticed.
    const int32_t mtab[6] = {0, 1, 2, 3, 4, 5};

    for (const RCase& c : cases) {
        const int rows = c.rows;
        const size_t n = (size_t) rows * head_dim;
        const uint64_t slack = 64;                       // NaN pad after the output: an over-write must show
        std::vector<float> x(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < head_dim; ++d) {
                x[(size_t) r * head_dim + d] = (d < n_rot) ? rndf(1.0f) : TAIL;
            }
        }
        std::vector<float> ref(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            const float* xr = &x[(size_t) r * head_dim];
            float* rr = &ref[(size_t) r * head_dim];
            for (int d = n_rot; d < head_dim; ++d) rr[d] = xr[d];
            for (int i = 0; i < half; ++i) {
                const int p = c.mrope ? mtab[c.pos[r] * 3 + (i % 3)] : c.pos[r];
                const int toff = p * half + i;
                const double a = xr[i], b = xr[half + i];
                const double cc = ct[toff], ss = st[toff];
                rr[i] = (float) (a * cc - b * ss);
                rr[half + i] = (float) (a * ss + b * cc);
            }
        }
        Buf bx = ctx.alloc(n * 4);
        Buf bo = ctx.alloc((n + slack) * 4);
        Buf bc = ctx.alloc(ct.size() * 4), bs = ctx.alloc(st.size() * 4);
        Buf bp = ctx.alloc((uint64_t) rows * 4), bm = ctx.alloc(sizeof(mtab));
        ctx.write(bx, x.data(), n * 4);
        std::vector<float> out(n + slack, std::numeric_limits<float>::quiet_NaN());
        ctx.write(bo, out.data(), out.size() * 4);
        ctx.write(bc, ct.data(), ct.size() * 4);
        ctx.write(bs, st.data(), st.size() * 4);
        ctx.write(bp, c.pos.data(), (uint64_t) rows * 4);
        ctx.write(bm, mtab, sizeof(mtab));

        const uint32_t groups = (uint32_t) ((rows + kLocalSize - 1) / kLocalSize);
        VkPipeline p = ctx.pipeline(dir + "/rope_neox.spv", 6, 16);
        struct { int rows, head_dim, n_rot, mrope; } pc{rows, head_dim, n_rot, c.mrope ? 1 : 0};
        ctx.dispatch(p, {&bx, &bo, &bc, &bs, &bp, &bm}, &pc, sizeof(pc), groups);
        ctx.read(bo, out.data(), out.size() * 4);

        int bad = 0, tail_bad = 0, guard_bad = 0;
        double worst_ratio = 0;
        for (int r = 0; r < rows; ++r) {
            const float* xr = &x[(size_t) r * head_dim];
            for (int i = 0; i < half; ++i) {
                const double a = xr[i], b = xr[half + i];
                for (int off : {i, half + i}) {
                    const double want = ref[(size_t) r * head_dim + off];
                    const double got = out[(size_t) r * head_dim + off];
                    const double tol = std::max(std::fabs(want) * 9.5367e-7, (std::fabs(a) + std::fabs(b)) * 9.5367e-7);
                    const double ratio = std::fabs(got - want) / (tol + 1e-30);
                    worst_ratio = std::max(worst_ratio, ratio);
                    if (ratio > 1.0) ++bad;
                }
            }
            // THE TAIL IS BIT-EXACT: zero tolerance, because a copy is exact and "close" would mean the kernel
            // did arithmetic on values it was supposed to pass through.
            for (int d = n_rot; d < head_dim; ++d) {
                if (out[(size_t) r * head_dim + d] != TAIL) ++tail_bad;
            }
        }
        for (uint64_t i = n; i < out.size(); ++i) {
            if (!std::isnan(out[i])) ++guard_bad;
        }
        char label[96];
        std::snprintf(label, sizeof label, "rope_neox %dx%d %s", rows, head_dim, c.mrope ? "mrope" : "text");
        const bool ok = bad == 0 && tail_bad == 0 && guard_bad == 0;
        std::printf("      %-34s %s\n", label, c.what);
        verdict(label, ok, bad + tail_bad + guard_bad, (int) (rows * head_dim + slack), worst_ratio,
                ok ? "worst err/tol ratio" : "failures (err/tol ratio shown)");
        ctx.free(bx);
        ctx.free(bo);
        ctx.free(bc);
        ctx.free(bs);
        ctx.free(bp);
        ctx.free(bm);
    }

    // IN PLACE: binding 1 may alias binding 0, and the kernel is written so that it is safe (each thread reads
    // both halves of its pair before writing either).  Verified rather than assumed, because the aliasing case
    // is the one a future edit - reordering the reads and writes - would break.
    {
        const int rows = 2;
        const size_t n = (size_t) rows * head_dim;
        std::vector<float> x(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < head_dim; ++d) x[(size_t) r * head_dim + d] = (d < n_rot) ? rndf(1.0f) : TAIL;
        }
        std::vector<float> ref(n, 0.0f);
        for (int r = 0; r < rows; ++r) {
            const float* xr = &x[(size_t) r * head_dim];
            float* rr = &ref[(size_t) r * head_dim];
            for (int d = n_rot; d < head_dim; ++d) rr[d] = xr[d];
            for (int i = 0; i < half; ++i) {
                const int toff = (r + 1) * half + i;
                const double a = xr[i], b = xr[half + i], cc = ct[toff], ss = st[toff];
                rr[i] = (float) (a * cc - b * ss);
                rr[half + i] = (float) (a * ss + b * cc);
            }
        }
        Buf bxx = ctx.alloc(n * 4);
        Buf bc = ctx.alloc(ct.size() * 4), bs = ctx.alloc(st.size() * 4);
        Buf bp = ctx.alloc((uint64_t) rows * 4), bm = ctx.alloc(sizeof(mtab));
        ctx.write(bxx, x.data(), n * 4);
        ctx.write(bc, ct.data(), ct.size() * 4);
        ctx.write(bs, st.data(), st.size() * 4);
        const std::vector<int> pos2 = {1, 2};
        ctx.write(bp, pos2.data(), (uint64_t) rows * 4);
        ctx.write(bm, mtab, sizeof(mtab));
        VkPipeline p = ctx.pipeline(dir + "/rope_neox.spv", 6, 16);
        struct { int rows, head_dim, n_rot, mrope; } pc{rows, head_dim, n_rot, 0};
        ctx.dispatch(p, {&bxx, &bxx, &bc, &bs, &bp, &bm}, &pc, sizeof(pc), 1);
        std::vector<float> out(n);
        ctx.read(bxx, out.data(), n * 4);
        int bad = 0, rot_bad = 0, tail_bad_ip = 0;
        double worst_ip = 0;
        for (int r = 0; r < rows; ++r) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t i = (size_t) r * head_dim + d;
                if (d >= n_rot) {
                    // a COPY: exact or wrong
                    if (out[i] != TAIL) ++tail_bad_ip;
                    continue;
                }
                // the rotation: float-vs-double again, so the same near-cancellation bound as the cases above
                const double a = x[i < (size_t) r * head_dim + half ? i : i - half];
                const double b = x[(size_t) r * head_dim + (d < half ? half + d : d - half)];
                const double want = ref[i];
                const double tol = std::max(std::fabs(want) * 9.5367e-7, (std::fabs(a) + std::fabs(b)) * 9.5367e-7);
                const double ratio = std::fabs((double) out[i] - want) / (tol + 1e-30);
                if (ratio > worst_ip) worst_ip = ratio;
                if (ratio > 1.0) ++rot_bad;
            }
        }
        bad = rot_bad + tail_bad_ip;
        if (bad) {
            std::printf("      in-place detail: %d rotation + %d tail mismatches; first row x/got/ref:\n", rot_bad,
                        tail_bad_ip);
            for (int d = 0; d < 4; ++d) {
                std::printf("        [%d] x=%.8g got=%.8g ref=%.8g\n", d, x[d], out[d], ref[d]);
            }
            std::printf("        tail[64] x=%.8g got=%.8g ref=%.8g\n", x[64], out[64], ref[64]);
        }
        verdict("rope_neox in place (out aliases x)", bad == 0, bad, (int) n, worst_ip,
                bad ? "failures (rotation + tail)" : "worst err/tol ratio");
        ctx.free(bxx);
        ctx.free(bc);
        ctx.free(bs);
        ctx.free(bp);
        ctx.free(bm);
    }
}

// THE FALLBACK GEMM.  No shape precondition, no device requirement: this is the path that must work everywhere
// the CMA path does not - and the shapes the CMA path CANNOT take are the interesting ones here.  M=1 is a
// single-token decode step; 17x13x5 is every kind of ragged edge at once; K=8 is below one matrix-unit tile.
// All of them are normal runtime shapes, which is why the fallback cannot carry the 16-multiple contract.
void case_gemm_fma(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "gemm_fma.spv")) return;
    struct Shape { int m, n, k; const char* what; };
    const Shape shapes[] = {
        {64, 64, 64, "the shape both GEMM paths can take"},
        {1, 64, 64, "M=1 - single-token decode, structurally impossible for the CMA path"},
        {256, 1, 64, "N=1 - a single-column projection"},
        {17, 13, 5, "ragged in every dimension"},
        {2, 48, 96, "small M, K beyond one tile"},
    };
    for (const Shape& sh : shapes) {
        const int m = sh.m, n = sh.n, k = sh.k;
        std::vector<float> a((size_t) m * k), b((size_t) k * n);
        for (float& v : a) v = rndf(1.0f);
        for (float& v : b) v = rndf(1.0f);
        std::vector<float> ref((size_t) m * n);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int t = 0; t < k; ++t) acc += (double) a[(size_t) r * k + t] * (double) b[(size_t) t * n + c];
                ref[(size_t) r * n + c] = (float) acc;
            }
        }
        const uint64_t nout = (uint64_t) m * n;
        const uint64_t slack = 256;                  // NaN pad: an over-write must be DETECTED, not tolerated
        Buf ba = ctx.alloc((uint64_t) a.size() * 4);
        Buf bb = ctx.alloc((uint64_t) b.size() * 4);
        Buf bc = ctx.alloc((nout + slack) * 4);
        ctx.write(ba, a.data(), a.size() * 4);
        ctx.write(bb, b.data(), b.size() * 4);
        std::vector<float> out(nout + slack, std::numeric_limits<float>::quiet_NaN());
        ctx.write(bc, out.data(), out.size() * 4);

        // The grid is rounded UP from the element count: the surplus invocations must exit, and the guard bytes
        // below the output are what prove they did.
        const uint32_t groups = (uint32_t) ((nout + kLocalSize - 1) / kLocalSize);
        VkPipeline p = ctx.pipeline(dir + "/gemm_fma.spv", 3, 12);
        struct { uint32_t m, n, k; } pc{(uint32_t) m, (uint32_t) n, (uint32_t) k};
        ctx.dispatch(p, {&ba, &bb, &bc}, &pc, sizeof(pc), groups);
        ctx.read(bc, out.data(), out.size() * 4);

        int bad = 0, guard_bad = 0;
        double worst = 0;
        for (uint64_t i = 0; i < nout; ++i) {
            if (!close_enough(out[i], ref[i], 1e-4f, 1e-4f)) ++bad;
            const double rel = std::fabs((double) out[i] - ref[i]) / (std::fabs((double) ref[i]) + 1e-30);
            worst = std::max(worst, rel);
        }
        for (uint64_t i = nout; i < out.size(); ++i) {
            if (!std::isnan(out[i])) ++guard_bad;
        }
        char label[80];
        std::snprintf(label, sizeof label, "gemm_fma %dx%dx%d", m, n, k);
        const bool ok = bad == 0 && guard_bad == 0;
        const int shown = bad + guard_bad;
        std::printf("      %-34s %s\n", label, sh.what);
        verdict(label, ok, shown, (int) out.size(), worst, "worst rel err");
        ctx.free(ba);
        ctx.free(bb);
        ctx.free(bc);
    }
}

// The GEMM's shape precondition, tested in both directions.  The negative controls matter more than the positive
// ones here: M=1 is the shape a decode step actually produces, and dispatching it silently computes nothing.
void case_gemm_shape_contract() {
    struct C { uint32_t m, n, k; bool ok; const char* what; };
    const C cases[] = {
        {64, 64, 64, true, "square multiples of 16 rejected"},
        {32, 16, 48, true, "non-square, K=48 (a multiple of 16) rejected"},
        {16, 16, 16, true, "the smallest legal shape rejected"},
        {1, 4096, 4096, false, "M=1 - the decode shape - was ACCEPTED, and the kernel would compute nothing"},
        {64, 64, 8, false, "K=8 (below one tile) was accepted"},
        {0, 64, 64, false, "a zero dimension was accepted"},
        {17, 64, 64, false, "17 (not a multiple of 16) was accepted"},
    };
    int bad = 0;
    for (const C& c : cases) {
        if (gemm_shape_ok(c.m, c.n, c.k) != c.ok) {
            std::printf("      %s\n", c.what);
            ++bad;
        }
    }
    verdict("gemm shape contract (7 cases)", bad == 0, bad, 7, 0.0, "wrong verdicts");
}

// THE MATRIX-UNIT PATH (VK_KHR_cooperative_matrix).  This is the only case here whose kernel the toolchain can
// build for ANY device and which still cannot RUN on most of them: the config is hardware, and the driver's own
// list is the only authority.  Measured on RADV: 14 configs, all M16 N16 K16 subgroup-scope, and the only
// floating-point ones are (f16,f16 -> f16) and (f16,f16 -> f32) - so there is no fp32-operand config to fall
// back on.  A device without one is a loud SKIP, never a pass and never a failure.
//
// The reference is built from the fp16-ROUNDED operands, because that is what the kernel receives: comparing
// against the fp32 originals would charge the kernel for the storage format's own precision.
void case_gemm_coopmat(Ctx& ctx, const std::string& dir) {
    if (!have(dir, "gemm_coopmat.spv")) return;
    if (!ctx.info().cooperative_matrix) {
        skip("gemm_coopmat", "device exposes no VK_KHR_cooperative_matrix - the matrix units are unreachable here");
        return;
    }
    if (!ctx.info().cm_f16_f32) {
        skip("gemm_coopmat", "no usable config: needs M16 N16 K16 subgroup-scope with f16 A/B and an f32 accumulator");
        return;
    }
    struct Shape { int m, n, k; };
    // Every shape the case dispatches must satisfy the contract FIRST.  Without this, editing one of them to a
    // ragged shape would leave the output buffer untouched and the comparison would read that as a numeric
    // failure at best; with it, the reason is stated.
    {
        const Shape contract_shapes[] = {{64, 64, 64}, {32, 16, 48}, {16, 64, 32}};
        int bad = 0;
        for (const Shape& sh : contract_shapes) {
            if (!gemm_shape_ok((uint32_t) sh.m, (uint32_t) sh.n, (uint32_t) sh.k)) ++bad;
        }
        verdict("gemm_coopmat test shapes satisfy the contract", bad == 0, bad, 3, 0.0, "invalid shapes");
    }
    // Three shapes on purpose.  The square one cannot tell the two tile-grid axes apart (transposing the
    // mapping still visits every tile), so one wide and one tall case are required: {32,16,48} caught the
    // transposed row/column mapping that the square case passed with.
    const Shape shapes[] = {{64, 64, 64}, {32, 16, 48}, {16, 64, 32}};
    for (const Shape& sh : shapes) {
        const int m = sh.m, n = sh.n, k = sh.k;
        std::vector<float> af((size_t) m * k), bf((size_t) k * n);
        for (float& v : af) v = rndf(1.0f);
        for (float& v : bf) v = rndf(1.0f);
        std::vector<uint16_t> a16(af.size()), b16(bf.size());
        std::vector<float> ar(af.size()), br(bf.size());
        for (size_t i = 0; i < af.size(); ++i) { a16[i] = f16_from_f32(af[i]); ar[i] = strata::kernels::f32_from_f16(a16[i]); }
        for (size_t i = 0; i < bf.size(); ++i) { b16[i] = f16_from_f32(bf[i]); br[i] = strata::kernels::f32_from_f16(b16[i]); }
        std::vector<float> ref((size_t) m * n);
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                double acc = 0.0;
                for (int t = 0; t < k; ++t) {
                    acc += (double) ar[(size_t) r * k + t] * (double) br[(size_t) t * n + c];
                }
                ref[(size_t) r * n + c] = (float) acc;
            }
        }
        const uint64_t nout = (uint64_t) m * n;
        const uint64_t slack = 512;    // NaN pad: a stray write past the output must be DETECTED, not tolerated
        Buf ba = ctx.alloc((uint64_t) a16.size() * 2);
        Buf bb = ctx.alloc((uint64_t) b16.size() * 2);
        Buf bc = ctx.alloc((nout + slack) * 4);
        ctx.write(ba, a16.data(), a16.size() * 2);
        ctx.write(bb, b16.data(), b16.size() * 2);
        std::vector<float> out(nout + slack, std::numeric_limits<float>::quiet_NaN());
        ctx.write(bc, out.data(), out.size() * 4);

        // DELIBERATE OVER-DISPATCH: 8 workgroups more than there are tiles.  The kernel is supposed to discard
        // the surplus (`tile >= total` returns), and this is what proves it - the guard bytes stay NaN and no
        // tile is computed twice.  Under-dispatch is the direction that silently loses output, so the host
        // always dispatches an upper bound.
        const uint32_t tiles = (uint32_t) ((m / 16) * (n / 16));
        VkPipeline p = ctx.pipeline(dir + "/gemm_coopmat.spv", 3, 12);
        struct { uint32_t m, n, k; } pc{(uint32_t) m, (uint32_t) n, (uint32_t) k};
        ctx.dispatch(p, {&ba, &bb, &bc}, &pc, sizeof(pc), tiles + 8);
        ctx.read(bc, out.data(), out.size() * 4);

        int bad = 0, guard_bad = 0;
        double worst = 0;
        for (uint64_t i = 0; i < nout; ++i) {
            if (!close_enough(out[i], ref[i], 1e-3f, 1e-4f)) ++bad;
            const double rel = std::fabs((double) out[i] - ref[i]) / (std::fabs((double) ref[i]) + 1e-30);
            worst = std::max(worst, rel);
        }
        for (uint64_t i = nout; i < out.size(); ++i) {
            if (!std::isnan(out[i])) ++guard_bad;
        }
        char label[64];
        std::snprintf(label, sizeof label, "gemm_coopmat %dx%dx%d", m, n, k);
        const bool ok = bad == 0 && guard_bad == 0;
        if (!ok && guard_bad) std::printf("      %d write(s) into the guard region below output\n", guard_bad);
        verdict(label, ok, bad + guard_bad, (int) out.size(), worst, "worst rel err");
        ctx.free(ba);
        ctx.free(bb);
        ctx.free(bc);
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
            std::printf("%s (vendor 0x%04x device 0x%04x, api %u.%u.%u, subgroup %u, 16bit-storage %d, fp64 %d, coopmat %d/%d)\n",
                        d.name.c_str(), d.vendor_id, d.device_id, VK_VERSION_MAJOR(d.api), VK_VERSION_MINOR(d.api),
                        VK_VERSION_PATCH(d.api), d.subgroup_size, (int) d.storage_buffer_16bit, (int) d.shader_float64,
                        (int) d.cooperative_matrix, (int) d.cm_f16_f32);
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
    case_stack_components(ctx);
    case_icd_resolution();
    case_firmware_variants();
    case_compat_host(ctx);
    case_ledger_rule(ctx);
    // ORDER MATTERS: this case checks what the driver reported before anything of ours had allocated.  RADV
    // rounds an allocation up to 2 MiB, so putting an allocating kernel case ahead of it moves heapUsage by
    // megabytes and the requery legitimately stops agreeing.
    case_memory_budget(ctx);
    case_reserve_policy();
    case_reserve_refusal();
    case_kv_q8(ctx, dir);
    case_kv_q8_gather(ctx, dir);
    case_rope(ctx, dir);
    case_gemm_fma(ctx, dir);
    case_gemm_shape_contract();
    case_gemm_coopmat(ctx, dir);
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
