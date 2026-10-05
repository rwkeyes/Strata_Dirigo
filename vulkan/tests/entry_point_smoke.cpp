// vulkan/tests/entry_point_smoke.cpp - the ENGINE-side target that proves the Vulkan backend LINKS and RUNS
// through the engine's own wrapper (BACKEND-INTEGRATION.md I1's build-system decision, and the "engine sources
// join the Vulkan configuration" mechanism's first user).
//
// It is deliberately NOT the numeric oracle: the port's gate (ports/vulkan/harness/vk_gate.cpp,
// case_fwht256_entry) is where the entry point is compared BITWISE against the ported shader and falsified.
// What this target proves is the part the gate cannot: that the `-DSTRATA_ENABLE_VULKAN=ON` CONFIGURATION
// builds the adopted device layer + the Vulkan kernel TU, links them against the engine's own header wrapper,
// and runs the entry point end to end on a real device.  It checks the round trip against an explicit
// Hadamard matrix on the host, so a build that runs but computes nothing is still caught.
#include "strata/kernels/kv_q4.hpp"      // the engine's wrapper: fwht256_inplace_cuda
#include "strata/vulkan/vk_backend.hpp"
#include "vk_arena.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
// The explicit Hadamard matrix the shader implements: H[i][j] = (-1)^popcount(i&j) / 16 (orthonormal, 256).
// Reproduced in DOUBLE as the engine's own oracle rather than as a second copy of the butterflies.
void hadamard_ref(const std::vector<float>& x, std::vector<double>& y, int n_rows) {
    y.assign(x.size(), 0.0);
    for (int r = 0; r < n_rows; ++r) {
        for (int i = 0; i < 256; ++i) {
            double acc = 0.0;
            for (int j = 0; j < 256; ++j) {
                const int s = __builtin_popcount((unsigned) (i & j)) & 1;
                acc += (s ? -1.0 : 1.0) * (double) x[(size_t) r * 256 + j];
            }
            y[(size_t) r * 256 + i] = acc / 16.0;
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string spv = (argc > 1) ? argv[1] : "ports/vulkan/shaders";
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR"); e && *e) spv = e;

    strata::vulkan::Stream* s = strata::vulkan::stream_open(/*arena_bytes=*/8ull << 20, spv);
    if (s == nullptr) { std::fprintf(stderr, "smoke: no stream\n"); return 1; }

    const int n_rows = 3;
    std::vector<float> x((size_t) n_rows * 256);
    // row 0 random-ish, row 1 an outlier (the case the rotation exists for), row 2 all zero
    uint32_t st = 12345u;
    for (float& v : x) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 4.0f - 2.0f; }
    for (int j = 0; j < 256; ++j) x[(size_t) 1 * 256 + j] = (j == 7) ? 40.0f : 0.0f;
    for (int j = 0; j < 256; ++j) x[(size_t) 2 * 256 + j] = 0.0f;

    float* dev = strata::vulkan::arena_alloc<float>(*s, (uint64_t) n_rows * 256);
    strata::vulkan::stream_write(*s, dev, x.data(), x.size() * sizeof(float));

    // THE ENGINE WRAPPER, in place - exactly how the decode path calls it.
    strata::kernels::fwht256_inplace_cuda(dev, n_rows, s);

    std::vector<float> got(x.size());
    strata::vulkan::stream_read(*s, dev, got.data(), got.size() * sizeof(float));

    std::vector<double> want;
    hadamard_ref(x, want, n_rows);
    int bad = 0;
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double rel = std::fabs((double) got[i] - want[i]) / (std::fabs(want[i]) + 1e-30);
        if (std::fabs((double) got[i] - want[i]) > 1e-3) { if (bad < 3) std::printf("   mismatch i=%zu got %.6g want %.6g\n", i, (double) got[i], want[i]); }
        worst = std::max(worst, rel);
        if (!(rel <= 2e-5 || std::fabs((double) got[i] - want[i]) <= 1e-6)) ++bad;
    }
    std::printf("strata_vk_entry_smoke: device \"%s\" | fwht256 wrapper vs explicit Hadamard: %s "
                "(worst rel %.3g over %zu values)\n",
                s->ctx->info().name.c_str(), bad == 0 ? "PASS" : "FAIL", worst, got.size());
    strata::vulkan::stream_close(s);
    return bad == 0 ? 0 : 1;
}
