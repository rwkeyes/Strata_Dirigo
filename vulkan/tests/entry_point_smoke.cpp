// vulkan/tests/entry_point_smoke.cpp - the ENGINE-side target that proves the Vulkan backend LINKS and RUNS
// through the engine's own wrappers (BACKEND-INTEGRATION.md I1's build-system decision, I2's first users).
//
// It is deliberately NOT the numeric oracle: the port's gate (ports/vulkan/harness/vk_gate.cpp) is where each
// entry point is compared against the ported shader path and falsified.  What this target proves is the part
// the gate cannot: that the `-DSTRATA_ENABLE_VULKAN=ON` CONFIGURATION builds the device layer + the Vulkan
// kernel TUs, links them against the engine's OWN header wrappers, and runs them end to end on a real device.
// I1 covered fwht256; I2 adds the three glue wrappers (silu_inplace, scale_inplace, f32_to_bf16_bulk) and the
// doorbell replacement (vulkan/src/device/sync.*).  Each is checked against a small hand-written oracle, so a
// build that runs but computes nothing is still caught.
#include "strata/kernels/kv_q4.hpp"         // the engine's wrapper: fwht256_inplace_cuda
#include "strata/kernels/elementwise.hpp"   // the engine's wrappers: silu/scale/f32_to_bf16
#include "strata/kernels/bf16_bits.hpp"     // bf16_from_f32: the engine's own converter, included not transcribed
#include "strata/vulkan/vk_backend.hpp"
#include "vk_arena.hpp"
#include "sync.hpp"                         // the doorbell replacement

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_bad = 0;

void check(const char* what, bool ok) {
    std::printf("  %-58s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++g_bad;
}

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

bool words_equal(const float& a, const float& b) {
    uint32_t u, v;
    std::memcpy(&u, &a, 4);
    std::memcpy(&v, &b, 4);
    return u == v;
}

}  // namespace

int main(int argc, char** argv) {
    std::string spv = (argc > 1) ? argv[1] : "ports/vulkan/shaders";
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR"); e && *e) spv = e;

    strata::vulkan::Stream* s = strata::vulkan::stream_open(/*arena_bytes=*/8ull << 20, spv);
    if (s == nullptr) { std::fprintf(stderr, "smoke: no stream\n"); return 1; }
    std::printf("strata_vk_entry_smoke: device \"%s\"\n", s->ctx->info().name.c_str());

    // ---- I1: fwht256, in place, through the engine wrapper ---------------------------------------------
    {
        const int n_rows = 3;
        std::vector<float> x((size_t) n_rows * 256);
        uint32_t st = 12345u;
        for (float& v : x) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 4.0f - 2.0f; }
        for (int j = 0; j < 256; ++j) x[(size_t) 1 * 256 + j] = (j == 7) ? 40.0f : 0.0f;
        for (int j = 0; j < 256; ++j) x[(size_t) 2 * 256 + j] = 0.0f;

        float* dev = strata::vulkan::arena_alloc<float>(*s, (uint64_t) n_rows * 256);
        strata::vulkan::stream_write(*s, dev, x.data(), x.size() * sizeof(float));
        strata::kernels::fwht256_inplace_cuda(dev, n_rows, s);          // THE ENGINE WRAPPER
        std::vector<float> got(x.size());
        strata::vulkan::stream_read(*s, dev, got.data(), got.size() * sizeof(float));

        std::vector<double> want;
        hadamard_ref(x, want, n_rows);
        int bad = 0;
        double worst = 0.0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double rel = std::fabs((double) got[i] - want[i]) / (std::fabs(want[i]) + 1e-30);
            worst = std::max(worst, rel);
            if (!(rel <= 2e-5 || std::fabs((double) got[i] - want[i]) <= 1e-6)) ++bad;
        }
        std::printf("  fwht256 wrapper vs explicit Hadamard (worst rel %.3g over %zu values)\n", worst, got.size());
        check("fwht256_inplace_cuda", bad == 0);
    }

    // ---- I2: the three glue wrappers --------------------------------------------------------------------
    const uint32_t N = 1000;
    std::vector<float> a(N);
    { uint32_t st = 999u; for (float& v : a) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 12.0f - 6.0f; } }

    {   // scale_inplace: x *= s, in place
        float* dev = strata::vulkan::arena_alloc<float>(*s, N);
        strata::vulkan::stream_write(*s, dev, a.data(), N * 4);
        strata::kernels::scale_inplace(dev, (int64_t) N, -1.75f, s);
        std::vector<float> got(N);
        strata::vulkan::stream_read(*s, dev, got.data(), N * 4);
        int bad = 0;
        for (uint32_t i = 0; i < N; ++i) bad += !words_equal(got[i], a[i] * -1.75f);
        check("scale_inplace == x*s, bitwise", bad == 0);
    }
    {   // silu_inplace: the engine's double reference (the same oracle the gate's shader case uses)
        float* dev = strata::vulkan::arena_alloc<float>(*s, N);
        strata::vulkan::stream_write(*s, dev, a.data(), N * 4);
        strata::kernels::silu_inplace(dev, (int64_t) N, s);
        std::vector<float> got(N);
        strata::vulkan::stream_read(*s, dev, got.data(), N * 4);
        int bad = 0;
        double worst = 0;
        for (uint32_t i = 0; i < N; ++i) {
            const double v = (double) a[i];
            const float want = (float) (v / (1.0 + std::exp(-v)));
            const double rel = std::fabs((double) got[i] - want) / (std::fabs((double) want) + 1e-30);
            worst = std::max(worst, rel);
            if (!(rel <= 2e-6 || std::fabs((double) got[i] - want) <= 1e-12)) ++bad;
        }
        std::printf("  silu wrapper vs double ref (worst rel %.3g)\n", worst);
        check("silu_inplace", bad == 0);
    }
    {   // f32_to_bf16_bulk: the engine's own bf16 converter
        float* dx = strata::vulkan::arena_alloc<float>(*s, N);
        uint16_t* dy = strata::vulkan::arena_alloc<uint16_t>(*s, N);
        strata::vulkan::stream_write(*s, dx, a.data(), N * 4);
        strata::kernels::f32_to_bf16_bulk(dx, dy, (int64_t) N, s);
        std::vector<uint16_t> got(N);
        strata::vulkan::stream_read(*s, dy, got.data(), N * 2);
        int bad = 0;
        for (uint32_t i = 0; i < N; ++i) bad += (got[i] != strata::kernels::bf16_from_f32(a[i]));
        check("f32_to_bf16_bulk == bf16_from_f32, bitwise", bad == 0);
    }

    // ---- I2: the doorbell replacement (vulkan/src/device/sync.*) ----------------------------------------
    // Device -> host publish, then the host answers, then the device consumes.  The ordering arm: a consume
    // BEFORE the answer must read the sentinel (this is the wrong order a translating spin would hang on).
    {
        const uint32_t H = 512;
        strata::vulkan::Handoff* h = strata::vulkan::sync_open(*s->ctx, (uint64_t) H * 4, (uint64_t) H * 4, spv);
        std::vector<float> src_host(H);
        for (uint32_t i = 0; i < H; ++i) src_host[i] = (float) (int) i - 100.0f;
        strata::vulkan::Buf src = s->ctx->alloc((uint64_t) H * 4);
        s->ctx->write(src, src_host.data(), H * 4);

        strata::vulkan::sync_publish(*h, src, (uint64_t) H * 4);
        std::vector<float> payload(H);
        strata::vulkan::sync_read_payload(*h, payload.data(), (uint64_t) H * 4);
        check("handoff: device->host publish is ordered", std::memcmp(payload.data(), src_host.data(), H * 4) == 0);

        strata::vulkan::Buf out = s->ctx->alloc((uint64_t) H * 4);
        strata::vulkan::sync_consume(*h, out, (uint64_t) H * 4);
        std::vector<float> early(H);
        s->ctx->read(out, early.data(), H * 4);
        int nz = 0;
        for (uint32_t i = 0; i < H; ++i) if (early[i] != 0.0f) ++nz;
        check("handoff: consume before the answer reads the sentinel", nz == 0);

        std::vector<float> answer(H);
        for (uint32_t i = 0; i < H; ++i) answer[i] = payload[i] + 1.0f;
        strata::vulkan::sync_write_answer(*h, answer.data(), (uint64_t) H * 4);
        strata::vulkan::sync_consume(*h, out, (uint64_t) H * 4);
        std::vector<float> got(H);
        s->ctx->read(out, got.data(), H * 4);
        check("handoff: host->device answer is consumed in order",
              std::memcmp(got.data(), answer.data(), H * 4) == 0);
        check("handoff: the ring is monotonic (1 after one publish)", strata::vulkan::sync_ring(*h) == 1u);

        strata::vulkan::sync_close(h);
        s->ctx->free(src);
        s->ctx->free(out);
    }

    strata::vulkan::stream_close(s);
    std::printf("strata_vk_entry_smoke: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}
