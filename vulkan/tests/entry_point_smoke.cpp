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
#include "strata/kernels/elementwise.hpp"   // the engine's wrappers: silu/scale/f32_to_bf16 (+ add/scale/f16/scatter)
#include "strata/kernels/verify_kernels.hpp"  // the engine's wrapper: gather_rows
#include "strata/kernels/cvec.hpp"          // the engine's wrapper + module: cvec_apply / cvec_upload
#include "strata/kernels/bf16_bits.hpp"     // bf16_from_f32: the engine's own converter, included not transcribed
#include "strata/kernels/f16_bits.hpp"      // f16_from_f32: ditto, for f32_to_f16_bulk
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

    // ---- I2, continued: the next three glue wrappers + the five doorbell_* entry points ------------------
    {   // gdn_gate: softplus(alpha + dt) * ssm_a  (the engine's rule; every third head crosses the softplus branch)
        const int h_v = 48, n_tok = 2, n = n_tok * h_v;
        std::vector<float> alpha(n), dt(h_v), sa(h_v);
        uint32_t st = 4242u;
        for (float& v : alpha) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 6.0f - 3.0f; }
        for (int h = 0; h < h_v; h += 3) alpha[(size_t) h] = 22.0f + (float) h;   // cross the x > 20 branch
        for (float& v : dt) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) - 0.5f; }
        for (float& v : sa) { st = st * 1664525u + 1013904223u; v = -(((st >> 8) / 16777216.0f) + 0.1f); }
        float* dA = strata::vulkan::arena_alloc<float>(*s, n);
        float* dD = strata::vulkan::arena_alloc<float>(*s, h_v);
        float* dS = strata::vulkan::arena_alloc<float>(*s, h_v);
        float* dG = strata::vulkan::arena_alloc<float>(*s, n);
        strata::vulkan::stream_write(*s, dA, alpha.data(), n * 4);
        strata::vulkan::stream_write(*s, dD, dt.data(), h_v * 4);
        strata::vulkan::stream_write(*s, dS, sa.data(), h_v * 4);
        strata::kernels::gdn_gate(dA, dD, dS, dG, n_tok, h_v, s);
        std::vector<float> got(n);
        strata::vulkan::stream_read(*s, dG, got.data(), n * 4);
        int bad = 0;
        double worst = 0;
        for (int i = 0; i < n; ++i) {
            const float x = alpha[i] + dt[i % h_v];
            const float want = (x > 20.0f ? x : (float) std::log1p(std::exp((double) x))) * sa[i % h_v];
            const double rel = std::fabs((double) got[i] - want) / (std::fabs((double) want) + 1e-30);
            worst = std::max(worst, rel);
            if (!(rel <= 5e-6)) ++bad;
        }
        std::printf("  gdn_gate wrapper vs softplus oracle (worst rel %.3g)\n", worst);
        check("gdn_gate", bad == 0);
    }
    {   // rms_norm_weighted: the QSA norm, weighted and unweighted, vs the double reference
        int bad = 0;
        for (int weighted = 0; weighted < 2; ++weighted) {
            const int rows = 4, cols = 256;
            const uint64_t n = (uint64_t) rows * cols;
            std::vector<float> x(n), w(n);
            uint32_t st = 77u;
            for (uint64_t i = 0; i < n; ++i) {
                st = st * 1664525u + 1013904223u;
                x[i] = ((st >> 8) / 16777216.0f) * 2.0f - 1.0f;
                w[i] = weighted ? (((st >> 8) / 16777216.0f) + 0.5f) : 1.0f;
            }
            float* dx = strata::vulkan::arena_alloc<float>(*s, n);
            float* dw = weighted ? strata::vulkan::arena_alloc<float>(*s, n) : nullptr;
            strata::vulkan::stream_write(*s, dx, x.data(), n * 4);
            if (dw != nullptr) strata::vulkan::stream_write(*s, dw, w.data(), n * 4);
            strata::kernels::rms_norm_weighted(dx, dw, rows, cols, 1e-6f, s);
            std::vector<float> got(n);
            strata::vulkan::stream_read(*s, dx, got.data(), n * 4);
            for (int r = 0; r < rows; ++r) {
                double acc = 0;
                for (int c = 0; c < cols; ++c) { const double t = x[(uint64_t) r * cols + c]; acc += t * t; }
                const double inv = 1.0 / std::sqrt(acc / (double) cols + 1e-6);
                for (int c = 0; c < cols; ++c) {
                    const uint64_t i = (uint64_t) r * cols + c;
                    const float want = (float) ((double) x[i] * (double) w[i] * inv);
                    const double rel = std::fabs((double) got[i] - want) / (std::fabs((double) want) + 1e-30);
                    if (!(rel <= 3e-3)) ++bad;
                }
            }
        }
        check("rms_norm_weighted (weighted + unweighted) vs double ref", bad == 0);
    }
    {   // embedding_gather: S4 packing, per-group scales + offsets, vs the engine's two-rounding rule
        const int n = 512, group_elems = 16, bits = 4, per_byte = 8 / bits;
        const uint32_t row_codes = (uint32_t) ((n + per_byte - 1) / per_byte);
        const uint32_t row_groups = (uint32_t) ((n + group_elems - 1) / group_elems);
        std::vector<uint8_t> codes(row_codes, 0);
        std::vector<float> scales(row_groups), offs(row_groups);
        for (int i = 0; i < n; ++i) {
            const int code = (i * 7) % (1 << bits);
            codes[i / per_byte] |= uint8_t((code & ((1 << bits) - 1)) << ((i % per_byte) * bits));
        }
        for (uint32_t k = 0; k < row_groups; ++k) { scales[k] = 0.1f + 0.01f * (float) (k % 7); offs[k] = 0.1f * (float) (k % 5) - 0.2f; }
        uint8_t* dc = strata::vulkan::arena_alloc<uint8_t>(*s, row_codes);
        float* ds = strata::vulkan::arena_alloc<float>(*s, row_groups);
        float* dof = strata::vulkan::arena_alloc<float>(*s, row_groups);
        float* dy = strata::vulkan::arena_alloc<float>(*s, n);
        strata::vulkan::stream_write(*s, dc, codes.data(), row_codes);
        strata::vulkan::stream_write(*s, ds, scales.data(), row_groups * 4);
        strata::vulkan::stream_write(*s, dof, offs.data(), row_groups * 4);
        strata::kernels::embedding_gather(dc, ds, dof, n, bits, 0, group_elems, dy, s);
        std::vector<float> got(n);
        strata::vulkan::stream_read(*s, dy, got.data(), (size_t) n * 4);
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            const unsigned code = (unsigned(codes[i / per_byte]) >> ((i % per_byte) * bits)) & ((1u << bits) - 1u);
            const int64_t g = i / group_elems;
            const float want = (float) (int(code) + 0) * scales[(size_t) g] + offs[(size_t) g];
            if (!words_equal(got[i], want)) ++bad;
        }
        check("embedding_gather == the two-rounding rule, bitwise", bad == 0);
    }
    {   // the five doorbell_* entry points, on the arena (the engine's own symbols)
        const uint32_t D = 256, K = 8;
        float* dx = strata::vulkan::arena_alloc<float>(*s, D);
        float* dxo = strata::vulkan::arena_alloc<float>(*s, D);
        int32_t* di = strata::vulkan::arena_alloc<int32_t>(*s, K);
        int32_t* dio = strata::vulkan::arena_alloc<int32_t>(*s, K);
        float* dw = strata::vulkan::arena_alloc<float>(*s, K);
        float* dwo = strata::vulkan::arena_alloc<float>(*s, K);
        uint32_t* dseq = strata::vulkan::arena_alloc<uint32_t>(*s, 1);
        uint32_t* dflag = strata::vulkan::arena_alloc<uint32_t>(*s, 1);
        std::vector<float> xin(D), win(K);
        std::vector<int32_t> iin(K);
        for (uint32_t i = 0; i < D; ++i) xin[i] = (float) i * 0.5f - 32.0f;
        for (uint32_t i = 0; i < K; ++i) { iin[i] = 200 + (int32_t) i; win[i] = (float) i * 0.25f; }
        strata::vulkan::stream_write(*s, dx, xin.data(), D * 4);
        strata::vulkan::stream_write(*s, di, iin.data(), K * 4);
        strata::vulkan::stream_write(*s, dw, win.data(), K * 4);
        strata::kernels::doorbell_publish(dx, di, dw, (int64_t) D, (int64_t) K, dxo, dio, dwo, dseq, s);
        std::vector<float> gx(D), gw(K);
        std::vector<int32_t> gi(K);
        uint32_t gseq = 0;
        strata::vulkan::stream_read(*s, dxo, gx.data(), D * 4);
        strata::vulkan::stream_read(*s, dio, gi.data(), K * 4);
        strata::vulkan::stream_read(*s, dwo, gw.data(), K * 4);
        strata::vulkan::stream_read(*s, dseq, &gseq, 4);
        bool ok = (gseq == 1u);
        for (uint32_t i = 0; i < D; ++i) ok = ok && words_equal(gx[i], xin[i]);
        for (uint32_t i = 0; i < K; ++i) ok = ok && gi[i] == iin[i] && words_equal(gw[i], win[i]);
        check("doorbell_publish: device->host ordered, ring == 1", ok);
        strata::kernels::doorbell_ring(dseq, s);
        uint32_t r2 = 0; strata::vulkan::stream_read(*s, dseq, &r2, 4);
        strata::kernels::doorbell_publish_value(dx, di, dw, (int64_t) D, (int64_t) K, dxo, dio, dwo, dseq, 5u, s);
        uint32_t r3 = 0; strata::vulkan::stream_read(*s, dseq, &r3, 4);
        check("doorbell_ring then publish_value: ring == 2 then 5", r2 == 2u && r3 == 5u);
        uint32_t served = r3;
        strata::vulkan::stream_write(*s, dflag, &served, 4);
        strata::kernels::doorbell_wait(dflag, dseq, s);        // returns: no kernel waits
        check("doorbell_wait: returns without a device wait", true);
    }

    // ---- I2, continued further: the next five wrappers ---------------------------------------------------
    {   // add_inplace: dst += src, bitwise
        const uint32_t N2 = 512;
        std::vector<float> d(N2), c(N2);
        uint32_t st = 2024u;
        for (float& v : d) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 4.0f - 2.0f; }
        for (float& v : c) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 4.0f - 2.0f; }
        float* dd = strata::vulkan::arena_alloc<float>(*s, N2);
        float* ds = strata::vulkan::arena_alloc<float>(*s, N2);
        strata::vulkan::stream_write(*s, dd, d.data(), N2 * 4);
        strata::vulkan::stream_write(*s, ds, c.data(), N2 * 4);
        strata::kernels::add_inplace(dd, ds, (int64_t) N2, s);
        std::vector<float> got(N2);
        strata::vulkan::stream_read(*s, dd, got.data(), N2 * 4);
        int bad = 0;
        for (uint32_t i = 0; i < N2; ++i) bad += !words_equal(got[i], d[i] + c[i]);
        check("add_inplace == dst+src, bitwise", bad == 0);
    }
    {   // f32_to_f16_bulk: the engine's own f16 converter, over the regimes that matter
        const uint32_t N2 = 1024;
        std::vector<float> f(N2);
        uint32_t st = 3141u;
        for (float& v : f) { st = st * 1664525u + 1013904223u; v = ((st >> 8) / 16777216.0f) * 20.0f - 10.0f; }
        f[0] = 0.0f; f[1] = -0.0f; f[2] = 65504.0f; f[3] = 1e30f; f[4] = 1e-8f; f[5] = 65536.0f;
        float* dx = strata::vulkan::arena_alloc<float>(*s, N2);
        uint16_t* dy = strata::vulkan::arena_alloc<uint16_t>(*s, N2);
        strata::vulkan::stream_write(*s, dx, f.data(), N2 * 4);
        strata::kernels::f32_to_f16_bulk(dx, dy, (int64_t) N2, s);
        std::vector<uint16_t> got(N2);
        strata::vulkan::stream_read(*s, dy, got.data(), N2 * 2);
        int bad = 0;
        for (uint32_t i = 0; i < N2; ++i) bad += (got[i] != strata::kernels::f16_from_f32(f[i]));
        check("f32_to_f16_bulk == f16_from_f32, bitwise", bad == 0);
    }
    {   // gather_rows: a derangement, both an aligned and an unaligned row width
        int bad = 0;
        struct GArm { uint32_t rb, n; } arms[] = {{64, 8}, {42, 6}};
        for (const GArm& a : arms) {
            std::vector<uint8_t> src((size_t) 16 * a.rb);
            for (uint32_t k = 0; k < 16; ++k)
                for (uint32_t j = 0; j < a.rb; ++j) src[(size_t) k * a.rb + j] = (uint8_t) ((k * 31 + j * 7 + 1) & 0xFF);
            std::vector<int32_t> ids = {9, 0, 13, 2, 15, 4, 11, 6, 1, 14, 3, 12, 5, 8, 7, 10};
            ids.resize(a.n);
            uint8_t* dsrc = strata::vulkan::arena_alloc<uint8_t>(*s, src.size());
            int32_t* dids = strata::vulkan::arena_alloc<int32_t>(*s, a.n);
            uint8_t* ddst = strata::vulkan::arena_alloc<uint8_t>(*s, (uint64_t) a.n * a.rb + 64);
            strata::vulkan::stream_write(*s, dsrc, src.data(), src.size());
            strata::vulkan::stream_write(*s, dids, ids.data(), a.n * 4);
            std::vector<uint8_t> sent((size_t) a.n * a.rb + 64, 0xA5);
            strata::vulkan::stream_write(*s, ddst, sent.data(), sent.size());
            strata::kernels::gather_rows(dsrc, a.rb, dids, a.n, ddst, s);
            std::vector<uint8_t> got(sent.size());
            strata::vulkan::stream_read(*s, ddst, got.data(), got.size());
            for (uint32_t r = 0; r < a.n; ++r)
                for (uint32_t o = 0; o < a.rb; ++o) bad += (got[(size_t) r * a.rb + o] != src[(size_t) ids[r] * a.rb + o]);
            for (uint32_t i = 0; i < 64; ++i) bad += (got[(size_t) a.n * a.rb + i] != 0xA5);
        }
        check("gather_rows: derangement, aligned + unaligned", bad == 0);
    }
    {   // scatter_rows_f32: permutation, an unnamed dst row survives; width 6 (the CUDA would refuse)
        int bad = 0;
        struct SArm { uint32_t width, n_dst; std::vector<int32_t> rows; } arms[] = {
            {512, 6, {3, 0, 5, 2}}, {6, 5, {3, 0, 4}}};
        for (const SArm& a : arms) {
            const uint32_t n_src = (uint32_t) a.rows.size();
            std::vector<float> src((size_t) n_src * a.width);
            for (uint32_t r = 0; r < n_src; ++r)
                for (uint32_t i = 0; i < a.width; ++i) src[(size_t) r * a.width + i] = (float) (r + 1) * 1.5f;
            std::vector<float> dst((size_t) a.n_dst * a.width, -7.5f);
            float* dsrc = strata::vulkan::arena_alloc<float>(*s, (size_t) n_src * a.width);
            float* ddst = strata::vulkan::arena_alloc<float>(*s, (size_t) a.n_dst * a.width);
            int32_t* drows = strata::vulkan::arena_alloc<int32_t>(*s, n_src);
            strata::vulkan::stream_write(*s, dsrc, src.data(), src.size() * 4);
            strata::vulkan::stream_write(*s, ddst, dst.data(), dst.size() * 4);
            strata::vulkan::stream_write(*s, drows, a.rows.data(), n_src * 4);
            strata::kernels::scatter_rows_f32(dsrc, ddst, drows, n_src, a.width, s);
            std::vector<float> got(dst.size());
            strata::vulkan::stream_read(*s, ddst, got.data(), got.size() * 4);
            std::vector<float> want = dst;
            for (uint32_t r = 0; r < n_src; ++r)
                for (uint32_t i = 0; i < a.width; ++i) want[(size_t) a.rows[r] * a.width + i] = src[(size_t) r * a.width + i];
            for (size_t i = 0; i < got.size(); ++i) bad += !words_equal(got[i], want[i]);
        }
        check("scatter_rows_f32: permutation + unnamed rows survive (width 6)", bad == 0);
    }
    {   // cvec_apply: the module upload + the engine wrapper, project mode (reflection) vs a double oracle
        const int n = 256, hc = 4, T = 2, L = 48, kLayer = 7;
        const int64_t r_ld = (int64_t) hc * n;
        std::vector<float> dirv((size_t) L * n, 0.0f), sl((size_t) L, 0.0f);
        {
            double nrm = 0.0;
            std::vector<float> vv((size_t) n);
            uint32_t st = 99u;
            for (float& x : vv) { st = st * 1664525u + 1013904223u; x = ((st >> 8) / 16777216.0f) * 2.0f - 1.0f; nrm += (double) x * x; }
            nrm = std::sqrt(nrm);
            for (int j = 0; j < n; ++j) dirv[(size_t) kLayer * n + j] = (float) (vv[(size_t) j] / nrm);
        }
        sl[kLayer] = 2.0f;
        std::string err;
        const bool up = strata::kernels::cvec_upload(dirv, sl, /*project*/ 0, kLayer, kLayer, n, hc, err);
        std::vector<float> R0((size_t) T * r_ld);
        { uint32_t st = 7u; for (float& x : R0) { st = st * 1664525u + 1013904223u; x = ((st >> 8) / 16777216.0f) * 6.0f - 3.0f; } }
        float* dR = strata::vulkan::arena_alloc<float>(*s, (size_t) T * r_ld);
        strata::vulkan::stream_write(*s, dR, R0.data(), R0.size() * 4);
        strata::kernels::cvec_apply(dR, kLayer, T, r_ld, nullptr, 0, nullptr, 0, false, s);
        std::vector<float> got(R0.size());
        strata::vulkan::stream_read(*s, dR, got.data(), got.size() * 4);
        double worst = 0.0;
        for (int t = 0; t < T; ++t)
            for (int cc = 0; cc < hc; ++cc) {
                double dot = 0.0;
                const float* row0 = &R0[(size_t) t * r_ld + (size_t) cc * n];
                for (int j = 0; j < n; ++j) dot += (double) row0[j] * dirv[(size_t) kLayer * n + j];
                for (int j = 0; j < n; ++j) {
                    const double want = (double) row0[j] - 2.0 * dot * dirv[(size_t) kLayer * n + j];
                    worst = std::max(worst, std::fabs((double) got[(size_t) t * r_ld + (size_t) cc * n + j] - want));
                }
            }
        std::printf("  cvec_apply wrapper vs double oracle (worst abs %.3g); upload %s\n", worst, up ? "ok" : "FAILED");
        check("cvec_apply (project) via the module + wrapper", worst <= 1e-4 && up);
    }

    strata::vulkan::stream_close(s);
    std::printf("strata_vk_entry_smoke: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}
