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
#include "strata/kernels/gdn.hpp"                 // gdn_conv_step / gdn_l2_norm
#include "strata/kernels/fused_gdn.hpp"           // fused_gdn_conv_l2 / fused_gdn_ab
#include "strata/kernels/native_gdn_preprocess.hpp"  // native_gdn_conv_silu / native_gdn_l2_norm
#include "strata/kernels/native_gdn.hpp"            // native_gdn_step / native_gdn_enabled
#include "strata/kernels/qsa.hpp"                  // QsaShapes (the KV-pool decode attention's geometry)
#include "strata/kernels/qsa_decode_attn.hpp"      // qsa_decode_attn_step / qsa_decode_attn_scratch_floats
#include "strata/kernels/gr.hpp"                  // PLE/GR: gr_write / gr_read / gr_workspace_init
#include "strata/kernels/ple.hpp"                 // PLE/GR: ple_block / ple_history_advance / ple_block_scratch_bytes
#include "strata/kernels/router_top10.hpp"        // PLE/GR: router_top10 (the generic MoE router)
#include "strata/kernels/native_moe.hpp"          // PLE/GR: native_moe_combine
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

    strata::vulkan::Stream* s = strata::vulkan::stream_open(/*arena_bytes=*/64ull << 20, spv);
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
                    const float want = (float) ((double) x[i] * (double) w[c] * inv);   // per-column weight: w[c]
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

    // ---- I2 (the GDN / DeltaNet MIXER): the six wrappers in vulkan/src/kernels/gdn_vk.cpp -----------------
    // Each is checked against a small hand-written oracle so a build that runs but computes nothing is caught;
    // the numeric proof (wrapper == the ported shader path, bitwise, and vs the case's oracle) is the gate.
    {   // gdn_conv_step: the legacy four-tap conv, out + the slid state
        const int C = 8, dc = 4;
        const size_t hist = (size_t) C * (dc - 1);
        std::vector<float> cs(hist), x((size_t) C), kw((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc - 1; ++i) cs[(size_t) c * (dc - 1) + i] = (float) (10 * (i + 1) + c);
        for (int c = 0; c < C; ++c) x[(size_t) c] = 100.0f + (float) c;
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kw[(size_t) c * dc + i] = ((i & 1) ? -1.0f : 1.0f) * (float) (i + 1);
        float* dcs = strata::vulkan::arena_alloc<float>(*s, hist);
        float* dx = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        float* dw = strata::vulkan::arena_alloc<float>(*s, (size_t) C * dc);
        float* dout = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        strata::vulkan::stream_write(*s, dcs, cs.data(), hist * 4);
        strata::vulkan::stream_write(*s, dx, x.data(), (size_t) C * 4);
        strata::vulkan::stream_write(*s, dw, kw.data(), (size_t) C * dc * 4);
        strata::kernels::gdn_conv_step(dcs, dx, dw, dout, C, dc, s);
        std::vector<float> got((size_t) C);
        strata::vulkan::stream_read(*s, dout, got.data(), (size_t) C * 4);
        int bad = 0;
        for (int c = 0; c < C; ++c) {
            double acc = 0;
            for (int i = 0; i < dc - 1; ++i) acc += (double) cs[(size_t) c * (dc - 1) + i] * kw[(size_t) c * dc + i];
            acc += (double) x[(size_t) c] * kw[(size_t) c * dc + dc - 1];
            if (!(std::fabs((double) got[(size_t) c] - acc) <= 1e-5 * (std::fabs(acc) + 1.0))) ++bad;
        }
        check("gdn_conv_step vs the four-tap rule (double)", bad == 0);
    }
    {   // native_gdn_conv_silu: the fused conv + SiLU, both outputs
        const int C = 8, dc = 4;
        const size_t hist = (size_t) C * 3;
        std::vector<float> cs(hist), x((size_t) C), kw((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < 3; ++i) cs[(size_t) c * 3 + i] = (float) (10 * (i + 1) + c);
        for (int c = 0; c < C; ++c) x[(size_t) c] = 100.0f + (float) c;
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kw[(size_t) c * dc + i] = ((i & 1) ? -1.0f : 1.0f) * (float) (i + 1);
        float* dcs = strata::vulkan::arena_alloc<float>(*s, hist);
        float* dx = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        float* dw = strata::vulkan::arena_alloc<float>(*s, (size_t) C * dc);
        float* draw = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        float* dsil = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        strata::vulkan::stream_write(*s, dcs, cs.data(), hist * 4);
        strata::vulkan::stream_write(*s, dx, x.data(), (size_t) C * 4);
        strata::vulkan::stream_write(*s, dw, kw.data(), (size_t) C * dc * 4);
        strata::kernels::native_gdn_conv_silu(dcs, dx, dw, draw, dsil, C, dc, s);
        std::vector<float> gr((size_t) C), gs((size_t) C);
        strata::vulkan::stream_read(*s, draw, gr.data(), (size_t) C * 4);
        strata::vulkan::stream_read(*s, dsil, gs.data(), (size_t) C * 4);
        int bad = 0;
        for (int c = 0; c < C; ++c) {
            double sum = 0;
            for (int i = 0; i < 3; ++i) sum += (double) cs[(size_t) c * 3 + i] * kw[(size_t) c * dc + i];
            sum += (double) x[(size_t) c] * kw[(size_t) c * dc + 3];
            if (!(std::fabs((double) gr[(size_t) c] - sum) <= 1e-5 * (std::fabs(sum) + 1.0))) ++bad;
            const double silu = sum / (1.0 + std::exp(-sum));
            if (!(std::fabs((double) gs[(size_t) c] - silu) <= 1e-5 * (std::fabs(silu) + 1.0))) ++bad;
        }
        check("native_gdn_conv_silu vs the conv+SiLU rule (double)", bad == 0);
    }
    {   // fused_gdn_conv_l2: conv + SiLU + the per-head L2 norm over the q/k heads
        const int C = 256, qk = 1, dc = 4;
        const size_t hist = (size_t) C * 3;
        std::vector<float> cs(hist), x((size_t) C), kw((size_t) C * dc);
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < 3; ++i) cs[(size_t) c * 3 + i] = (float) (10 * (i + 1) + (c % 90));
        for (int c = 0; c < C; ++c) x[(size_t) c] = 100.0f + (float) c;
        for (int c = 0; c < C; ++c)
            for (int i = 0; i < dc; ++i) kw[(size_t) c * dc + i] = ((i & 1) ? -1.0f : 1.0f) * (float) (i + 1);
        float* dcs = strata::vulkan::arena_alloc<float>(*s, hist);
        float* dx = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        float* dw = strata::vulkan::arena_alloc<float>(*s, (size_t) C * dc);
        float* dh = strata::vulkan::arena_alloc<float>(*s, (size_t) C);
        strata::vulkan::stream_write(*s, dcs, cs.data(), hist * 4);
        strata::vulkan::stream_write(*s, dx, x.data(), (size_t) C * 4);
        strata::vulkan::stream_write(*s, dw, kw.data(), (size_t) C * dc * 4);
        strata::kernels::fused_gdn_conv_l2(dcs, dx, dw, dh, C, qk, 1e-6f, s);
        std::vector<float> got((size_t) C);
        strata::vulkan::stream_read(*s, dh, got.data(), (size_t) C * 4);
        std::vector<double> pre((size_t) C);
        for (int c = 0; c < C; ++c) {
            double sum = 0;
            for (int i = 0; i < 3; ++i) sum += (double) cs[(size_t) c * 3 + i] * kw[(size_t) c * dc + i];
            sum += (double) x[(size_t) c] * kw[(size_t) c * dc + 3];
            pre[(size_t) c] = sum / (1.0 + std::exp(-sum));
        }
        double ysum = 0;
        for (int c = 0; c < 128; ++c) ysum += pre[(size_t) c] * pre[(size_t) c];
        const double scale = 1.0 / std::sqrt(ysum + 1e-6);
        int bad = 0;
        for (int c = 0; c < C; ++c) {
            const double head = (c / 128 < qk) ? pre[(size_t) c] * scale : pre[(size_t) c];
            if (!(std::fabs((double) got[(size_t) c] - head) <= 1e-5 * (std::fabs(head) + 1.0))) ++bad;
        }
        check("fused_gdn_conv_l2 vs the conv+SiLU+L2 rule (double)", bad == 0);
    }
    {   // native_gdn_l2_norm / gdn_l2_norm: the two L2 rules, both in place
        const int rows = 2, cols = 128;
        const uint64_t n = (uint64_t) rows * cols;
        std::vector<float> x((size_t) n);
        for (uint64_t i = 0; i < n; ++i) x[i] = (float) ((int) (i % 37) - 18) * 0.25f;
        float* dx = strata::vulkan::arena_alloc<float>(*s, n);
        strata::vulkan::stream_write(*s, dx, x.data(), n * 4);
        strata::kernels::native_gdn_l2_norm(dx, rows, cols, 1e-6f, s);
        std::vector<float> gn((size_t) n);
        strata::vulkan::stream_read(*s, dx, gn.data(), n * 4);
        const float inv_sqrt_cols = 1.0f / std::sqrt((float) cols);
        int bad_nat = 0;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) { const double t = x[(size_t) r * cols + c]; acc += t * t; }
            const double sc = 1.0 / std::sqrt(acc / cols + 1e-6 / cols);
            for (int c = 0; c < cols; ++c) {
                const size_t i = (size_t) r * cols + c;
                const double want = sc * x[i] * inv_sqrt_cols;
                if (!(std::fabs((double) gn[i] - want) <= 1e-5 * (std::fabs(want) + 1.0))) ++bad_nat;
            }
        }
        check("native_gdn_l2_norm vs the native rule (double)", bad_nat == 0);

        strata::vulkan::stream_write(*s, dx, x.data(), n * 4);
        strata::kernels::gdn_l2_norm(dx, rows, cols, 1e-6f, s);
        std::vector<float> gl((size_t) n);
        strata::vulkan::stream_read(*s, dx, gl.data(), n * 4);
        int bad_leg = 0;
        for (int r = 0; r < rows; ++r) {
            double acc = 0;
            for (int c = 0; c < cols; ++c) { const double t = x[(size_t) r * cols + c]; acc += t * t; }
            const double inv = 1.0 / std::sqrt(acc + 1e-6);
            for (int c = 0; c < cols; ++c) {
                const size_t i = (size_t) r * cols + c;
                const double want = x[i] * inv;
                if (!(std::fabs((double) gl[i] - want) <= 1e-5 * (std::fabs(want) + 1.0))) ++bad_leg;
            }
        }
        check("gdn_l2_norm vs the eps-on-the-squared-norm rule (double)", bad_leg == 0);
    }
    {   // fused_gdn_ab: the BF16 alpha/beta matvec with the softplus/sigmoid epilogues
        const int n2 = 64, hv = 4;
        std::vector<uint16_t> wa((size_t) hv * n2), wb((size_t) hv * n2);
        for (size_t i = 0; i < wa.size(); ++i) {
            wa[i] = strata::kernels::bf16_from_f32(((i & 1) ? 0.01f : 1.0f) * (float) ((int) (i % 11) - 5) * 0.1f);
            wb[i] = strata::kernels::bf16_from_f32(((i & 1) ? 0.1f : 1.0f) * (float) ((int) (i % 7) - 3) * 0.1f);
        }
        std::vector<float> x((size_t) n2), dt((size_t) hv), sa((size_t) hv);
        for (int j = 0; j < n2; ++j) x[(size_t) j] = (float) ((j % 9) - 4) * 0.02f;
        for (int r = 0; r < hv; ++r) { dt[(size_t) r] = (float) r * 0.05f; sa[(size_t) r] = -0.5f - (float) r; }
        dt[0] = 30.0f;
        float* dx = strata::vulkan::arena_alloc<float>(*s, (size_t) n2);
        uint16_t* dwa = strata::vulkan::arena_alloc<uint16_t>(*s, (size_t) hv * n2);
        uint16_t* dwb = strata::vulkan::arena_alloc<uint16_t>(*s, (size_t) hv * n2);
        float* ddt = strata::vulkan::arena_alloc<float>(*s, (size_t) hv);
        float* dsa = strata::vulkan::arena_alloc<float>(*s, (size_t) hv);
        float* dg = strata::vulkan::arena_alloc<float>(*s, (size_t) hv);
        float* db = strata::vulkan::arena_alloc<float>(*s, (size_t) hv);
        strata::vulkan::stream_write(*s, dx, x.data(), (size_t) n2 * 4);
        strata::vulkan::stream_write(*s, dwa, wa.data(), (size_t) hv * n2 * 2);
        strata::vulkan::stream_write(*s, dwb, wb.data(), (size_t) hv * n2 * 2);
        strata::vulkan::stream_write(*s, ddt, dt.data(), (size_t) hv * 4);
        strata::vulkan::stream_write(*s, dsa, sa.data(), (size_t) hv * 4);
        strata::kernels::fused_gdn_ab(dx, dwa, dwb, ddt, dsa, dg, db, n2, hv, s);
        std::vector<float> gg((size_t) hv), gb((size_t) hv);
        strata::vulkan::stream_read(*s, dg, gg.data(), (size_t) hv * 4);
        strata::vulkan::stream_read(*s, db, gb.data(), (size_t) hv * 4);
        int bad = 0;
        for (int r = 0; r < hv; ++r) {
            double acc = 0, accb = 0, terms = 0;
            for (int j = 0; j < n2; ++j) {
                const double wv = strata::kernels::f32_from_bf16(wa[(size_t) r * n2 + j]);
                const double wvb = strata::kernels::f32_from_bf16(wb[(size_t) r * n2 + j]);
                acc += wv * x[(size_t) j];
                accb += wvb * x[(size_t) j];
                terms += std::fabs(wv * x[(size_t) j]);
            }
            const double v = acc + dt[(size_t) r];
            const double sp = v > 20.0 ? v : std::log1p(std::exp(v));
            const double want_g = sp * sa[(size_t) r];
            const double want_b = 1.0 / (1.0 + std::exp(-accb));
            const double bg_ = 1e-4 * std::fabs(want_g) + 1e-6 * terms + 1e-30;
            if (!(std::fabs((double) gg[(size_t) r] - want_g) <= bg_)) ++bad;
            if (!(std::fabs((double) gb[(size_t) r] - want_b) <= 1e-4 * (std::fabs(want_b) + 1.0))) ++bad;
        }
        check("fused_gdn_ab vs the gate/beta rule (double, terms-bound)", bad == 0);
    }


    // ---- I2e (the GDN / DeltaNet MIXER, COMPLETED): the remaining EIGHT wrappers -------------------------
    // The beta/gate and step/norm stages.  Each is driven through the engine wrapper and checked against a
    // double transcription of the engine's OWN rule (the gate carries the shader-path bitwise proof).
    {
        const int S = 128, h_k = 16, h_v = 48;
        const size_t nstate = (size_t) S * h_v * S, no = (size_t) h_v * S;
        {   // native_gdn_beta_gate / gdn_beta_gate: in-place sigmoid
            std::vector<float> b((size_t) h_v), want((size_t) h_v);
            for (int h = 0; h < h_v; ++h)
                b[(size_t) h] = (h % 3 == 0) ? -20.0f : (h % 3 == 1) ? 20.0f : 0.5f * (float) ((h * 7) % 11 - 5);
            for (int h = 0; h < h_v; ++h)
                want[(size_t) h] = (float) (1.0 / (1.0 + std::exp(-(double) b[(size_t) h])));
            for (int which = 0; which < 2; ++which) {
                float* d = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
                strata::vulkan::stream_write(*s, d, b.data(), (size_t) h_v * 4);
                if (which == 0) strata::kernels::native_gdn_beta_gate(d, h_v, s);
                else            strata::kernels::gdn_beta_gate(d, h_v, s);
                std::vector<float> got((size_t) h_v);
                strata::vulkan::stream_read(*s, d, got.data(), (size_t) h_v * 4);
                int bad = 0; double worst = 0;
                for (int h = 0; h < h_v; ++h) {
                    const double e = std::fabs((double) got[(size_t) h] - (double) want[(size_t) h]);
                    worst = std::max(worst, e / (std::fabs((double) want[(size_t) h]) + 1e-30));
                    if (!(e <= 2e-6 * std::fabs((double) want[(size_t) h]) + 1e-7)) ++bad;
                }
                const char* nm = which ? "gdn_beta_gate" : "native_gdn_beta_gate";
                std::printf("  %s wrapper vs the sigmoid rule (worst rel %.3g)\n", nm, worst);
                check(nm, bad == 0);
            }
        }
        {   // native_gdn_gate: softplus(alpha + dt) * ssm_a
            std::vector<float> al((size_t) h_v), dt((size_t) h_v), sa((size_t) h_v), want((size_t) h_v);
            for (int h = 0; h < h_v; ++h) {
                al[(size_t) h] = (h % 3 == 0) ? 25.0f : 0.3f * (float) (h % 5);
                dt[(size_t) h] = 0.1f * (float) (h % 3);
                sa[(size_t) h] = -0.5f - 0.01f * (float) h;
            }
            for (int h = 0; h < h_v; ++h) {
                const double v = (double) al[(size_t) h] + (double) dt[(size_t) h];
                want[(size_t) h] = (float) ((v > 20.0 ? v : std::log1p(std::exp(v))) * (double) sa[(size_t) h]);
            }
            float* dA = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            float* dD = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            float* dS = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            float* dG = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            strata::vulkan::stream_write(*s, dA, al.data(), (size_t) h_v * 4);
            strata::vulkan::stream_write(*s, dD, dt.data(), (size_t) h_v * 4);
            strata::vulkan::stream_write(*s, dS, sa.data(), (size_t) h_v * 4);
            strata::kernels::native_gdn_gate(dA, dD, dS, dG, h_v, s);
            std::vector<float> got((size_t) h_v);
            strata::vulkan::stream_read(*s, dG, got.data(), (size_t) h_v * 4);
            int bad = 0; double worst = 0;
            for (int h = 0; h < h_v; ++h) {
                const double e = std::fabs((double) got[(size_t) h] - (double) want[(size_t) h]);
                worst = std::max(worst, e / (std::fabs((double) want[(size_t) h]) + 1e-30));
                if (!(e <= 5e-6 * std::fabs((double) want[(size_t) h]) + 1e-30)) ++bad;
            }
            std::printf("  native_gdn_gate wrapper vs softplus*ssm_a (worst rel %.3g)\n", worst);
            check("native_gdn_gate", bad == 0);
        }
        {   // native_gdn_step / gdn_step / fused_gdn_step_norm
            std::vector<float> st(nstate), q((size_t) h_k * S), k((size_t) h_k * S), v(no), gate((size_t) h_v),
                beta((size_t) h_v), z(no), gam((size_t) S);
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j)
                    for (int h = 0; h < h_v; ++h)
                        st[(size_t) (i * h_v + h) * S + j] = 0.001f * (float) (i + 2 * j + 3 * h) - 1.0f;
            for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * (float) ((int) (i % 7) - 3);
            for (size_t i = 0; i < k.size(); ++i) k[i] = 0.05f * (float) ((int) (i % 5) - 2);
            for (size_t i = 0; i < v.size(); ++i) v[i] = 0.1f * (float) ((int) (i % 9) - 4);
            for (int h = 0; h < h_v; ++h) {
                gate[(size_t) h] = -1.5f - 0.02f * (float) (h % 10);
                beta[(size_t) h] = 0.3f + 0.01f * (float) (h % 5);
            }
            for (int h = 0; h < h_v; ++h)
                for (int j = 0; j < S; ++j) {
                    const int kk = j % 3;
                    z[(size_t) h * S + j] = (kk == 0) ? -10.0f : (kk == 1) ? 10.0f : 0.2f * (float) (j % 4);
                }
            for (int j = 0; j < S; ++j) gam[(size_t) j] = 1.0f + 0.01f * (float) (j % 3);
            const double sqrtS = std::sqrt((double) S);
            const float eps = 1e-6f;

            float* dst = strata::vulkan::arena_alloc<float>(*s, nstate);
            float* dq = strata::vulkan::arena_alloc<float>(*s, q.size());
            float* dk = strata::vulkan::arena_alloc<float>(*s, k.size());
            float* dv = strata::vulkan::arena_alloc<float>(*s, v.size());
            float* dg = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            float* dbb = strata::vulkan::arena_alloc<float>(*s, (size_t) h_v);
            float* dz = strata::vulkan::arena_alloc<float>(*s, z.size());
            float* dgm = strata::vulkan::arena_alloc<float>(*s, gam.size());
            float* dy = strata::vulkan::arena_alloc<float>(*s, no);
            for (int mode = 0; mode < 3; ++mode) {   // 0 native_step, 1 gdn_step, 2 fused
                const bool legacy = (mode == 1), fused = (mode == 2);
                strata::vulkan::stream_write(*s, dst, st.data(), nstate * 4);
                strata::vulkan::stream_write(*s, dq, q.data(), q.size() * 4);
                strata::vulkan::stream_write(*s, dk, k.data(), k.size() * 4);
                strata::vulkan::stream_write(*s, dv, v.data(), v.size() * 4);
                strata::vulkan::stream_write(*s, dg, gate.data(), (size_t) h_v * 4);
                strata::vulkan::stream_write(*s, dbb, beta.data(), (size_t) h_v * 4);
                strata::vulkan::stream_write(*s, dz, z.data(), z.size() * 4);
                strata::vulkan::stream_write(*s, dgm, gam.data(), gam.size() * 4);
                strata::kernels::GdnShapes sh; sh.S = S; sh.h_k = h_k; sh.h_v = h_v;
                if (mode == 0) strata::kernels::native_gdn_step(dst, dq, dk, dv, dg, dbb, dy, sh, s);
                else if (mode == 1) strata::kernels::gdn_step(dst, dq, dk, dv, dg, dbb, dy, sh, s);
                else strata::kernels::fused_gdn_step_norm(dst, dq, dk, dv, dg, dbb, dz, dgm, eps, dy, h_k, h_v, s);
                std::vector<float> o(no), stt(nstate);
                strata::vulkan::stream_read(*s, dy, o.data(), no * 4);
                strata::vulkan::stream_read(*s, dst, stt.data(), nstate * 4);

                std::vector<double> so(st.begin(), st.end()), oc(no, 0.0);
                const size_t stride = (size_t) h_v * S;
                for (int h = 0; h < h_v; ++h) {
                    const int src = h % h_k;
                    const double g = std::exp((double) gate[(size_t) h]);
                    const double b = (double) beta[(size_t) h];
                    for (int j = 0; j < S; ++j) {
                        double* col = &so[(size_t) h * S + j];
                        if (legacy) for (int i = 0; i < S; ++i) col[(size_t) i * stride] *= g;
                        double kv = 0;
                        for (int i = 0; i < S; ++i) kv += col[(size_t) i * stride] * (double) k[(size_t) src * S + i];
                        const double delta = ((double) v[(size_t) h * S + j] - (legacy ? kv : g * kv)) * b;
                        double attn = 0;
                        for (int i = 0; i < S; ++i) {
                            col[(size_t) i * stride] = (legacy ? col[(size_t) i * stride] : g * col[(size_t) i * stride]) +
                                                       (double) k[(size_t) src * S + i] * delta;
                            attn += col[(size_t) i * stride] * (double) q[(size_t) src * S + i];
                        }
                        oc[(size_t) h * S + j] = attn * (legacy ? 1.0 : 1.0 / sqrtS);
                    }
                }
                std::vector<float> want(no);
                if (fused) {
                    for (int h = 0; h < h_v; ++h) {
                        double ss = 0;
                        for (int j = 0; j < S; ++j) ss += oc[(size_t) h * S + j] * oc[(size_t) h * S + j];
                        const double scl = 1.0 / std::sqrt(ss / (double) S + (double) eps);
                        for (int j = 0; j < S; ++j) {
                            const size_t id = (size_t) h * S + j;
                            want[id] = (float) ((scl * oc[id]) * (double) gam[(size_t) j] *
                                                (1.0 / (1.0 + std::exp(-(double) z[id]))));
                        }
                    }
                } else {
                    for (size_t i = 0; i < no; ++i) want[i] = (float) oc[i];
                }
                int bad = 0; double worst = 0;
                for (size_t i = 0; i < no; ++i) {
                    const double e = std::fabs((double) o[i] - (double) want[i]);
                    worst = std::max(worst, e / (2e-4 * std::fabs((double) want[i]) + 1e-5));
                    if (!(e <= 2e-4 * std::fabs((double) want[i]) + 1e-5)) ++bad;
                }
                for (size_t i = 0; i < nstate; ++i) {
                    const double e = std::fabs((double) stt[i] - so[i]);
                    worst = std::max(worst, e / (2e-4 * std::fabs(so[i]) + 1e-5));
                    if (!(e <= 2e-4 * std::fabs(so[i]) + 1e-5)) ++bad;
                }
                const char* nm = mode == 0 ? "native_gdn_step" : mode == 1 ? "gdn_step" : "fused_gdn_step_norm";
                std::printf("  %s wrapper vs the rule (double, worst err/tol %.3g)\n", nm, worst);
                check(nm, bad == 0);
            }
        }
        {   // native_gdn_out_norm / gdn_out_norm: rms_norm(o)*gamma*sigmoid(z)
            const size_t n = (size_t) h_v * S;
            std::vector<float> o(n), z(n), sn((size_t) S), want(n);
            for (size_t i = 0; i < n; ++i) o[i] = 0.05f * (float) ((int) (i % 13) - 6);
            for (int h = 0; h < h_v; ++h)
                for (int j = 0; j < S; ++j) {
                    const int kk = j % 3;
                    z[(size_t) h * S + j] = (kk == 0) ? -15.0f : (kk == 1) ? 15.0f : 0.1f * (float) (j % 5);
                }
            for (int j = 0; j < S; ++j) sn[(size_t) j] = 1.0f + 0.02f * (float) (j % 3);
            for (int h = 0; h < h_v; ++h) {
                double acc = 0;
                for (int j = 0; j < S; ++j) { const double t = o[(size_t) h * S + j]; acc += t * t; }
                const double scl = 1.0 / std::sqrt(acc / (double) S + 1e-6);
                for (int j = 0; j < S; ++j) {
                    const size_t id = (size_t) h * S + j;
                    want[id] = (float) ((scl * (double) o[id]) * (double) sn[(size_t) j] *
                                        (1.0 / (1.0 + std::exp(-(double) z[id]))));
                }
            }
            float* dO = strata::vulkan::arena_alloc<float>(*s, n);
            float* dZ = strata::vulkan::arena_alloc<float>(*s, n);
            float* dSN = strata::vulkan::arena_alloc<float>(*s, (size_t) S);
            float* dY = strata::vulkan::arena_alloc<float>(*s, n);
            for (int which = 0; which < 2; ++which) {
                strata::vulkan::stream_write(*s, dO, o.data(), n * 4);
                strata::vulkan::stream_write(*s, dZ, z.data(), n * 4);
                strata::vulkan::stream_write(*s, dSN, sn.data(), (size_t) S * 4);
                if (which == 0) strata::kernels::native_gdn_out_norm(dO, dZ, dSN, dY, h_v, S, 1e-6f, s);
                else            strata::kernels::gdn_out_norm(dO, dZ, dSN, dY, h_v, S, 1e-6f, s);
                std::vector<float> got(n);
                strata::vulkan::stream_read(*s, dY, got.data(), n * 4);
                int bad = 0; double worst = 0;
                for (size_t i = 0; i < n; ++i) {
                    const double e = std::fabs((double) got[i] - (double) want[i]);
                    worst = std::max(worst, e / (std::fabs((double) want[i]) + 1e-30));
                    if (!(e <= 1e-5 * std::fabs((double) want[i]) + 1e-6)) ++bad;
                }
                const char* nm = which ? "gdn_out_norm" : "native_gdn_out_norm";
                std::printf("  %s wrapper vs the norm rule (double, worst rel %.3g)\n", nm, worst);
                check(nm, bad == 0);
            }
        }
    }

    {   // qsa_decode_attn_step: the DEFAULT decode attention - the KV POOLS read through the PAGE TABLE.
        const int NH = 4, KH = 2, HD = 256, PS = 4, PAGES = 8;
        const int rows = PAGES * KH * PS;
        std::vector<uint16_t> kp((size_t) rows * HD), vp((size_t) rows * HD);
        for (int r = 0; r < rows; ++r)
            for (int d = 0; d < HD; ++d) {
                kp[(size_t) r * HD + d] = strata::kernels::f16_from_f32(0.01f * (float) ((r * 7 + d * 3) % 100 - 50));
                vp[(size_t) r * HD + d] = strata::kernels::f16_from_f32(0.01f * (float) ((r * 11 + d * 5) % 100 - 50));
            }
        std::vector<int32_t> table(PAGES);
        for (int i = 0; i < PAGES; ++i) table[i] = PAGES - 1 - i;
        table[5] = -1;                                     // page 5 is NOT resident: cell 20 is masked
        std::vector<int32_t> ids = {3, 17, 20, 8, 30, 11};
        const int n_ids = (int) ids.size();
        std::vector<int32_t> step = {0, 0, 0, n_ids, 0};
        std::vector<float> q((size_t) NH * HD);
        for (size_t i = 0; i < q.size(); ++i) q[i] = 0.05f * (float) ((int) (i % 17) - 8);
        strata::kernels::QsaShapes shp{};
        shp.n_head = NH; shp.n_head_kv = KH; shp.head_dim = HD; shp.page_size = PS;
        const int G = NH / KH;
        std::vector<float> want((size_t) NH * HD, 0.0f);
        for (int h = 0; h < NH; ++h) {
            const int kvh = h / G;
            std::vector<int> rr;
            std::vector<double> sc;
            for (int c = 0; c < n_ids; ++c) {
                const int cell = ids[c];
                const int page = table[cell / PS];
                if (page < 0) continue;
                const int row = (page * KH + kvh) * PS + (cell % PS);
                double dot = 0;
                for (int d = 0; d < HD; ++d) dot += (double) q[(size_t) h * HD + d] * (double) strata::kernels::f32_from_f16(kp[(size_t) row * HD + d]);
                rr.push_back(row); sc.push_back(dot / std::sqrt((double) HD));
            }
            if (rr.empty()) continue;
            double mx = -1e300; for (double v : sc) mx = std::max(mx, v);
            double den = 0; for (double v : sc) den += std::exp(v - mx);
            for (int d = 0; d < HD; ++d) {
                double acc = 0;
                for (size_t i = 0; i < rr.size(); ++i) acc += std::exp(sc[i] - mx) * (double) strata::kernels::f32_from_f16(vp[(size_t) rr[i] * HD + d]);
                want[(size_t) h * HD + d] = (float) (acc / den);
            }
        }
        float* dq = strata::vulkan::arena_alloc<float>(*s, q.size());
        uint16_t* dk = strata::vulkan::arena_alloc<uint16_t>(*s, kp.size());
        uint16_t* dv = strata::vulkan::arena_alloc<uint16_t>(*s, vp.size());
        int32_t* dt = strata::vulkan::arena_alloc<int32_t>(*s, table.size());
        int32_t* di = strata::vulkan::arena_alloc<int32_t>(*s, ids.size());
        int32_t* ds = strata::vulkan::arena_alloc<int32_t>(*s, step.size());
        float* dout = strata::vulkan::arena_alloc<float>(*s, (size_t) NH * HD);
        float* dscr = strata::vulkan::arena_alloc<float>(*s, strata::kernels::qsa_decode_attn_scratch_floats(n_ids, shp));
        strata::vulkan::stream_write(*s, dq, q.data(), q.size() * 4);
        strata::vulkan::stream_write(*s, dk, kp.data(), kp.size() * 2);
        strata::vulkan::stream_write(*s, dv, vp.data(), vp.size() * 2);
        strata::vulkan::stream_write(*s, dt, table.data(), table.size() * 4);
        strata::vulkan::stream_write(*s, di, ids.data(), ids.size() * 4);
        strata::vulkan::stream_write(*s, ds, step.data(), step.size() * 4);
        strata::kernels::QsaAttnPools pools{};
        pools.k_pool = dk; pools.v_pool = dv; pools.page_table = dt;
        strata::kernels::qsa_decode_attn_step(dq, pools, di, ds, n_ids, shp, dscr, dout, s);
        std::vector<float> got((size_t) NH * HD);
        strata::vulkan::stream_read(*s, dout, got.data(), got.size() * 4);
        int bad = 0; double worst = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double e = std::fabs((double) got[i] - (double) want[i]);
            worst = std::max(worst, e / (std::fabs((double) want[i]) + 1e-30));
            if (!(e <= 1e-4 * std::fabs((double) want[i]) + 1e-6)) ++bad;
        }
        std::printf("  qsa_decode_attn_step wrapper vs the engine's rule (double, worst rel %.3g)\n", worst);
        check("qsa_decode_attn_step", bad == 0);
    }

    {   // PLE / GR shared stages + the MoE routing rows (ple_vk.cpp).  THE NUMERIC PROOF IS THE GATE'S
        // `case_*_entry` (bitwise vs the shader path AND vs the explicit oracle); this smoke RUNS each wrapper
        // end to end on the engine stream.  Referencing ANY of the six pulls the whole TU into the link.
        // gr_write: a ZERO injection is an EXACT plain residual add (2*sigmoid(0) == 1).
        { const int ne = 64, hc = 3; std::vector<float> R((size_t) ne * hc), bo(ne), inj(hc, 0.0f);
          for (size_t i = 0; i < R.size(); ++i) R[i] = 0.01f * (float) ((int) (i % 23) - 11);
          for (int i = 0; i < ne; ++i) bo[i] = 0.02f * (float) (i % 17);
          float* dR = strata::vulkan::arena_alloc<float>(*s, R.size());
          float* dbo = strata::vulkan::arena_alloc<float>(*s, (size_t) ne);
          float* dinj = strata::vulkan::arena_alloc<float>(*s, (size_t) hc);
          strata::vulkan::stream_write(*s, dR, R.data(), R.size() * 4);
          strata::vulkan::stream_write(*s, dbo, bo.data(), (size_t) ne * 4);
          strata::vulkan::stream_write(*s, dinj, inj.data(), (size_t) hc * 4);
          strata::kernels::GrShapes gs{}; gs.n_embd = ne; gs.hc = hc; gs.hc_lr = 1;
          strata::kernels::gr_write(dR, dbo, dinj, gs, dR, s);
          std::vector<float> got(R.size());
          strata::vulkan::stream_read(*s, dR, got.data(), got.size() * 4);
          int bad = 0; for (size_t i = 0; i < R.size(); ++i) if (got[i] != R[i] + bo[i % ne]) ++bad;
          check("gr_write (zero inject = plain residual add)", bad == 0); }
        // ple_history_advance: the row-fastest shift is a copy, so bit-exact.  The geometry is FIXED by the
        // engine's contract (NG_HC_DIM x NG_HIST), so a tiny buffer is (correctly) REFUSED.
        { const int channels = 10240, nhist = 9; std::vector<float> hist((size_t) nhist * channels), nrm(channels);
          for (int c = 0; c < channels; ++c) { for (int r = 0; r < nhist; ++r) hist[(size_t) r + (size_t) nhist * c] = (float) (r + 1) + 0.1f * (float) (c % 7); nrm[c] = (float) c; }
          float* dh = strata::vulkan::arena_alloc<float>(*s, hist.size());
          float* dn = strata::vulkan::arena_alloc<float>(*s, (size_t) channels);
          strata::vulkan::stream_write(*s, dh, hist.data(), hist.size() * 4);
          strata::vulkan::stream_write(*s, dn, nrm.data(), (size_t) channels * 4);
          strata::kernels::ple_history_advance(dh, dn, s);
          std::vector<float> got(hist.size());
          strata::vulkan::stream_read(*s, dh, got.data(), got.size() * 4);
          int bad = 0;
          for (int c = 0; c < channels; ++c) { for (int r = 0; r + 1 < nhist; ++r) if (got[(size_t) r + (size_t) nhist * c] != hist[(size_t) (r + 1) + (size_t) nhist * c]) ++bad; if (got[(size_t) (nhist - 1) + (size_t) nhist * c] != nrm[c]) ++bad; }
          check("ple_history_advance (row-fastest, bit-exact)", bad == 0); }
        // native_moe_combine k=1: out = parts*w[0] (the native first term is a product).
        { const int ne = 37; std::vector<float> parts(ne), w(1, 0.75f);
          for (int i = 0; i < ne; ++i) parts[i] = 0.01f * (float) (i % 13);
          float* dp = strata::vulkan::arena_alloc<float>(*s, (size_t) ne);
          float* dw = strata::vulkan::arena_alloc<float>(*s, 1);
          float* dout = strata::vulkan::arena_alloc<float>(*s, (size_t) ne);
          strata::vulkan::stream_write(*s, dp, parts.data(), (size_t) ne * 4);
          strata::vulkan::stream_write(*s, dw, w.data(), 4);
          strata::kernels::native_moe_combine(dp, dw, nullptr, dout, ne, 1, s);
          std::vector<float> got(ne);
          strata::vulkan::stream_read(*s, dout, got.data(), (size_t) ne * 4);
          int bad = 0; for (int i = 0; i < ne; ++i) if (got[i] != parts[i] * 0.75f) ++bad;
          check("native_moe_combine (k=1 = parts*w[0])", bad == 0); }
        // router_top10: 512 experts, top-10 -> 10 DISTINCT ids in range, weights renormalised to sum ~1.
        { const int NE = 512, K = 10; std::vector<float> lg(NE);
          for (int e = 0; e < NE; ++e) lg[e] = 0.01f * (float) ((e * 37) % 211 - 105);
          float* dl = strata::vulkan::arena_alloc<float>(*s, (size_t) NE);
          int* di = strata::vulkan::arena_alloc<int>(*s, (size_t) K);
          float* dwt = strata::vulkan::arena_alloc<float>(*s, (size_t) K);
          strata::vulkan::stream_write(*s, dl, lg.data(), (size_t) NE * 4);
          strata::kernels::router_top10(dl, 1, NE, K, di, dwt, s);
          std::vector<int> ids(K); std::vector<float> wts(K);
          strata::vulkan::stream_read(*s, di, ids.data(), (size_t) K * 4);
          strata::vulkan::stream_read(*s, dwt, wts.data(), (size_t) K * 4);
          int bad = 0; std::vector<int> seen(NE, 0);
          for (int i = 0; i < K; ++i) { if (ids[i] < 0 || ids[i] >= NE || seen[ids[i]]) ++bad; seen[ids[i]] = 1; }
          double sum = 0; for (int i = 0; i < K; ++i) sum += wts[i];
          if (!(std::fabs(sum - 1.0) < 1e-3)) ++bad;
          check("router_top10 (10 distinct in-range ids, weights sum 1)", bad == 0); }
    }

    strata::vulkan::stream_close(s);
    std::printf("strata_vk_entry_smoke: %s\n", g_bad == 0 ? "PASS" : "FAIL");
    return g_bad == 0 ? 0 : 1;
}
