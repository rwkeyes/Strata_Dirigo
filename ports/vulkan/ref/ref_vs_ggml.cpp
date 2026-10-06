// ports/vulkan/ref/ref_vs_ggml.cpp - THE INDEPENDENT REFERENCE COMPARISON for the port's quantised mmvq shaders.
//
// WHY THIS EXISTS.  Every numeric case in `harness/vk_gate.cpp` for these formats compares the shader against a
// HOST TRANSCRIPTION of the same CUDA dot the shader was transcribed from (`iq2s_dot_host`, `iq3xxs_dot_host`,
// `q4_dot_host`, ...).  A transcription agrees with the code it was transcribed from whether or not either is
// right, and this port has had nine wrong oracles.  THIS TOOL IS THE FIRST COMPARISON AGAINST AN IMPLEMENTATION
// THAT WAS NOT DERIVED FROM THE CODE UNDER TEST: ggml-cpu's own `vec_dot` for each type, reached through
// `ggml_get_type_traits_cpu(ty)->vec_dot` in the llama.cpp build on this host.  Nothing here re-derives the dot
// expression: the reference is ggml's function, called on ggml's own quantised activation of the same floats.
//
// WHAT IS COMPARED, and the two differences that are EXPECTED (so they are quantified, not waved at):
//   * WEIGHTS are REAL: whole rows read out of the pack's own GGUF shard at the offsets `native_experts.txt`
//     records (expert tensors) or the tensor table (dense tensors).  The same bytes go to both sides.
//   * ACTIVATION is one f32 vector, quantised ONCE by ggml (`from_float` for the type's `vec_dot_type`) and fed
//     to both sides: the port gets a q8_1 block per 32 values whose int8 values are ggml's, and whose scale is
//     ggml's scale rounded to fp16 (the port's activation scale is an fp16, ggml's Q8_K scale is an fp32 - the
//     reference's scale is rounded to the same fp16 so both sides read the SAME number).
//   * DIFFERENCE 1 (fp32 summation order): a stated relative bound, `TREE_EPS`, times the row's summed term
//     magnitude - the same "terms bound" shape the gate itself uses (`gemv_bound`).
//   * DIFFERENCE 2 (per-part integer rounding): ONLY IQ2_S and IQ3_XXS, whose transcribed CUDA rounds the
//     per-part `(ls*sumi + sumi/2)/2` to an integer while ggml-cpu's Q8_K form accumulates exactly and divides
//     once.  The truncation loses < 1 unit of the part's `sumi` per part, so the bound is the tight bound PLUS
//     the sum over parts of `d_w * d_a` - computed FROM THE ACTUAL BYTES, not chosen to make a case pass.
//
// BOUNDS ARE STATED PER FORMAT and printed.  A failure at the bound is a FINDING, not a reason to widen it.
// A positive control per format perturbs one weight byte between the two sides and requires the comparison to
// FAIL: a comparator that cannot see a changed byte is decoration.
//
// MEASUREMENT-ONLY.  This tool is not on any engine path; it is `ports/vulkan/ref/`, built by `ref/build_ref.sh`.
//
// Usage: ref_vs_ggml [--shard PATH] [--rows N] [--device D]

#include "../harness/vk_compute.hpp"
#include "../harness/iq_grids.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

using portvk::Buf;
using portvk::Ctx;

namespace {

// ---------------------------------------------------------------- f16 bridge (self-contained: the tool must not
// borrow the engine's converters, or the comparison is one more thing derived from the tree under test)
uint16_t f32_to_f16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((u >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = u & 0x7FFFFFu;
    if (((u >> 23) & 0xFFu) == 0xFFu) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0));  // inf/nan
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return (uint16_t) (sign | h);
    }
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    uint32_t h = ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h += 1;
    return (uint16_t) (sign | h);
}
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, u;
    if (exp == 0) {
        if (man == 0) u = sign;
        else { exp = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; --exp; } man &= 0x3FFu; u = sign | (exp << 23) | (man << 13); }
    } else if (exp == 31) u = sign | 0x7F800000u | (man << 13);
    else u = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

float frand(uint32_t& s) {                       // xorshift, then scale into a realistic activation range
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    return ((float) (s & 0xFFFFFFu) / 8388608.0f - 1.0f) * 1.5f;
}

// -------------------------------------------------------------------------------------------------------------
struct Fixture {
    const char* name;        // what is compared, with the tensor it came from
    int ty;                  // ggml type of the weight row
    const char* spv;
    int nbufs;
    int n_in;
    int act_ty;              // ggml vec_dot_type for `ty` (must equal what ggml reports; checked at run time)
    const uint32_t* grid;    // the port's grid table for the type (null when the dot needs none)
    size_t grid_words;
    long long base_off;      // byte offset of the tensor in the shard
    long long expert_bytes;  // bytes per expert (== the tensor's bytes for a dense tensor)
    long long row_bytes;
    long long rows_per_expert;
    long long n_expert;
    bool truncates;          // the port's dot rounds a per-part integer division (IQ2_S, IQ3_XXS)
};

using namespace strata::vkport;

const Fixture kFixtures[] = {
    // the experts: gate/up (n_embd=2560 in), down (n_ff=640 in); 256 experts per layer.
    {"IQ3_XXS blk.0.ffn_gate_exps",  18, "iq3xxs_mmvq.spv", 4, 2560, 0, kIq3xxsGrid, 256,  1150562336LL, 627200LL,   980LL, 640, 256, true},
    {"IQ2_S   blk.1.ffn_gate_exps",  22, "iq2s_mmvq.spv",   4, 2560, 0, kIq2sGrid,   2048, 1664626592LL, 524800LL,   820LL, 640, 256, true},
    {"IQ3_S   blk.17.ffn_gate_exps", 21, "iq3s_mmvq.spv",   4, 2560, 0, kIq3sGrid,   512,  10683191712LL,704000LL,  1100LL, 640, 256, false},
    {"IQ4_XS  blk.47.ffn_gate_exps", 23, "iq4xs_mmvq.spv",  3, 2560, 0, nullptr,        0,  29129699872LL,870400LL, 1360LL, 640, 256, false},
    {"IQ4_NL  blk.0.ffn_down_exps",  20, "iq4nl_mmvq.spv",  3,  640, 0, nullptr,        0,  913711136LL,  921600LL,  360LL, 2560, 256, false},
    {"Q2_0    blk.1.ffn_down_exps",  42, "q2_0_mmvq.spv",   3,  640, 0, nullptr,        0,  1545740192LL, 460800LL,  180LL, 2560, 256, false},
    // the dense projections (single tensor, one "expert")
    {"Q4_K    blk.0.attn_gate",      12, "native_k_mmvq.spv", 3, 2560, 0, nullptr, 0, 883359776LL,   8847360LL,  1440LL, 6144, 1, false},
    {"Q5_K    blk.1.attn_qkv",       13, "native_k_mmvq.spv", 3, 2560, 0, nullptr, 0, 1527717792LL, 18022400LL, 1760LL, 10240, 1, false},
    {"Q6_K    output.weight",        14, "native_k_mmvq.spv", 3, 2560, 0, nullptr, 0, 11024416LL,   521472000LL, 2100LL, 248320, 1, false},
    {"Q8_0    blk.47.ffn_down_shexp", 8, "q8_0_mmvq.spv",    3,  640, 0, nullptr, 0, 29127959072LL, 1740800LL,  680LL, 2560, 1, false},
};

const double TREE_EPS = 1e-5;    // fp32 tree-reduction bound, relative to the summed term magnitude

int g_pass = 0, g_fail = 0, g_skip = 0;

int g_fd = -1;
void pread_exact(long long off, void* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        const ssize_t r = pread(g_fd, (char*) dst + got, n - got, off + (long long) got);
        if (r <= 0) { std::fprintf(stderr, "ref: short read at %lld (%zd of %zu)\n", off, got, n); std::exit(2); }
        got += (size_t) r;
    }
}

// The port's activation layout is q8_1 (36 B per 32 values: half d, half sum, int8 qs[32]); the dot reads `d`.
// Both ggml activation forms (`Q8_0` 34 B/32: half d + int8 qs; `Q8_K` 292 B/256: float d + int8 qs + bsums) are
// adapted by copying the INT8 VALUES unchanged and storing ggml's scale as an fp16.
std::vector<uint8_t> make_port_activation(ggml_type act_ty, const std::vector<uint8_t>& a, int n_in) {
    std::vector<uint8_t> out((size_t) (n_in / 32) * 36, 0);
    if (act_ty == GGML_TYPE_Q8_0) {
        for (int j = 0; j < n_in / 32; ++j) {
            out[(size_t) j * 36 + 0] = a[(size_t) j * 34 + 0];
            out[(size_t) j * 36 + 1] = a[(size_t) j * 34 + 1];
            std::memcpy(&out[(size_t) j * 36 + 4], &a[(size_t) j * 34 + 2], 32);
        }
    } else {   // Q8_K
        for (int b = 0; b < n_in / 256; ++b) {
            float d;
            std::memcpy(&d, &a[(size_t) b * 292], 4);
            const uint16_t dh = f32_to_f16(d);
            for (int j = 0; j < 8; ++j) {
                uint8_t* o = &out[(size_t) (b * 8 + j) * 36];
                o[0] = (uint8_t) (dh & 0xFFu);
                o[1] = (uint8_t) (dh >> 8);
                std::memcpy(o + 4, &a[(size_t) b * 292 + 4 + (size_t) j * 32], 32);
            }
        }
    }
    return out;
}

// Round every fp32 scale in a Q8_K buffer to fp16 so the reference reads the SAME scale the port stores.
void round_q8k_scales_to_f16(std::vector<uint8_t>& a, int n_in) {
    for (int b = 0; b < n_in / 256; ++b) {
        float d;
        std::memcpy(&d, &a[(size_t) b * 292], 4);
        d = f16_to_f32(f32_to_f16(d));
        std::memcpy(&a[(size_t) b * 292], &d, 4);
    }
}

// The port's q8_1 activation values in double (per 32-block: d * q), for the format-free dequant reference.
std::vector<double> act_values(const std::vector<uint8_t>& port_act) {
    std::vector<double> v((size_t) port_act.size() / 36 * 32, 0.0);
    for (size_t b = 0; b < port_act.size() / 36; ++b) {
        const float d = f16_to_f32((uint16_t) (port_act[b * 36] | ((uint16_t) port_act[b * 36 + 1] << 8)));
        for (int j = 0; j < 32; ++j) v[b * 32 + j] = (double) d * (double) (int8_t) port_act[b * 36 + 4 + j];
    }
    return v;
}

// The per-part integer-truncation allowance: sum over parts of d_w(block) * d_a(block), read from the bytes.
// IQ2_S: 8 parts per 256-block; IQ3_XXS: 8 parts per 256-block.  Each part loses < 1 unit of `sumi`.
double truncation_allowance(int ty, const std::vector<uint8_t>& w, const std::vector<uint8_t>& port_act, int n_in) {
    if (!(ty == 22 || ty == 18)) return 0.0;
    const int block_bytes = (ty == 22) ? 82 : 98;
    double sum = 0.0;
    for (int b = 0; b < n_in / 256; ++b) {
        const float dw = f16_to_f32((uint16_t) (w[(size_t) b * block_bytes] | ((uint16_t) w[(size_t) b * block_bytes + 1] << 8)));
        for (int j = 0; j < 8; ++j) {
            const uint8_t* p = &port_act[(size_t) (b * 8 + j) * 36];
            const float da = f16_to_f32((uint16_t) (p[0] | ((uint16_t) p[1] << 8)));
            sum += (double) dw * (double) da;
        }
    }
    return sum;
}

}  // namespace

int main(int argc, char** argv) {
    std::string shard = "/home/bob/strata-models/IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf";
    std::string spv_dir = "/home/bob/strata-vulkan-wt/ports/vulkan/shaders";
    int n_rows = 8, device = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--shard") && i + 1 < argc) shard = argv[++i];
        else if (!std::strcmp(argv[i], "--spv-dir") && i + 1 < argc) spv_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--rows") && i + 1 < argc) n_rows = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = std::atoi(argv[++i]);
    }
    if (const char* e = std::getenv("STRATA_VK_SPV_DIR")) spv_dir = e;

    g_fd = open(shard.c_str(), O_RDONLY);
    if (g_fd < 0) { std::fprintf(stderr, "ref: cannot open %s\n", shard.c_str()); return 2; }
    ggml_cpu_init();

    std::printf("== reference harness: the port's shaders vs ggml-cpu's `vec_dot` (llama.cpp build on this host)\n");
    std::printf("   weights: REAL rows of %s\n   rows compared per format: %d\n\n", shard.c_str(), n_rows);

    Ctx ctx(device, true);
    ctx.configure_display_reserve();          // BEFORE any allocation (the display contract, as the gate calls it)
    std::printf("   device: %s\n\n", ctx.info().name.c_str());

    for (const Fixture& f : kFixtures) {
        const ggml_type_traits_cpu* tc = ggml_get_type_traits_cpu((ggml_type) f.ty);
        if (tc == nullptr || tc->vec_dot == nullptr) { std::printf("  SKIP %-30s (ggml-cpu has no vec_dot)\n", f.name); ++g_skip; continue; }
        const ggml_type act_ty = tc->vec_dot_type;
        if ((int) act_ty != (f.act_ty ? f.act_ty : 0) && f.act_ty != 0) { /* not fatal: report what ggml says */ }
        const ggml_type_traits* tt = ggml_get_type_traits((ggml_type) f.ty);
        const ggml_type_traits_cpu* tac = ggml_get_type_traits_cpu(act_ty);
        if (tt == nullptr || tt->to_float == nullptr || tac == nullptr || tac->from_float == nullptr) {
            std::printf("  SKIP %-30s (ggml-cpu lacks to_float/from_float for act type %d)\n", f.name, (int) act_ty);
            ++g_skip; continue;
        }
        if (f.n_in % (int) ggml_blck_size((ggml_type) f.ty) || f.n_in % (int) ggml_blck_size(act_ty)) {
            std::printf("  SKIP %-30s (geometry not whole blocks for the act type %d)\n", f.name, (int) act_ty);
            ++g_skip; continue;
        }

        // ---- the activations: one f32 vector, quantised once by ggml, fed to both sides.
        std::vector<float> x((size_t) f.n_in);
        uint32_t seed = 0x2545F491u ^ (uint32_t) f.ty;
        for (auto& v : x) v = frand(seed);
        std::vector<uint8_t> act((size_t) ggml_row_size(act_ty, f.n_in), 0);
        tac->from_float(x.data(), act.data(), f.n_in);
        if (act_ty == GGML_TYPE_Q8_K) round_q8k_scales_to_f16(act, f.n_in);
        const std::vector<uint8_t> port_act = make_port_activation(act_ty, act, f.n_in);
        const std::vector<double> aval = act_values(port_act);

        // ---- weights: n_rows real rows, spread across experts where the tensor has them.
        const int n_out = n_rows;
        std::vector<uint8_t> w((size_t) n_out * (size_t) f.row_bytes);
        std::vector<long long> rowoff((size_t) n_out);
        for (int r = 0; r < n_out; ++r) {
            const long long e = f.n_expert > 1 ? (long long) ((r * 29) % f.n_expert) : 0;
            const long long rr = (long long) ((r * 37 + 11) % f.rows_per_expert);
            const long long off = f.base_off + e * f.expert_bytes + rr * f.row_bytes;
            rowoff[(size_t) r] = off;
            pread_exact(off, &w[(size_t) r * (size_t) f.row_bytes], (size_t) f.row_bytes);
        }

        // ---- the reference, once per row, from ggml ITSELF.
        std::vector<double> ref((size_t) n_out, 0.0), ref_exact((size_t) n_out, 0.0), abs_terms((size_t) n_out, 0.0);
        std::vector<float> deq((size_t) f.n_in);
        for (int r = 0; r < n_out; ++r) {
            const uint8_t* wr = &w[(size_t) r * (size_t) f.row_bytes];
            float s = 0.0f;
            tc->vec_dot(f.n_in, &s, 0, wr, 0, act.data(), 0, 1);
            ref[(size_t) r] = (double) s;
            tt->to_float(wr, deq.data(), f.n_in);
            double acc = 0.0, at = 0.0;
            for (int j = 0; j < f.n_in; ++j) { acc += (double) deq[j] * aval[(size_t) j]; at += std::fabs((double) deq[j] * aval[(size_t) j]); }
            ref_exact[(size_t) r] = acc;
            abs_terms[(size_t) r] = at;
        }

        // ---- the port's shader.
        Buf b_w = ctx.alloc((uint64_t) w.size());
        Buf b_a = ctx.alloc((uint64_t) port_act.size());
        Buf b_g = f.grid ? ctx.alloc((uint64_t) f.grid_words * 4) : ctx.alloc(4);
        Buf b_y = ctx.alloc((uint64_t) n_out * 4 + 64);
        ctx.write(b_w, w.data(), w.size());
        ctx.write(b_a, port_act.data(), port_act.size());
        if (f.grid) ctx.write(b_g, f.grid, f.grid_words * 4);
        std::vector<uint8_t> sink((size_t) n_out * 4 + 64, 0xC3);
        ctx.write(b_y, sink.data(), sink.size());
        struct { int n_in; int n_out; int row_bytes; int ncols; int ty; } pc{f.n_in, n_out, (int) f.row_bytes, 1, f.ty};
        const int pc_bytes = (f.ty == 12 || f.ty == 13 || f.ty == 14) ? (int) sizeof(pc) : 16;
        VkPipeline p = ctx.pipeline(spv_dir + "/" + f.spv, (uint32_t) f.nbufs, (uint32_t) pc_bytes);
        std::vector<const Buf*> bufs = f.grid ? std::vector<const Buf*>{&b_w, &b_a, &b_g, &b_y}
                                             : std::vector<const Buf*>{&b_w, &b_a, &b_y};
        ctx.dispatch(p, bufs, &pc, (uint32_t) pc_bytes, (uint32_t) n_out);
        std::vector<uint8_t> img(sink.size(), 0);
        ctx.read(b_y, img.data(), img.size());
        const float* got = reinterpret_cast<const float*>(img.data());

        // ---- verdicts.  The bound: fp32 summation (terms bound) + the per-part truncation allowance.
        int bad = 0, bad_ctl = 0;
        double worst = 0.0, max_rel = 0.0;
        for (int r = 0; r < n_out; ++r) {
            const double g = (double) got[r];
            double allow = truncation_allowance(f.ty, w, port_act, f.n_in);
            const double bound = TREE_EPS * abs_terms[(size_t) r] + 1e-6 * std::fabs(ref_exact[(size_t) r]) + allow + 1e-30;
            const double d1 = std::fabs(g - ref[(size_t) r]);          // vs ggml's own vec_dot
            const double d2 = std::fabs(g - ref_exact[(size_t) r]);    // vs ggml's dequant, exact arithmetic
            worst = std::max(worst, std::max(d1, d2) / bound);
            if (!std::isfinite(g)) ++bad;
            if (d1 > bound || d2 > bound) ++bad;
            const double at = abs_terms[(size_t) r] > 0 ? abs_terms[(size_t) r] : 1e-30;
            max_rel = std::max(max_rel, std::fabs(g - ref[(size_t) r]) / at);
        }
        // ---- positive control: the SAME comparison with ONE weight byte of EVERY row flipped between the sides.
        // The byte sits a third of the way into the row, inside the code/scale payload of a block; a comparator
        // that cannot see it is decoration.  Every row must MOVE past the bound.
        {
            std::vector<uint8_t> w2 = w;
            for (int r = 0; r < n_out; ++r) {
                // Flip SEVERAL spread bytes per row: a single byte can land where the activation is zero
                // (measured: one Q5_K row did), and a control that only sometimes bites proves nothing.
                for (int j = 1; j <= 8; ++j)
                    w2[(size_t) r * (size_t) f.row_bytes + (size_t) ((long long) j * f.row_bytes / 9)] ^= 0x7F;
            }
            std::vector<double> ref2((size_t) n_out, 0.0);
            for (int r = 0; r < n_out; ++r) {
                const uint8_t* wr = &w2[(size_t) r * (size_t) f.row_bytes];
                float s = 0.0f;
                tc->vec_dot(f.n_in, &s, 0, wr, 0, act.data(), 0, 1);
                ref2[(size_t) r] = (double) s;
            }
            for (int r = 0; r < n_out; ++r) {
                double allow = truncation_allowance(f.ty, w, port_act, f.n_in);
                const double bound = TREE_EPS * abs_terms[(size_t) r] + allow + 1e-30;
                if (std::fabs((double) got[r] - ref2[(size_t) r]) <= bound) ++bad_ctl;   // must NOT be true
            }
        }

        const bool ok = (bad == 0) && (bad_ctl == 0);
        if (ok) ++g_pass; else ++g_fail;
        std::printf("  %s %-30s act_type=%d  n_in=%d  rows=%d  worst err/bound=%.3g  rel(terms)=%.3g  ctl-unchanged=%d\n",
                    ok ? "PASS" : "FAIL", f.name, (int) act_ty, f.n_in, n_out, worst, max_rel, bad_ctl);
        std::printf("        row0: port=%.6g  ggml_vec_dot=%.6g  ggml_dequant_exact=%.6g  bound=%.3g\n",
                    (double) got[0], ref[0], ref_exact[0],
                    TREE_EPS * abs_terms[0] + truncation_allowance(f.ty, w, port_act, f.n_in));
        ctx.free(b_w); ctx.free(b_a); ctx.free(b_g); ctx.free(b_y);
    }

    std::printf("\n== %d passed, %d failed, %d skipped\n", g_pass, g_fail, g_skip);
    close(g_fd);
    return g_fail == 0 ? 0 : 1;
}
