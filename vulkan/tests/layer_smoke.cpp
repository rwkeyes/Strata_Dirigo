// vulkan/tests/layer_smoke.cpp - M-B: the first ENGINE-LINKED target that instantiates ONE layer body
// (the GDN / DeltaNet mixer `gdn_layer`, 36 of the model's 48 layers) with RANDOM weights and runs it on the
// Arc against the Vulkan backend.  No model file, no tokeniser, no prompt.
//
// THE WEIGHTS.  `WeightTable`'s storage is private and its only public constructor is `load(pack_dir, ...)`,
// so random weights are delivered through the loader's OWN pack format: this program writes a synthetic
// `dense.bin` + `index.txt` of RANDOM bytes for exactly the nine tensors `gdn_layer` resolves, then calls
// `WeightTable::load`.  That is not a model - it is random data in the loader's layout - and the loader's own
// segment check (sum of planes == the row's declared src/dst bytes) refuses a mis-sized plane rather than
// loading a wrong byte offset.
//
// WHAT IT DOES, IN ORDER (the plan's M-B):
//   1. open the engine stream (device + arena),
//   2. write the synthetic random pack and load it into the arena,
//   3. carve the layer's GdnBuffers (from `gdn_buffers_bytes`) and zero the state,
//   4. call strata::core::gdn_layer(...),
//   5. read `out` back and check shapes / finiteness / non-degenerate variance / repeatability.
//
// Every step that cannot be done is reported with the exact symbol or value, not stubbed.
#if !defined(STRATA_ENABLE_VULKAN)
#error "layer_smoke.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"
#include "strata/vulkan/vk_backend.hpp"
#include "vk_arena.hpp"
#include <cuda_runtime.h>   // the shim's seam: cuda_compat_set_stream (the stand-in for CUDA's current device)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// A deterministic LCG, so a failure is reproducible from the seed alone.
uint32_t g_seed = 0x5EED1234u;
uint32_t next_u32() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return g_seed;
}
// ~[-1, 1)
float next_f32() { return (float) ((int32_t) (next_u32() >> 8) - (1 << 23)) / (float) (1 << 23); }

struct Row {
    std::string name;
    int file = 0;
    int kind = 0;               // 0 VERBATIM, 1 BF16_IN_F32, 2 F32
    uint64_t src_off = 0, src_bytes = 0, dst_off = 0, dst_bytes = 0;
    int64_t ne0 = 0, ne1 = 0;
    int code_bits = 0, code_bias = 0, group_elems = 0, codebook = 0, has_offset = 0;
    uint64_t codes_bytes = 0, scales_bytes = 0, offset_bytes = 0;
    int scales_fp16 = 0, act_kind = 0;
    bool quantized = false;
    uint64_t elems = 0;
};

void add_s8(Row& r, int64_t ne0, int64_t ne1) {   // Q8_0 legacy: 8 bits, group 32, bias -128, affine, no offset
    r.quantized = true;
    r.ne0 = ne0;
    r.ne1 = ne1;
    r.kind = 0;
    r.code_bits = 8;
    r.code_bias = -128;
    r.group_elems = 32;
    r.codebook = 0;
    r.has_offset = 0;
    r.act_kind = 0;            // wants the Q8_0 activation image -> s_gemv_q8_0_split
    r.codes_bytes = (uint64_t) ne0 * (uint64_t) ne1;
    r.scales_bytes = (uint64_t) ne1 * ((uint64_t) ne0 / 32) * 4;
    r.offset_bytes = 0;
    r.src_bytes = r.codes_bytes + r.scales_bytes;
    r.dst_bytes = r.src_bytes;
}

void add_f32(Row& r, int64_t ne0) {
    r.kind = 2;
    r.ne0 = ne0;
    r.ne1 = 0;
    r.elems = (uint64_t) ne0;
    r.src_bytes = r.elems * 4;
    r.dst_bytes = r.elems * 4;
}

void add_bf16(Row& r, int64_t ne0, int64_t ne1) {
    r.kind = 1;                // the pack holds the bf16 value promoted to f32
    r.ne0 = ne0;
    r.ne1 = ne1;
    r.elems = (uint64_t) ne0 * (uint64_t) ne1;
    r.src_bytes = r.elems * 4;
    r.dst_bytes = r.elems * 2;
}

bool write_pack(const std::string& packdir, std::vector<Row>& rows, std::string& err) {
    // ---- the raw file, laid out in row order; src_off recorded per row.  Values are drawn in a SMALL, FINITE
    // range: raw random bytes make NaN/Inf f32 SCALES (~0.8% of f32 bit patterns) and a NaN scale makes the
    // whole layer NaN - a fixture bug, not a kernel one.  `ssm_a` is drawn NEGATIVE because the delta-rule
    // recurrence's `dec = exp(softplus(...) * ssm_a)` overflows to Inf on a positive draw.
    auto push_f32 = [](std::vector<uint8_t>& f, float v) {
        uint32_t u;
        std::memcpy(&u, &v, 4);
        for (int b = 0; b < 4; ++b) f.push_back((uint8_t) (u >> (8 * b)));
    };
    std::vector<uint8_t> file;
    for (Row& r : rows) {
        r.src_off = file.size();
        const bool is_a = r.name.size() >= 5 && r.name.compare(r.name.size() - 5, 5, "ssm_a") == 0;
        if (r.quantized) {
            for (uint64_t i = 0; i < r.codes_bytes; ++i) file.push_back((uint8_t) next_u32());   // int8 codes
            for (uint64_t i = 0; i < r.scales_bytes / 4; ++i) push_f32(file, next_f32() * 0.4f); // f32 scales
        } else if (r.kind == 1) {   // BF16 held as f32: the loader keeps the high 16 bits
            for (uint64_t i = 0; i < r.elems; ++i) push_f32(file, next_f32() * 0.4f);
        } else {                    // F32 verbatim
            for (uint64_t i = 0; i < r.elems; ++i)
                push_f32(file, is_a ? -0.05f - 0.45f * (0.5f * (next_f32() + 1.0f)) : next_f32() * 0.4f);
        }
    }
    std::FILE* fb = std::fopen((packdir + "/dense.bin").c_str(), "wb");
    if (!fb) { err = "cannot write dense.bin"; return false; }
    const bool okf = std::fwrite(file.data(), 1, file.size(), fb) == file.size();
    std::fclose(fb);
    if (!okf) { err = "short write to dense.bin"; return false; }

    // ---- the arena layout: 256-byte aligned destinations.
    uint64_t at = 0;
    for (Row& r : rows) {
        r.dst_off = (at + 255) & ~(uint64_t) 255;
        at = r.dst_off + r.dst_bytes;
    }
    const uint64_t pool = (at + 255) & ~(uint64_t) 255;

    std::FILE* fi = std::fopen((packdir + "/index.txt").c_str(), "wb");
    if (!fi) { err = "cannot write index.txt"; return false; }
    std::fprintf(fi, "# strata pack index v3  --  generated by layer_smoke.cpp (SYNTHETIC random weights)\n");
    std::fprintf(fi, "# align 256 pool %llu tensors %zu\n", (unsigned long long) pool, rows.size());
    std::fprintf(fi, "# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias "
                     "group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind\n");
    for (const Row& r : rows)
        std::fprintf(fi,
                     "%s %d %d %llu %llu %llu %llu %lld %lld %d %d %d %d %d %llu %llu %llu %d %d\n",
                     r.name.c_str(), r.file, r.kind, (unsigned long long) r.src_off,
                     (unsigned long long) r.src_bytes, (unsigned long long) r.dst_off,
                     (unsigned long long) r.dst_bytes, (long long) r.ne0, (long long) r.ne1, r.code_bits,
                     r.code_bias, r.group_elems, r.codebook, r.has_offset, (unsigned long long) r.codes_bytes,
                     (unsigned long long) r.scales_bytes, (unsigned long long) r.offset_bytes, r.scales_fp16,
                     r.act_kind);
    std::fclose(fi);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "shaders";
    std::printf("layer_smoke: spv dir %s\n", dir.c_str());

    // ---- 1. DEVICE + ARENA --------------------------------------------------------------------------------
    strata::vulkan::Stream* s = strata::vulkan::stream_open(256ull << 20, dir);
    if (s == nullptr) {
        std::printf("STOP: device/arena creation failed (stream_open returned null)\n");
        return 3;
    }
    std::printf("STEP 1 ok: device + arena opened (256 MiB)\n");
    strata::vulkan::cuda_compat_set_stream(s);   // the shim's "current device": the stream-less cudaMalloc/upload calls

    // ---- A SMALL GEOMETRY that satisfies every GDN contract (channels%128==0, n_embd%8==0, S==128,
    //      h_v%h_k==0).  Small on purpose: the check is that the LAYER runs, not that it runs at 2560x10240.
    strata::core::ModelGeometry g;
    g.n_embd = 256;            // must be a multiple of the Q8_K block (256): gdn_layer always builds x_q8k
    g.ssm_state_size = 128;
    g.ssm_k_heads = 2;
    g.ssm_v_heads = 4;
    g.ssm_d_conv = 4;
    g.ssm_conv_channels = 2 * g.ssm_state_size * g.ssm_k_heads + g.ssm_state_size * g.ssm_v_heads;   // 1024
    g.ssm_value_dim = g.ssm_state_size * g.ssm_v_heads;                                                // 512
    std::printf("STEP 2: geometry n_embd=%lld C=%lld S=%lld h_k=%lld h_v=%lld V=%lld\n",
                (long long) g.n_embd, (long long) g.ssm_conv_channels, (long long) g.ssm_state_size,
                (long long) g.ssm_k_heads, (long long) g.ssm_v_heads, (long long) g.ssm_value_dim);

    // ---- 2. THE SYNTHETIC RANDOM PACK and the WeightTable -------------------------------------------------
    const char* tmp = std::getenv("TMPDIR");
    const std::string packdir = std::string(tmp ? tmp : "/tmp") + "/strata_mb_pack";
    std::string mk = "mkdir -p " + packdir;
    if (std::system(mk.c_str()) != 0) { std::printf("STOP: cannot create %s\n", packdir.c_str()); return 3; }
    std::vector<Row> rows;
    auto add = [&](const char* suffix, Row r) { r.name = "blk.0." + std::string(suffix); rows.push_back(r); };
    { Row r; add_s8(r, g.n_embd, g.ssm_conv_channels); add("attn_qkv.weight", r); }
    { Row r; add_s8(r, g.n_embd, g.ssm_value_dim); add("attn_gate.weight", r); }
    { Row r; add_s8(r, g.ssm_value_dim, g.n_embd); add("ssm_out.weight", r); }
    { Row r; add_bf16(r, g.n_embd, g.ssm_v_heads); add("ssm_alpha.weight", r); }
    { Row r; add_bf16(r, g.n_embd, g.ssm_v_heads); add("ssm_beta.weight", r); }
    { Row r; add_f32(r, g.ssm_conv_channels * g.ssm_d_conv); add("ssm_conv1d.weight", r); }
    { Row r; add_f32(r, g.ssm_state_size); add("ssm_norm.weight", r); }
    { Row r; add_f32(r, g.ssm_v_heads); add("ssm_dt.bias", r); }
    { Row r; add_f32(r, g.ssm_v_heads); add("ssm_a", r); }
    std::string err;
    if (!write_pack(packdir, rows, err)) { std::printf("STOP: synthetic pack: %s\n", err.c_str()); return 3; }
    uint64_t pool = 0;
    if (!strata::core::WeightTable::pool_bytes(packdir, pool, err)) {
        std::printf("STOP: pool_bytes: %s\n", err.c_str());
        return 3;
    }
    std::printf("STEP 3: synthetic pack at %s, pool %llu B, %zu tensors\n", packdir.c_str(),
                (unsigned long long) pool, rows.size());

    const uint64_t wbytes = pool + (1ull << 20);
    void* wbase = strata::vulkan::arena_alloc(*s, wbytes);
    strata::core::WeightTable tables;
    if (!tables.load(packdir, wbase, wbytes, err)) {
        std::printf("STOP: WeightTable::load refused the synthetic pack: %s\n", err.c_str());
        return 4;
    }
    std::printf("STEP 4 ok: WeightTable loaded %zu tensors (random weights)\n", tables.all().size());

    // ---- 3. THE LAYER'S BUFFERS ---------------------------------------------------------------------------
    const uint64_t bytes = strata::core::gdn_buffers_bytes(g);
    void* base = strata::vulkan::arena_alloc(*s, bytes);
    strata::core::GdnBuffers b;
    strata::core::gdn_buffers_init(g, base, b);
    strata::core::gdn_buffers_zero_state(b, g, s);
    std::printf("STEP 5 ok: GdnBuffers carved (%llu B) and state zeroed\n", (unsigned long long) bytes);

    // ---- 4. THE LAYER BODY, on random weights -----------------------------------------------------------------
    std::vector<float> mixed((size_t) g.n_embd);
    for (float& v : mixed) v = next_f32();
    float* dmixed = strata::vulkan::arena_alloc<float>(*s, (size_t) g.n_embd);
    float* dout1 = strata::vulkan::arena_alloc<float>(*s, (size_t) g.n_embd);
    float* dout2 = strata::vulkan::arena_alloc<float>(*s, (size_t) g.n_embd);
    strata::vulkan::stream_write(*s, dmixed, mixed.data(), mixed.size() * 4);

    const bool ok1 = strata::core::gdn_layer(tables, g, /*layer=*/0, b, dmixed, dout1, s, err);
    std::printf("STEP 6: gdn_layer(...) #1 returned %d; err=\"%s\"\n", (int) ok1, err.c_str());
    if (!ok1) {
        std::printf("STOP: the layer body ran and refused: %s\n", err.c_str());
        strata::vulkan::stream_close(s);
        return 5;
    }
    std::vector<float> out1((size_t) g.n_embd);
    strata::vulkan::stream_read(*s, dout1, out1.data(), out1.size() * 4);

    // ---- 5a. INPUT DEPENDENCE (a cheap rival): a DIFFERENT activation must move the output.  Without this the
    //          output could be a constant that ignores `mixed` entirely and every check above would still pass.
    std::vector<float> mixed2(mixed);
    for (float& v : mixed2) v += 0.25f;
    strata::vulkan::stream_write(*s, dmixed, mixed2.data(), mixed2.size() * 4);
    strata::core::gdn_buffers_zero_state(b, g, s);
    const bool ok2b = strata::core::gdn_layer(tables, g, 0, b, dmixed, dout2, s, err);
    std::vector<float> out2b((size_t) g.n_embd);
    strata::vulkan::stream_read(*s, dout2, out2b.data(), out2b.size() * 4);
    int input_moved = 0;
    for (size_t i = 0; i < out1.size(); ++i) if (out1[i] != out2b[i]) ++input_moved;
    // restore the original activation for the repeatability arm
    strata::vulkan::stream_write(*s, dmixed, mixed.data(), mixed.size() * 4);

    // ---- 5b. REPEATABILITY: the SAME weights and the SAME input from a FIXED starting state (the state is
    //          mutated in place, so it must be re-zeroed between runs for the comparison to mean anything).
    strata::core::gdn_buffers_zero_state(b, g, s);
    const bool ok2 = strata::core::gdn_layer(tables, g, 0, b, dmixed, dout2, s, err);
    std::vector<float> out2((size_t) g.n_embd);
    strata::vulkan::stream_read(*s, dout2, out2.data(), out2.size() * 4);

    // ---- 6. THE CHECKS -------------------------------------------------------------------------------------
    int finite = 0, repeat = 0;
    double lo = 1e30, hi = -1e30, mean = 0.0;
    for (size_t i = 0; i < out1.size(); ++i) {
        const float v = out1[i];
        if (std::isfinite(v)) ++finite;
        lo = std::min(lo, (double) v);
        hi = std::max(hi, (double) v);
        mean += v;
        if (std::memcmp(&out1[i], &out2[i], 4) == 0) ++repeat;
    }
    mean /= (double) out1.size();
    double var = 0.0;
    for (float v : out1) var += ((double) v - mean) * ((double) v - mean);
    var /= (double) out1.size();

    std::printf("STEP 7 (evidence):\n");
    std::printf("  shape: out has %zu floats (expected n_embd=%lld)\n", out1.size(), (long long) g.n_embd);
    std::printf("  finite: %d/%zu\n", finite, out1.size());
    std::printf("  range: [%.6g, %.6g]  mean %.6g  variance %.6g\n", lo, hi, mean, var);
    std::printf("  input dependence (a DIFFERENT activation): %d/%zu output elements moved (run #2b ret %d)\n",
                input_moved, out1.size(), (int) ok2b);
    std::printf("  repeatability (run #3 after re-zeroed state): %d/%zu elements bitwise equal to run #1\n", repeat,
                out1.size());
    std::printf("  run #3 returned %d\n", (int) ok2);
    const bool pass = (out1.size() == (size_t) g.n_embd) && finite == (int) out1.size() && ok2 && ok2b &&
                      repeat == (int) out1.size() && input_moved > 0 && var > 0.0 && std::isfinite(var);
    std::printf("%s: ONE GDN layer body executed on the Vulkan backend with random weights\n",
                pass ? "PASS" : "FAIL");
    std::printf("WHAT THIS DOES NOT PROVE: numerical correctness of the layer.  It executed and produced finite,\n"
                "non-degenerate, repeatable output of the right shape; there is NO reference comparison here (no\n"
                "cheap oracle for a random-weight layer), so agreement with the engine's CPU/CUDA path is NOT\n"
                "established.  The per-kernel correctness is the gate's job, not this target's.\n");
    strata::vulkan::stream_close(s);
    return pass ? 0 : 6;
}
