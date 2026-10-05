// vulkan/src/kernels/elementwise_vk.cpp - the Vulkan backend's ELEMENTWISE GLUE entry points (I2).
//
// ============================================================================================================
// WHICH THREE, AND WHY THESE THREE (derived from the layer body, not from the plan's list)
// ============================================================================================================
//
// The plan's I2 list (`plan/BACKEND-INTEGRATION.md` §3) names eleven kernel rows.  This TU wires the first
// three the layer chain reaches IN EXECUTION ORDER, read out of `src/core/layer.cpp`'s layer body:
//
//   block_layer (layer.cpp:1344) -> block_layer_pre -> the MIXER -> ... -> block_layer_post
//
// and the mixer for 36 of the 48 layers is `gdn_layer` (layer.cpp:223).  Its body reaches the glue in this
// order, and the first three are:
//
//   1. `silu_inplace`      layer.cpp:257   gdn_layer stage 10 - SiLU of the conv output (`b.h`), the
//                                          legacy branch of the conv+SiLU pair (`!native_gdn_enabled()`).
//   2. `scale_inplace`     layer.cpp:276   gdn_layer stage 11 - the recurrence's extra 1/sqrt(S) on q
//                                          (`b.h`, qk floats), again the legacy branch.  Its own comment
//                                          says the scale is a backend-specific boundary: the native CUDA
//                                          applies it after the readout dot, this legacy path before.
//   3. `f32_to_bf16_bulk`  layer.cpp:290   gdn_layer stage 12 - the alpha/beta activation, reached by
//                                          DEFAULT (`native_bf16_projections` defaults false, layer.cpp:91)
//                                          and again at moe_route:357, moe_shared:430 and qsa_layer:900.
//
// The plan's list is alphabetical in part and is NOT an order: reading it as one would start with
// `add_inplace` (which the layer body never calls - its call sites are `expert_source.cpp:2353` and
// `remote_expert_opt.cu:127`, the R4 hit/miss split) then `embedding_gather` (a token-level lookup, and at
// `layer.cpp:1083` inside `embed_row`, before layer 0).  The three above are the first three the LAYER body
// itself reaches.
//
// THE NEXT THREE, in the same reading, are `gdn_gate` (layer.cpp:300, stage 12), `rms_norm_weighted`
// (layer.cpp:880, the QSA mixer's norm) and `embedding_gather` (layer.cpp:1083).  They are I2's remainder.
//
// ============================================================================================================
// THE WIRING PATTERN (the plan's §2, not an invention)
// ============================================================================================================
//
// The engine's HEADERS ARE NOT EDITED.  Each entry point is the thin wrapper already declared in
// `include/strata/kernels/elementwise.hpp`; this TU answers the `strata::kernels::` symbol the wrapper
// calls.  On a CUDA/HIP build `src/kernels/cuda/elementwise.cu` answers them; on this build THIS file does.
// The bodies resolve the engine's raw device pointers to arena views (vk_arena.hpp), take the pipeline the
// device layer caches for the shader's own signature, and dispatch - the same shape as `fwht_vk.cpp`, and
// the shader each one drives is the shader the port's numeric gate has already gated:
//
//     scale_inplace      -> scale.spv        (case_scale,       gate verdict "scale_inplace")
//     silu_inplace       -> silu_f32.spv     (case_silu,        gate verdict "silu_inplace (f32 vs double ref)")
//     f32_to_bf16_bulk   -> f32_to_bf16.spv  (run_conversion<>,"f32_to_bf16 (bit-exact)")
//
// The gate's `case_*_entry` cases (I2) re-run each with the ENGINE WRAPPER and compare BITWISE to that same
// shader path, so this file's claim is not "it compiles" but "the wrapper's answer equals the ported
// shader's answer".
#if !defined(STRATA_ENABLE_VULKAN)
#error "elementwise_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/elementwise.hpp"  // the engine's wrappers: silu_inplace / scale_inplace / f32_to_bf16_bulk
#include "strata/vulkan/vk_backend.hpp"    // the backend's seam: Stream, stream_of, the elementwise decls
#include "vk_arena.hpp"                    // the arena + pointer->buffer resolution

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::vulkan {

// The port's elementwise shaders are all `local_size_x = 256` with one element per invocation (checked
// against the host's constant by the gate's shader census), so a grid is the element count over 256.
static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// `x[i] /= 1 + exp(-x[i])`, in place (shader silu_f32.spv: 1 storage buffer, push {int n}).
void silu_inplace(Stream& s, float* x, int64_t n) {
    if (n <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv)) {
        std::fprintf(stderr, "strata::vulkan::silu_inplace: x is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::silu_inplace: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/silu_f32.spv", 1, sizeof(int32_t));
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `x[i] *= s`, in place (shader scale.spv: 1 storage buffer, push {int n; float s}).
void scale_inplace(Stream& s, float* x, int64_t n, float factor) {
    if (n <= 0) return;
    Buf xv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv)) {
        std::fprintf(stderr, "strata::vulkan::scale_inplace: x is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::scale_inplace: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/scale.spv", 1, 8);
    struct Push {
        int32_t n;
        float s;
    } pc{};
    pc.n = (int32_t) n;
    pc.s = factor;
    s.ctx->dispatch(pipe, {&xv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `y[i] = bf16(x[i])` (shader f32_to_bf16.spv: X float[] read, Y uint16_t[] write, push {int n}).
void f32_to_bf16_bulk(Stream& s, const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n * sizeof(float), xv) ||
        !arena_resolve(s, y, (uint64_t) n * sizeof(uint16_t), yv)) {
        std::fprintf(stderr, "strata::vulkan::f32_to_bf16_bulk: x or y is not inside this stream's arena "
                             "(n=%lld) - refusing rather than binding a wrong view\n",
                     (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::f32_to_bf16_bulk: n=%lld overflows the shader's int\n",
                     (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/f32_to_bf16.spv", 2, sizeof(int32_t));
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `y[t][h] = softplus(alpha[t][h] + dt[h]) * ssm_a[h]`, one invocation per (t, h) (shader gdn_gate.spv: 4 float[]
// buffers ALPHA/DT/SSM_A/GATE, push {int h_v; int n_tokens}).  The shader carries the port's own log1p series
// (its header records the two failed forms), so this wrapper only binds and sizes.
void gdn_gate(Stream& s, const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const uint64_t n = (uint64_t) n_tokens * (uint64_t) h_v;
    Buf av{}, dv{}, sv{}, gv{};
    if (!arena_resolve(s, alpha, n * 4, av) || !arena_resolve(s, dt, (uint64_t) h_v * 4, dv) ||
        !arena_resolve(s, ssm_a, (uint64_t) h_v * 4, sv) || !arena_resolve(s, gate, n * 4, gv)) {
        std::fprintf(stderr, "strata::vulkan::gdn_gate: a pointer is not inside this stream's arena "
                             "(n_tokens=%lld h_v=%lld) - refusing rather than binding a wrong view\n",
                     (long long) n_tokens, (long long) h_v);
        std::exit(2);
    }
    if (n > INT32_MAX || h_v > INT32_MAX || n_tokens > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::gdn_gate: a dimension overflows the shader's int\n");
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gdn_gate.spv", 4, 8);
    struct Push {
        int32_t h_v;
        int32_t n_tokens;
    } pc{};
    pc.h_v = (int32_t) h_v;
    pc.n_tokens = (int32_t) n_tokens;
    s.ctx->dispatch(pipe, {&av, &dv, &sv, &gv}, &pc, sizeof(pc), groups_for(n));
}

// `x[r][c] = x[r][c]/sqrt(MEAN_c(x[r]^2)+eps) * w[c]`, over the last axis (shader rms_norm.spv: X read-write,
// W read, push {int rows; int cols; float eps}; ONE WORKGROUP PER ROW - see the shader header for why the
// subgroup-per-row form was replaced).  The CUDA contract allows `w == nullptr`; Vulkan has no null descriptor,
// so ones are supplied and the arithmetic is identical (`r * 1.0f * inv == r * inv` in IEEE).
void rms_norm_weighted(Stream& s, float* x, const float* w, int64_t rows, int64_t cols, float eps) {
    if (rows <= 0 || cols <= 0) return;
    const uint64_t n = (uint64_t) rows * (uint64_t) cols;
    Buf xv{};
    if (!arena_resolve(s, x, n * 4, xv)) {
        std::fprintf(stderr, "strata::vulkan::rms_norm_weighted: x is not inside this stream's arena "
                             "(rows=%lld cols=%lld) - refusing rather than binding a wrong view\n",
                     (long long) rows, (long long) cols);
        std::exit(2);
    }
    Buf wv{};
    bool own_ones = false;
    if (w != nullptr) {
        if (!arena_resolve(s, w, n * 4, wv)) {
            std::fprintf(stderr, "strata::vulkan::rms_norm_weighted: w is not inside this stream's arena\n");
            std::exit(2);
        }
    } else {
        // The engine's unweighted form.  A weight buffer is required here, so allocate ones for this call.
        wv = s.ctx->alloc(n * 4);
        std::vector<float> ones((size_t) n, 1.0f);
        s.ctx->write(wv, ones.data(), n * 4);
        own_ones = true;
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/rms_norm.spv", 2, 12);
    struct Push {
        int32_t rows;
        int32_t cols;
        float eps;
    } pc{};
    pc.rows = (int32_t) rows;
    pc.cols = (int32_t) cols;
    pc.eps = eps;
    s.ctx->dispatch(pipe, {&xv, &wv}, &pc, sizeof(pc), (uint32_t) rows);
    if (own_ones) s.ctx->free(wv);
}

// The token embedding: packed codes + per-group scales (+ optional offsets) -> ONE float row (the CUDA entry
// point decodes one plane; the shader is the general form, so this uses its single-plane mode: has_tokens=0).
// Shader embedding_gather.spv: 5 buffers CODES/SCALES/OFFSETS/TOKENS/OUT, push PC (below).  The packing is
// LSB-first and the multiply/add are TWO roundings (`precise` in the shader) - see the shader header; this
// wrapper passes them through and must not "improve" either.
void embedding_gather(Stream& s, const uint8_t* codes, const float* scales, const float* offsets, int64_t n,
                      int code_bits, int code_bias, int group_elems, float* out) {
    if (n <= 0) return;
    if (code_bits != 2 && code_bits != 4 && code_bits != 8) {
        std::fprintf(stderr, "strata::vulkan::embedding_gather: code_bits=%d is not 2, 4 or 8\n", code_bits);
        std::exit(2);
    }
    if (group_elems <= 0) {
        std::fprintf(stderr, "strata::vulkan::embedding_gather: group_elems=%d must be positive\n", group_elems);
        std::exit(2);
    }
    const int per_byte = 8 / code_bits;
    const uint64_t row_codes = (uint64_t) ((n + per_byte - 1) / per_byte);   // packed bytes in this row
    const uint64_t row_groups = (uint64_t) ((n + group_elems - 1) / group_elems);
    Buf cv{}, sv{}, ov{}, tv{}, yv{};
    if (!arena_resolve(s, codes, row_codes, cv) || !arena_resolve(s, scales, row_groups * 4, sv) ||
        !arena_resolve(s, out, (uint64_t) n * 4, yv)) {
        std::fprintf(stderr, "strata::vulkan::embedding_gather: codes/scales/out is not inside this stream's "
                             "arena - refusing rather than binding a wrong view\n");
        std::exit(2);
    }
    bool own_off = false, own_tok = false;
    if (offsets != nullptr) {
        if (!arena_resolve(s, offsets, row_groups * 4, ov)) {
            std::fprintf(stderr, "strata::vulkan::embedding_gather: offsets is not inside this stream's arena\n");
            std::exit(2);
        }
    } else {
        ov = s.ctx->alloc(4);                       // has_offsets = 0: the shader never reads it, but must bind it
        own_off = true;
    }
    tv = s.ctx->alloc(4);                           // has_tokens = 0: the TOKENS binding is never read
    own_tok = true;
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::embedding_gather: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    struct Pc {
        int32_t n, code_bits, code_bias, group_elems;
        uint32_t row_codes, row_groups;
        int32_t has_offsets, has_tokens, single_token;
    } pc{};
    pc.n = (int32_t) n;
    pc.code_bits = code_bits;
    pc.code_bias = code_bias;
    pc.group_elems = group_elems;
    pc.row_codes = (uint32_t) row_codes;
    pc.row_groups = (uint32_t) row_groups;
    pc.has_offsets = (offsets != nullptr) ? 1 : 0;
    pc.has_tokens = 0;
    pc.single_token = 0;
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/embedding_gather.spv", 5, sizeof(pc));
    s.ctx->dispatch(pipe, {&cv, &sv, &ov, &tv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
    if (own_off) s.ctx->free(ov);
    if (own_tok) s.ctx->free(tv);
}

}  // namespace strata::vulkan

// ---- the engine's entry points: the symbols include/strata/kernels/elementwise.hpp declares ---------------
namespace strata::kernels {

// elementwise.hpp: `void silu_inplace(float* x, int64_t n, void* stream);`
void silu_inplace(float* x, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "silu_inplace: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::silu_inplace(*s, x, n);
}

// elementwise.hpp: `void scale_inplace(float* x, int64_t n, float s, void* stream);`
void scale_inplace(float* x, int64_t n, float s, void* stream) {
    strata::vulkan::Stream* st = strata::vulkan::stream_of(stream);
    if (st == nullptr) {
        std::fprintf(stderr, "scale_inplace: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::scale_inplace(*st, x, n, s);
}

// elementwise.hpp: `void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);`
void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "f32_to_bf16_bulk: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::f32_to_bf16_bulk(*s, x, y, n);
}

// elementwise.hpp: `void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate,
//                                int64_t n_tokens, int64_t h_v, void* stream);`   (layer.cpp:300)
void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens, int64_t h_v,
              void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gdn_gate: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gdn_gate(*s, alpha, dt, ssm_a, gate, n_tokens, h_v);
}

// elementwise.hpp: `void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps,
//                                          void* stream);`   (layer.cpp:880)
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "rms_norm_weighted: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::rms_norm_weighted(*s, x, w, rows, cols, eps);
}

// elementwise.hpp: `void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
//                                         int64_t n, int code_bits, int code_bias, int group_elems, float* out,
//                                         void* stream);`   (layer.cpp:1083)
void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets, int64_t n, int code_bits,
                      int code_bias, int group_elems, float* out, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "embedding_gather: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::embedding_gather(*s, codes, scales, offsets, n, code_bits, code_bias, group_elems, out);
}

}  // namespace strata::kernels
