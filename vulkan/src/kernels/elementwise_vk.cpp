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
// (layer.cpp:880, the QSA mixer's norm) and `embedding_gather` (layer.cpp:1083).  They landed in I2's
// continuation; the FIVE after them (add_inplace, scatter_rows_f32, cvec_apply, gather_rows, f32_to_f16_bulk -
// the order the FORWARD PATH's call sites give, NOT the plan's list) are the block further down this file.
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
#include "strata/kernels/verify_kernels.hpp"  // the engine's wrapper: gather_rows
#include "strata/kernels/cvec.hpp"         // the engine's wrapper + module: cvec_apply / cvec() / cvec_upload
#include "strata/vulkan/vk_backend.hpp"    // the backend's seam: Stream, stream_of, the elementwise decls
#include "vk_arena.hpp"                    // the arena + pointer->buffer resolution

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
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

// ============================================================================================================
// I2, CONTINUED FURTHER - THE NEXT FIVE ENTRY POINTS THE FORWARD PATH REACHES
// ============================================================================================================
//
// Read from the decode path, not from the plan's list.  The layer body (`block_layer`, layer.cpp:1344 ->
// `block_layer_pre` / `block_layer_post`) reaches these in this order:
//
//   1. `add_inplace`      the MoE expert pool's hit combine (`parts += hit_out`), expert_source.cpp:2353,
//                         inside the layer's stage-4 MoE (moe_route, layer.cpp:1282) -> the host pool.
//   2. `scatter_rows_f32` the peer experts' row write-back, peer_experts.cpp:241 (the same MoE region, the
//                         multi-GPU peer path).
//   3. `cvec_apply`       layer.cpp:1330/:1333/:1336, `block_layer_post` stage 6 - the ONLY one of the five
//                         layer.cpp calls DIRECTLY (the per-layer control-vector apply).
//   4. `gather_rows`      the MTP draft head's token gather, mtp.cpp:450 - the speculative DRAFT pass, not the
//                         layer body.  `f32_to_f16_bulk` is 5th and has NO `src/core/` call site at all
//                         (reached only from src/kernels/elementwise_parity.cpp:202); it is wired because the
//                         plan's I2 list names it and elementwise.hpp carries the contract.
//
// FOUR ARE THIN wrappers over an already-gated shader (add.spv, scatter_rows_f32.spv, gather_rows.spv,
// f32_to_f16.spv).  THE FIFTH IS NOT: `cvec_apply` reads MODULE STATE (`strata::kernels::cvec()`) that the
// engine owns in cvec.cu - the direction/scale/switch tables, mode, first/last, n_embd/hc - so this TU also
// answers the cvec `host` row (`cvec`, `cvec_upload`, `cvec_replicate`, `cvec_set_enabled`, `cvec_enabled`).
// The DEVICE tables are placed LAZILY into each Stream's arena (see cvec_apply), because cvec_upload carries
// no stream and the CUDA's per-"current device" table has no Vulkan analogue here.

// `dst[i] += src[i]` (shader add.spv: DST read-write, SRC read; push {int n}).
void add_inplace(Stream& s, float* dst, const float* src, int64_t n) {
    if (n <= 0) return;
    Buf dv{}, sv{};
    if (!arena_resolve(s, dst, (uint64_t) n * 4, dv) || !arena_resolve(s, src, (uint64_t) n * 4, sv)) {
        std::fprintf(stderr, "strata::vulkan::add_inplace: dst or src is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n", (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::add_inplace: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/add.spv", 2, 4);
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&dv, &sv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `y[i] = f16(x[i])` (shader f32_to_f16.spv: X float[] read, Y uint16_t[] write, push {int n}) - the F16
// sibling of f32_to_bf16_bulk below, a SEPARATE entry point (5 vs 8 exponent bits; a wrong pick is a plausible
// tensor at the wrong precision).
void f32_to_f16_bulk(Stream& s, const float* x, uint16_t* y, int64_t n) {
    if (n <= 0) return;
    Buf xv{}, yv{};
    if (!arena_resolve(s, x, (uint64_t) n * 4, xv) || !arena_resolve(s, y, (uint64_t) n * 2, yv)) {
        std::fprintf(stderr, "strata::vulkan::f32_to_f16_bulk: x or y is not inside this stream's arena (n=%lld) - "
                             "refusing rather than binding a wrong view\n", (long long) n);
        std::exit(2);
    }
    if (n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::f32_to_f16_bulk: n=%lld overflows the shader's int\n", (long long) n);
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/f32_to_f16.spv", 2, 4);
    struct Push {
        int32_t n;
    } pc{};
    pc.n = (int32_t) n;
    s.ctx->dispatch(pipe, {&xv, &yv}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// Resolve from a pointer to the LIVE END of the arena.  `gather_rows` and `scatter_rows_f32` carry no size for
// one of their tables (the CUDA's need none): the source table / destination region is bound as the span from
// the pointer to `bump`, so every row the ids name is covered and nothing outside the carved region is bound.
static bool arena_resolve_span(const Stream& s, const void* p, Buf& out) {
    if (p == nullptr) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (a < Stream::kArenaBase) return false;
    const uint64_t off = (uint64_t) (a - Stream::kArenaBase);
    if (off >= s.bump) return false;
    return arena_resolve(s, p, s.bump - off, out);
}

// `dst[r][o] = src[ids[r]][o]`, one byte per invocation over the n*row_bytes output (shader gather_rows.spv:
// SRC, IDS, DST; push {uint row_bytes; uint n}).  The CUDA's uint4/uint32/uint8 element WIDTH is a performance
// choice, NOT the rule - the shader is the byte form, so this wrapper imposes no width or alignment.
void gather_rows(Stream& s, const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst) {
    if (n <= 0 || row_bytes <= 0) return;
    const uint64_t total = (uint64_t) n * (uint64_t) row_bytes;   // output bytes = the flat dispatch index space
    Buf sv{}, iv{}, dv{};
    if (!arena_resolve_span(s, src, sv) || !arena_resolve(s, ids, (uint64_t) n * 4, iv) ||
        !arena_resolve(s, dst, total, dv)) {
        std::fprintf(stderr, "strata::vulkan::gather_rows: src/ids/dst is not inside this stream's arena "
                             "(n=%lld row_bytes=%lld) - refusing rather than binding a wrong view\n",
                     (long long) n, (long long) row_bytes);
        std::exit(2);
    }
    if (total > 0xFFFFFFFFull || row_bytes > INT32_MAX || n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::gather_rows: n*row_bytes or n overflows the shader's uint\n");
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/gather_rows.spv", 3, 8);
    struct Push {
        uint32_t row_bytes;
        uint32_t n;
    } pc{};
    pc.row_bytes = (uint32_t) row_bytes;
    pc.n = (uint32_t) n;
    s.ctx->dispatch(pipe, {&sv, &iv, &dv}, &pc, sizeof(pc), groups_for(total));
}

// `dst[rows[r]][i] = src[r][i]`, ONE WORKGROUP per source row (shader scatter_rows_f32.spv: SRC, DST, ROWS;
// push {uint width; uint n}).  `r` is a POSITION and `rows[r]` the DESTINATION; a destination row nobody names
// is untouched.  The CUDA REFUSES `width % 4 != 0` and unaligned pointers (its float4 cast - a performance
// choice); this wrapper imposes NEITHER, so the port's width=6 arm is legal here.
void scatter_rows_f32(Stream& s, const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width) {
    if (n <= 0 || width <= 0) return;
    Buf sv{}, dv{}, rv{};
    if (!arena_resolve(s, src, (uint64_t) n * (uint64_t) width * 4, sv) || !arena_resolve_span(s, dst, dv) ||
        !arena_resolve(s, rows, (uint64_t) n * 4, rv)) {
        std::fprintf(stderr, "strata::vulkan::scatter_rows_f32: src/dst/rows is not inside this stream's arena "
                             "(n=%lld width=%lld) - refusing rather than binding a wrong view\n",
                     (long long) n, (long long) width);
        std::exit(2);
    }
    if (width > INT32_MAX || n > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::scatter_rows_f32: a dimension overflows the shader's uint\n");
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/scatter_rows_f32.spv", 3, 8);
    struct Push {
        uint32_t width;
        uint32_t n;
    } pc{};
    pc.width = (uint32_t) width;
    pc.n = (uint32_t) n;
    s.ctx->dispatch(pipe, {&sv, &dv, &rv}, &pc, sizeof(pc), (uint32_t) n);   // one workgroup per source row
}

// ---- THE CONTROL-VECTOR MODULE (the engine's cvec.hpp `host` row) -------------------------------------------
//
// cvec.cu owns this state on a CUDA build; a Vulkan build compiles no cvec.cu, so the backend answers the row.
// The HOST tables (direction, per-layer scale, the switch) are recorded by cvec_upload; the DEVICE tables are
// placed into the calling Stream's arena lazily by cvec_apply, at the generation the host state carries.  A
// new upload, a replicate or a switch change bumps the generation, so the next apply rebuilds the tables.
namespace {
struct CvecHost {
    std::vector<float> dir, s;      // n_layers * n_embd, and n_layers
    int mode = 0, first = 0, last = -1;
    int64_t n_embd = 0, hc = 0;
    bool on = true;
    bool loaded = false;
    uint64_t gen = 1;
};
CvecHost g_cvec_host;
kernels::Cvec g_cvec_pub;           // what strata::kernels::cvec() hands the engine
float g_cvec_sentinel_f = 0.0f;     // non-null stand-ins for the device pointers (loaded() only tests null)
int g_cvec_sentinel_i = 0;
}  // namespace

// `R[t][c][:] <- the layer's vector` (shader cvec_apply.spv: R rw, DIR, SL, ONV, BO, INJ; push PC below).
// Grid (hc, T): ONE WORKGROUP per (stream, token), matching the CUDA's block-per-(stream, token) shape.
void cvec_apply(Stream& s, float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
                const float* inj, int64_t inj_ld, bool write) {
    if (!g_cvec_host.loaded || T < 1) return;
    const CvecHost& h = g_cvec_host;
    if (h.n_embd < 1 || h.n_embd > 4096 || h.hc < 1) {
        std::fprintf(stderr, "strata::vulkan::cvec_apply: n_embd=%lld hc=%lld outside the shader's contract\n",
                     (long long) h.n_embd, (long long) h.hc);
        std::exit(2);
    }
    if (layer < 0 || layer >= (int64_t) h.s.size()) {
        std::fprintf(stderr, "strata::vulkan::cvec_apply: layer=%lld outside the vector's %zu layers\n",
                     (long long) layer, h.s.size());
        std::exit(2);
    }
    // Place (or replace) this stream's tables at the current generation.
    if (!s.cvec_tables.valid || s.cvec_tables.gen != h.gen) {
        if (s.cvec_tables.valid) {
            s.ctx->free(s.cvec_tables.dir);
            s.ctx->free(s.cvec_tables.s);
            s.ctx->free(s.cvec_tables.on);
        }
        const uint64_t dir_bytes = (uint64_t) h.dir.size() * 4;
        const uint64_t s_bytes = (uint64_t) h.s.size() * 4;
        s.cvec_tables.dir = s.ctx->alloc(dir_bytes ? dir_bytes : 4);
        s.cvec_tables.s = s.ctx->alloc(s_bytes ? s_bytes : 4);
        s.cvec_tables.on = s.ctx->alloc(4);
        s.ctx->write(s.cvec_tables.dir, h.dir.data(), dir_bytes);
        s.ctx->write(s.cvec_tables.s, h.s.data(), s_bytes);
        const int32_t onv = h.on ? 1 : 0;
        s.ctx->write(s.cvec_tables.on, &onv, 4);
        s.cvec_tables.gen = h.gen;
        s.cvec_tables.valid = true;
    }
    Buf rv{};
    if (!arena_resolve(s, R, (uint64_t) T * (uint64_t) r_ld * 4, rv)) {
        std::fprintf(stderr, "strata::vulkan::cvec_apply: R is not inside this stream's arena (T=%lld r_ld=%lld)\n",
                     (long long) T, (long long) r_ld);
        std::exit(2);
    }
    Buf bov{}, injv{};
    if (write) {
        if (!arena_resolve(s, bo, (uint64_t) T * (uint64_t) bo_ld * 4, bov) ||
            !arena_resolve(s, inj, (uint64_t) T * (uint64_t) inj_ld * 4, injv)) {
            std::fprintf(stderr, "strata::vulkan::cvec_apply: bo/inj is not inside this stream's arena\n");
            std::exit(2);
        }
    } else {
        // write == 0: the shader never reads BO/INJ, but Vulkan has no null descriptor, so a dummy is bound.
        // Allocate ONCE per stream and reuse it (the engine calls this every layer with bo == nullptr).
        if (!s.cvec_tables.valid || s.cvec_tables.dummy.buffer == VK_NULL_HANDLE) s.cvec_tables.dummy = s.ctx->alloc(4);
        bov = s.cvec_tables.dummy;
        injv = s.cvec_tables.dummy;
    }
    if (r_ld > INT32_MAX || bo_ld > INT32_MAX || inj_ld > INT32_MAX || T > INT32_MAX) {
        std::fprintf(stderr, "strata::vulkan::cvec_apply: a dimension overflows the shader's int\n");
        std::exit(2);
    }
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/cvec_apply.spv", 6, 32);
    struct Push {
        int32_t mode, layer, n, hc, r_ld, bo_ld, inj_ld, write;
    } pc{};
    pc.mode = h.mode;
    pc.layer = (int32_t) layer;
    pc.n = (int32_t) h.n_embd;
    pc.hc = (int32_t) h.hc;
    pc.r_ld = (int32_t) r_ld;
    pc.bo_ld = (int32_t) bo_ld;
    pc.inj_ld = (int32_t) inj_ld;
    pc.write = write ? 1 : 0;
    s.ctx->dispatch(pipe, {&rv, &s.cvec_tables.dir, &s.cvec_tables.s, &s.cvec_tables.on, &bov, &injv},
                    &pc, sizeof(pc), (uint32_t) h.hc, (uint32_t) T);
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

// ---- I2 continuation: the next five engine entry points (see the block above for the order and call sites) --

// elementwise.hpp: `void add_inplace(float* dst, const float* src, int64_t n, void* stream);`
// (expert_source.cpp:2353, the MoE expert pool's hit combine)
void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "add_inplace: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::add_inplace(*s, dst, src, n);
}

// elementwise.hpp: `void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream);`
void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "f32_to_f16_bulk: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::f32_to_f16_bulk(*s, x, y, n);
}

// verify_kernels.hpp: `void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n,
//                                         uint8_t* dst, void* stream);`   (mtp.cpp:450, the MTP draft head)
void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "gather_rows: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::gather_rows(*s, src, row_bytes, ids, n, dst);
}

// elementwise.hpp: `void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n,
//                                          int64_t width, void* stream);`   (peer_experts.cpp:241)
void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "scatter_rows_f32: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::scatter_rows_f32(*s, src, dst, rows, n, width);
}

// cvec.hpp: THE CONTROL-VECTOR MODULE.  cvec.cu owns these on a CUDA build; a Vulkan build compiles no cvec.cu,
// so the backend answers the engine's cvec `host` row (PORT-MAP.tsv: "the table the cvec_apply kernel reads").
// The HOST state is recorded here; cvec_apply places the device tables, lazily, into the calling Stream.
const Cvec& cvec() { return strata::vulkan::g_cvec_pub; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > 4096) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    strata::vulkan::CvecHost& h = strata::vulkan::g_cvec_host;
    h.dir = dir;
    h.s = s;
    h.mode = mode;
    h.first = first;
    h.last = last;
    h.n_embd = n_embd;
    h.hc = hc;
    h.on = true;
    h.loaded = true;
    ++h.gen;                                    // the tables on every stream are now stale
    Cvec& p = strata::vulkan::g_cvec_pub;
    // The published pointers are NON-NULL sentinels: the engine only tests them for null (`loaded()`); the real
    // device tables live per stream and are never dereferenced on the host (`covers()` reads `steered`).
    p.dir = &strata::vulkan::g_cvec_sentinel_f;
    p.s = &strata::vulkan::g_cvec_sentinel_f;
    p.on = &strata::vulkan::g_cvec_sentinel_i;
    p.mode = mode;
    p.first = first;
    p.last = last;
    p.n_embd = n_embd;
    p.hc = hc;
    p.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) p.steered[l] = s[l] != 0.0f;
    return true;
}

// A layer split: the CUDA re-uploads to the CURRENT device.  Here the tables are placed lazily per stream, so
// this only invalidates the placed copies; the next cvec_apply rebuilds them on whatever stream it runs on.
bool cvec_replicate(std::string& err) {
    (void) err;
    if (strata::vulkan::g_cvec_host.loaded) ++strata::vulkan::g_cvec_host.gen;
    return true;
}

void cvec_set_enabled(bool on) {
    if (!strata::vulkan::g_cvec_host.loaded || on == strata::vulkan::g_cvec_host.on) return;
    strata::vulkan::g_cvec_host.on = on;
    ++strata::vulkan::g_cvec_host.gen;          // the device switch buffer must be rewritten
}

bool cvec_enabled() { return strata::vulkan::g_cvec_host.loaded && strata::vulkan::g_cvec_host.on; }

// cvec.hpp: `void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld,
//                            const float* inj, int64_t inj_ld, bool write, void* stream);`   (layer.cpp:1330)
void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    strata::vulkan::Stream* s = strata::vulkan::stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "cvec_apply: the stream handle is not a live Vulkan stream; refusing\n");
        std::exit(2);
    }
    strata::vulkan::cvec_apply(*s, R, layer, T, r_ld, bo, bo_ld, inj, inj_ld, write);
}

}  // namespace strata::kernels
