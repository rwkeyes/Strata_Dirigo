// vulkan/src/kernels/iq_vk.cpp - the two STANDALONE I-quant entry points the engine reaches from the FORWARD
// path: `iq_dequant_f32` and `iq_embed_rows` (src/core/native_head.cpp:191/195).
//
// WHY THESE TWO AND NOT THE WHOLE iq_kernels FAMILY.  The port already answers the `iq_mmvq` composite
// (`native_mmvq`) and the q8_1 quantiser (`native_quantize_q8_1`); these two are the remaining symbols
// `include/strata/kernels/iq_kernels.hpp` declares that the decode path actually reaches - the STANDALONE
// dequantiser and the token-embedding ROW GATHER built on it.  Each drives ONE shader the port's numeric gate
// already grades (`case_iq_dequant_f32`, `case_iq_embed_rows`), so the wrapper's claim is "the engine's own
// argument list, resolved to the arena, drives the SAME .spv the case grades".
//
// BOTH SHADERS BIND ALL SIX I-QUANT GRIDS (G1..G6) even for a format that reads only one; Vulkan has no null
// descriptor, and the grids are the port's own `__constant__`-replacement storage buffers.  They are placed
// LAZILY on the stream (a per-call upload would exhaust the arena, which never shrinks), the same discipline the
// `native_mmvq` composite uses.
#if !defined(STRATA_ENABLE_VULKAN)
#error "iq_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/iq_kernels.hpp"   // iq_dequant_f32, iq_embed_rows, iq_row_bytes

#include "iq_grids_vk.hpp"                 // the six grid tables (the port's generated harness copy)
#include "strata/vulkan/vk_backend.hpp"    // Stream, stream_of
#include "vk_arena.hpp"                    // Buf, arena_resolve

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static constexpr uint32_t kLocalSize = 256;

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

static Stream& stream_for(const char* who, void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return *s;
}

// THE SIX GRIDS, placed on the stream on first use (the dequant shaders bind all six regardless of the format).
static void iq_grids(Stream& s, Buf& g1, Buf& g2, Buf& g3, Buf& g4, Buf& g5, Buf& g6) {
    if (s.iq_grids.iq1s.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq1s = s.ctx->alloc(sizeof(strata::vkport::kIq1sGrid));
        s.ctx->write(s.iq_grids.iq1s, strata::vkport::kIq1sGrid, sizeof(strata::vkport::kIq1sGrid));
    }
    if (s.iq_grids.iq2s.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq2s = s.ctx->alloc(sizeof(strata::vkport::kIq2sGrid));
        s.ctx->write(s.iq_grids.iq2s, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    }
    if (s.iq_grids.iq3s.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq3s = s.ctx->alloc(sizeof(strata::vkport::kIq3sGrid));
        s.ctx->write(s.iq_grids.iq3s, strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid));
    }
    if (s.iq_grids.iq3xxs.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq3xxs = s.ctx->alloc(sizeof(strata::vkport::kIq3xxsGrid));
        s.ctx->write(s.iq_grids.iq3xxs, strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid));
    }
    if (s.iq_grids.iq2xxs.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq2xxs = s.ctx->alloc(sizeof(strata::vkport::kIq2xxsGrid));
        s.ctx->write(s.iq_grids.iq2xxs, strata::vkport::kIq2xxsGrid, sizeof(strata::vkport::kIq2xxsGrid));
    }
    if (s.iq_grids.iq2xs.buffer == VK_NULL_HANDLE) {
        s.iq_grids.iq2xs = s.ctx->alloc(sizeof(strata::vkport::kIq2xsGrid));
        s.ctx->write(s.iq_grids.iq2xs, strata::vkport::kIq2xsGrid, sizeof(strata::vkport::kIq2xsGrid));
    }
    g1 = s.iq_grids.iq1s; g2 = s.iq_grids.iq2s; g3 = s.iq_grids.iq3xxs;
    g4 = s.iq_grids.iq3s; g5 = s.iq_grids.iq2xxs; g6 = s.iq_grids.iq2xs;
}

// ---- `iq_dequant_f32` -> iq_dequant_f32.spv ----------------------------------------------------------------
// Bindings (0..7): W(uint8 raw blocks), G1(iq1s), G2(iq2s), G3(iq3xxs), G4(iq3s), G5(iq2xxs), G6(iq2xs), OUT(f32).
// Push {int ty}.  Grid = n/256 superblocks; one 32-lane decode inside each 256-lane workgroup.
void iq_dequant_f32(Stream& s, int ggml_type, const void* src, int64_t n, float* dst) {
    if (n <= 0) return;
    if (n % 256 != 0) refuse("iq_dequant_f32", "n is not a multiple of 256 (the superblock)");
    const int64_t nb = n / 256;
    const uint64_t src_bytes = strata::kernels::iq_row_bytes(ggml_type, n);
    Buf wv{}, g1{}, g2{}, g3{}, g4{}, g5{}, g6{}, ov{};
    if (!arena_resolve(s, src, src_bytes, wv) ||
        !arena_resolve(s, dst, (uint64_t) n * 4, ov))
        refuse("iq_dequant_f32", "a pointer is not inside this stream's arena");
    iq_grids(s, g1, g2, g3, g4, g5, g6);
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/iq_dequant_f32.spv", 8, 4);
    struct { int32_t ty; } pc{ggml_type};
    s.ctx->dispatch(pipe, {&wv, &g1, &g2, &g3, &g4, &g5, &g6, &ov}, &pc, sizeof(pc), (uint32_t) nb);
}

// ---- `iq_embed_rows` -> iq_embed_rows.spv ------------------------------------------------------------------
// Bindings (0..8): W(table bytes), G1..G6, TOK(int32), OUT(f32).  Push {int ty; int n_embd; uint row_bytes}.
// Grid = (n_embd/256, n_tok): block (b,t) dequantises superblock b of row tokens[t] into out[t*n_embd + b*256].
// The table row starts at `tokens[t] * row_bytes` - a caller STRIDE, the whole reason this is its own kernel.
void iq_embed_rows(Stream& s, int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens,
                   int64_t n_tok, int64_t n_embd, float* out) {
    if (n_tok <= 0 || n_embd <= 0) return;
    if (n_embd % 256 != 0) refuse("iq_embed_rows", "n_embd is not a multiple of 256 (the superblock)");
    if (row_bytes == 0) refuse("iq_embed_rows", "row_bytes is zero");
    Buf wv{}, g1{}, g2{}, g3{}, g4{}, g5{}, g6{}, tv{}, ov{};
    // The table is one arena region; the first row sizes the RANGE CHECK.  Every row the shader reads is inside
    // the SAME buffer, so identifying the buffer + offset from the first row is what the shader needs (it does
    // its own 64-bit `token * row_bytes` arithmetic inside that buffer).
    if (!arena_resolve(s, table, row_bytes, wv) ||
        !arena_resolve(s, tokens, (uint64_t) n_tok * 4, tv) ||
        !arena_resolve(s, out, (uint64_t) n_tok * (uint64_t) n_embd * 4, ov))
        refuse("iq_embed_rows", "a pointer is not inside this stream's arena");
    iq_grids(s, g1, g2, g3, g4, g5, g6);
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/iq_embed_rows.spv", 9, 12);
    struct { int32_t ty; int32_t n_embd; uint32_t row_bytes; } pc{ggml_type, (int32_t) n_embd, (uint32_t) row_bytes};
    s.ctx->dispatch(pipe, {&wv, &g1, &g2, &g3, &g4, &g5, &g6, &tv, &ov}, &pc, sizeof(pc),
                    (uint32_t) (n_embd / 256), (uint32_t) n_tok);
}

}  // namespace strata::vulkan

// ============================================================================================================
// THE ENGINE'S OWN SYMBOLS (the thin wrappers the engine headers declare)
// ============================================================================================================
namespace strata::kernels {

void iq_dequant_f32(int ggml_type, const void* src, int64_t n, float* dst, void* stream) {
    if (n <= 0) return;
    strata::vulkan::iq_dequant_f32(strata::vulkan::stream_for("iq_dequant_f32", stream), ggml_type, src, n, dst);
}
void iq_embed_rows(int ggml_type, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok,
                   int64_t n_embd, float* out, void* stream) {
    if (n_tok <= 0 || n_embd <= 0) return;
    strata::vulkan::iq_embed_rows(strata::vulkan::stream_for("iq_embed_rows", stream), ggml_type, table, row_bytes,
                                  tokens, n_tok, n_embd, out);
}

// ---- THE IQ CAPABILITY PREDICATES (model load, not decode) ---------------------------------------------------
// `embed_type_supported` (`native_head.cpp:119`) validates the token-embedding type before NativeEmbed loads it;
// `iq_supported` (`iq_parity.cpp`, excluded from this build) is its dequant-only sibling.  The engine's rule is
// `is_iq(t) || t == 30` (iq_kernels.cu:1777-1778).  This backend must answer its OWN coverage, and the answer
// is MEASURED off the shader the two entry points above drive: `shaders/common/iq_dequant.glsl`'s `dq_dispatch`
// switches on exactly this set of types (30 BF16, 20 IQ4_NL, 23 IQ4_XS, 8 Q8_0, 6 Q5_0, 7 Q5_1, 42 Q2_0,
// 12 Q4_K, 13 Q5_K, 11 Q3_K, 18 IQ3_XXS, 21 IQ3_S, 22 IQ2_S, 29 IQ1_M, 16 IQ2_XXS, 17 IQ2_XS) - the SAME set
// the engine's `is_iq()` names, so the port answers TRUE for exactly the types its dequantizer implements.
// NOTE: this is the engine's type TABLE, verified case-by-case against the port's shader above; the port's
// `sample_tokens`/decode path is unaffected (the predicates are load-time only).
namespace {
bool iq_type_covered(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 ||
           t == 11 || t == 12 || t == 13 || t == 7 || t == 6 || t == 8;
}
}  // namespace
bool iq_supported(int ggml_type) noexcept { return iq_type_covered(ggml_type); }
bool embed_type_supported(int ggml_type) noexcept { return iq_type_covered(ggml_type) || ggml_type == 30; }

}  // namespace strata::kernels
