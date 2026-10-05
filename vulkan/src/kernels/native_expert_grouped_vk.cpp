// vulkan/src/kernels/native_expert_grouped_vk.cpp - THE GROUPED NATIVE-EXPERT LAUNCHER.
//
// WHY THIS FILE EXISTS.  `native_expert_grouped` (src/kernels/cuda/iq_kernels.cu:2448) is the one kernel an IQ
// (native) pack's EXPERTS run on: it takes a list of GROUPS (distinct experts this window hit), each with a range
// of ENTRIES (which token hit it), reads that expert's weights IN PLACE from the pack, and computes
// gate/up -> SwiGLU -> q8_1 -> down for the whole window.  A Vulkan build compiles no `iq_kernels.cu`, so the
// symbol was a LOUD REFUSAL; the engine asks `native_expert_supported` at load (generate.cpp:2007) and refused
// layer 0 of `coder-iq1_m` because of it.  This TU answers it for real.
//
// THE ONE INTERFACE DECISION, and it is the reason this is not a transcription.  The reference passes
// `grp_ptr` - an array of DEVICE POINTERS (`const unsigned long long*`), one per group, each pointing at that
// expert's weights inside the pack.  A Vulkan shader cannot dereference a device pointer, so the port's grouped
// shaders take a per-group BYTE OFFSET into ONE weights buffer instead.  That indirection is the seam this file
// implements, and it is done in TWO capture-safe dispatches, never a host readback:
//
//   1. `ptr_to_off.spv` reads the pointer table (lo/hi words - no shaderInt64 feature needed) and writes, per
//      group, the byte offset from the arena base and the 4 GiB WINDOW that offset falls in.
//   2. each window is one dispatch of `native_gu_any.spv` / `native_down_any.spv`, whose weights binding is the
//      arena VIEWED AT `win_id * WIN_BYTES`.  glslang 15.1 has no 64-bit buffer index (measured 2026-10-05:
//      `w_b.b[uint64_t]` does not compile), so one binding reaches only 4 GiB - and the real pack's experts live
//      at 1.4 .. 24.8 GiB.  Splitting by window is what lets a 32-bit index reach the whole region.
//
// CAPTURE DISCIPLINE.  Every dispatch here goes through `Ctx::dispatch`, which RECORDS under capture and
// submits-and-waits otherwise, so the whole chain is recordable - the verifier runs `native_expert_grouped`
// INSIDE `cudaStreamBeginCapture` (verify.cpp:1202) and a host round-trip would invalidate it.  Nothing in this
// file reads device memory on the host.
#if !defined(STRATA_ENABLE_VULKAN)
#error "native_expert_grouped_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/iq_kernels.hpp"      // NativeExpertLayout, native_expert_grouped, native_expert_scratch_bytes
#include "strata/kernels/native_mmvq.hpp"     // native_quantize_q8_1 (the intermediate's q8_1 image)

#include "strata/vulkan/vk_backend.hpp"       // Stream, stream_of
#include "vk_arena.hpp"                       // Buf, arena_resolve, arena_alloc, view
#include "iq_grids_vk.hpp"                    // the IQ lookup grids the gu shader indexes

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// The window the weights binding advances by.  4 GiB minus 64 MiB, so a blob (the pack's largest is 2.66 MB)
// cannot straddle the 4 GiB index limit at the top of a window.  The rebase shader and this host MUST agree;
// the value is passed to the shader as a push constant so there is only one number in the file.
static constexpr uint64_t kWinBytes = (1ull << 32) - (64ull << 20);

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata_vulkan %s: %s\n", who, what);
    std::exit(2);
}

// Place (once per stream) the three IQ grids the gu shader's IQ2_S / IQ3_XXS / IQ3_S dots index.  The same lazy
// pattern `matvec_vk.cpp::grid_for` uses - `arena_alloc` never decreases, so a per-dispatch upload would grow
// the arena on every layer.
static Buf grid_buf(Stream& s, Buf& slot, const uint32_t* data, size_t bytes) {
    if (slot.buffer == VK_NULL_HANDLE) {
        slot = s.ctx->alloc(bytes);
        s.ctx->write(slot, data, bytes);
    }
    return slot;
}

}  // namespace strata::vulkan

namespace strata::kernels {

void native_grouped_set_v1(bool) { /* the port always uses the v1 shape; the A/B knob is a CUDA perf switch */ }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr,
                           const int32_t* grp_start, const int32_t* n_groups, const int32_t* ent_dst,
                           const int32_t* ent_tok, int64_t cap_groups, int64_t cap_entries,
                           const void* x_q8_1, void* scratch, float* out, void* stream, int64_t grid_groups) {
    using namespace strata::vulkan;
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) refuse("native_expert_grouped", "n_ff is not a multiple of 32");
    Stream* sp = stream_of(stream);
    if (sp == nullptr) refuse("native_expert_grouped", "the stream handle is not a live Vulkan stream");
    Stream& s = *sp;

    const uint64_t n_embd = (uint64_t) L.n_embd, n_ff = (uint64_t) L.n_ff;
    // THE SCRATCH LAYOUT (iq_kernels.cu:2455-2459): gate | up | h | hq, each 256-byte aligned.
    const size_t fa = (size_t) (((uint64_t) cap_entries * n_ff * 4 + 255) & ~(uint64_t) 255);
    const size_t hq_bytes = (size_t) (((uint64_t) cap_entries * (n_ff / 32) * 36 + 255) & ~(uint64_t) 255);

    Buf b_ptr{}, b_start{}, b_ng{}, b_dst{}, b_tok{}, b_act{}, b_scr{}, b_out{};
    if (!arena_resolve(s, grp_ptr, (uint64_t) cap_groups * 8, b_ptr) ||
        !arena_resolve(s, grp_start, (uint64_t) (cap_groups + 1) * 4, b_start) ||
        !arena_resolve(s, n_groups, 4, b_ng) ||
        !arena_resolve(s, ent_dst, (uint64_t) cap_entries * 4, b_dst) ||
        !arena_resolve(s, ent_tok, (uint64_t) cap_entries * 4, b_tok) ||
        !arena_resolve(s, x_q8_1, (uint64_t) cap_entries * (n_embd / 32) * 36, b_act) ||
        !arena_resolve(s, scratch, 3 * fa + hq_bytes, b_scr) ||
        !arena_resolve(s, out, (uint64_t) cap_entries * n_embd * 4, b_out))
        refuse("native_expert_grouped", "a pointer is not inside this stream's arena");

    // Per-call offset table: [grp_off (within-window byte offset) | grp_win (window index)].
    const uint64_t off_bytes = (uint64_t) cap_groups * 4;
    Buf b_off = s.ctx->alloc(off_bytes);
    Buf b_win = s.ctx->alloc(off_bytes);

    const uint64_t base = Stream::kArenaBase;
    struct { uint32_t base_lo, base_hi, win_bytes; } pc{
        (uint32_t) (base & 0xFFFFFFFFu), (uint32_t) (base >> 32), (uint32_t) kWinBytes};
    // THE REBASE PASS: device pointers -> (within-window offset, window).  One lane per group slot; the count is
    // read on the device so no host copy is needed.
    {
        VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/ptr_to_off.spv", 4, sizeof(pc));
        s.ctx->dispatch(pipe, {&b_ptr, &b_ng, &b_off, &b_win}, &pc, sizeof(pc), groups_for((uint64_t) cap_groups));
    }

    const uint32_t gy = (grid_groups > 0 && grid_groups <= cap_groups) ? (uint32_t) grid_groups : (uint32_t) cap_groups;

    // The gu shader declares all three IQ grids as bindings whatever the format is, so ALL are placed and bound
    // (a null descriptor is invalid here).  They are small (8 KiB + 1 KiB + 2 KiB) and live with the stream.
    Buf g1 = grid_buf(s, s.iq_grids.iq2s, strata::vkport::kIq2sGrid, sizeof(strata::vkport::kIq2sGrid));
    Buf g2 = grid_buf(s, s.iq_grids.iq3xxs, strata::vkport::kIq3xxsGrid, sizeof(strata::vkport::kIq3xxsGrid));
    Buf g3 = grid_buf(s, s.iq_grids.iq3s, strata::vkport::kIq3sGrid, sizeof(strata::vkport::kIq3sGrid));

    const uint32_t nwin = (uint32_t) ((s.arena_bytes + kWinBytes - 1) / kWinBytes);
    struct { int32_t n_embd, n_ff, ty, win_id; } pgu{(int32_t) n_embd, (int32_t) n_ff, L.gu_type, 0};
    struct { int32_t n_embd, n_ff, ty, win_id, d_row, down_off; } pdn{
        (int32_t) n_embd, (int32_t) n_ff, L.d_type, 0, (int32_t) L.d_row, (int32_t) (2 * L.up_off)};
    // Named views into the scratch (gate | up | h | hq at 0, fa, 2fa, 3fa).
    Buf v_gate = view(b_scr, 0), v_up = view(b_scr, fa), v_h = view(b_scr, 2 * fa), v_hq = view(b_scr, 3 * fa);

    for (uint32_t w = 0; w < nwin; ++w) {
        const uint64_t wbase = (uint64_t) w * kWinBytes;
        if (wbase >= s.arena_bytes) break;
        Buf wview = view(s.arena, wbase);
        // 1. gate + up, entry-major.
        pgu.win_id = (int32_t) w;
        {
            VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_gu_any.spv", 12, sizeof(pgu));
            s.ctx->dispatch(pipe, {&wview, &b_act, &g1, &g2, &g3, &b_off, &b_win, &b_start, &b_ng, &b_tok,
                                   &v_gate, &v_up},
                            &pgu, sizeof(pgu), (uint32_t) (2 * n_ff), gy);
        }
        // 2. SwiGLU over cap_entries * n_ff pairs: gate in [0, nh), up in [nh, 2nh), written to h.
        {
            struct { int32_t n; } pcs{(int32_t) ((uint64_t) cap_entries * n_ff)};
            VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/swiglu_f32.spv", 3, sizeof(pcs));
            s.ctx->dispatch(pipe, {&v_gate, &v_up, &v_h}, &pcs, sizeof(pcs),
                            groups_for((uint64_t) cap_entries * n_ff));
        }
        // 3. the intermediate's q8_1 image (the down dot's activation contract).  `native_quantize_q8_1` is the
        //    port's own q8_1 quantiser, one column per entry: column e sits at h + e*n_ff, so the down shader's
        //    `arow = e * (n_ff/32) * 36` addresses exactly this image.
        native_quantize_q8_1((const float*) ((uint8_t*) scratch + 2 * fa), (uint8_t*) scratch + 3 * fa,
                             (int) n_ff, (int) cap_entries, stream);
        // 4. down: writes out[ent_dst[e]*n_embd + r].
        pdn.win_id = (int32_t) w;
        {
            VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/native_down_any.spv", 8, sizeof(pdn));
            s.ctx->dispatch(pipe, {&wview, &v_hq, &b_off, &b_win, &b_start, &b_ng, &b_dst, &b_out},
                            &pdn, sizeof(pdn), (uint32_t) n_embd, gy);
        }
    }
}

}  // namespace strata::kernels
