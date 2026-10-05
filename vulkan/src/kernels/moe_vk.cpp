// vulkan/src/kernels/moe_vk.cpp - the RESIDENT-EXPERT (S2 tier) entry points the engine's capture path reaches:
// `moe_hit_select`, `moe_hit_grouped_s2`, `moe_hit_grouped_s2_dev`, `moe_hit_add`, and the
// `moe_hit_grouped_scratch_bytes` size the caller carves from.
//
// WHY THESE FOUR.  `src/core/session.cpp:866-877` drives them INSIDE the captured block - select the resident
// hits, run the grouped expert chain on the device, then fold the hits back into `parts` - so each must RECORD
// under capture, not submit-and-wait.  They are called with the ENGINE's raw device pointers, which this file
// resolves to arena views and dispatches.  The per-hit chain is a COMPOSITION of four shaders the port already
// grades (`case_moe_hit_grouped_s2`): gate+up (`s2expert_gu`), SwiGLU (`s2expert_swiglu`), the intermediate's
// own quantiser (`quantize_q8_0` / `quantize_q8_0_scaled`), and down (`s2expert_down`).  The two single-kernel
// symbols (`moe_hit_select`, `moe_hit_add`) are one dispatch each.
//
// THE GEOMETRY IS BAKED, as it is in the source: `src/kernels/cuda/s2_expert_grouped.cu:37-38` fixes H = 2560 and
// FF = 640, and the entry-point signatures carry NEITHER.  So `n_embd`/`n_ff` (push-constant fields the shaders
// take) are the same two constants here.
//
// CAPTURE DISCIPLINE (the graph batch's finding): `Ctx::dispatch` RECORDS while a capture is active, so every
// chain below is recordable UNLESS it needs a host round-trip.  `moe_hit_grouped_s2_dev` is the one that does
// (it must read the device's live hit COUNT to size its launches against the capacity grid), so it REFUSES
// LOUDLY under capture - it invalidates the recording so `cudaStreamEndCapture` returns
// `cudaErrorStreamCaptureUnsupported` - rather than replay a stale count.  The others record.
#if !defined(STRATA_ENABLE_VULKAN)
#error "moe_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/s2_expert_grouped.hpp"   // the five symbols this file answers
#include "strata/kernels/quantize_act.hpp"        // quantize_q8_0 / quantize_q8_0_scaled (the intermediate)

#include "strata/vulkan/vk_backend.hpp"           // Stream, stream_of
#include "vk_arena.hpp"                           // Buf, arena_resolve, stream_read

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

// THE BAKED GEOMETRY (s2_expert_grouped.cu:37-38).
static constexpr int64_t kH = 2560;
static constexpr int64_t kFF = 640;

[[noreturn]] static void refuse(const char* who, const char* what) {
    std::fprintf(stderr, "strata::vulkan::%s: %s - refusing rather than dispatching a wrong view\n", who, what);
    std::exit(2);
}

static Stream& need_stream(const char* who, void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return *s;
}

// `swiglu_f32` -> swiglu_f32.spv (G read, U read, O write; push {int n}): out[i] = (g/(1+exp(-g))) * u[i].
static void swiglu_f32(Stream& s, const float* gate, const float* up, float* out, int64_t n) {
    Buf gv{}, uv{}, ov{};
    if (!arena_resolve(s, gate, (uint64_t) n * 4, gv) || !arena_resolve(s, up, (uint64_t) n * 4, uv) ||
        !arena_resolve(s, out, (uint64_t) n * 4, ov))
        refuse("swiglu_f32", "an operand is not inside this stream's arena");
    struct { int32_t n; } pc{(int32_t) n};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/swiglu_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&gv, &uv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// ---- `moe_hit_select` -> moe_hit_select.spv (IDS, RES, SLOT, DST, CNT; push {int k; int n_expert}; grid 1) ---
// Device-side hit list: only a RESIDENT expert (res_row[e] >= 0) is a hit, compacted in ascending lane order, and
// `dst[at]` is the ROUTING POSITION `lane`, not the slot.  A pure dispatch with a one-lane body - recordable.
void moe_hit_select(Stream& s, const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot,
                    int32_t* dst, int32_t* count) {
    if (k < 1 || k > 32) refuse("moe_hit_select", "k must be 1..32 (the CUDA's own bound)");
    Buf iv{}, rv{}, sv{}, dv{}, cv{};
    if (!arena_resolve(s, ids, (uint64_t) k * 4, iv) ||
        !arena_resolve(s, res_row, (uint64_t) (n_expert > 0 ? n_expert : 1) * 4, rv) ||
        !arena_resolve(s, slot, (uint64_t) k * 4, sv) || !arena_resolve(s, dst, (uint64_t) k * 4, dv) ||
        !arena_resolve(s, count, 4, cv))
        refuse("moe_hit_select", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/moe_hit_select.spv", 5, 8);
    struct { int32_t k; int32_t n_expert; } pc{k, n_expert};
    s.ctx->dispatch(pipe, {&iv, &rv, &sv, &dv, &cv}, &pc, sizeof(pc), 1);
}

// ---- `moe_hit_add` -> moe_hit_add.spv (PARTS rw, HIT ro, DST ro, CNT ro; push {int n_embd}; grid (1, cap)) ----
// `parts[dst[h]] += hit_out[dst[h]]` for every LIVE hit (the count is on the device; rows past it return
// unwritten - the capacity grid, which is what makes a RECORDED launch replay-safe at a later token).  Pure
// dispatch.  The `out` rows are (cap, n_embd): the destination `dst[h] < cap`, so the region is cap rows.
void moe_hit_add(Stream& s, float* parts, const float* hit_out, const int32_t* dst, const int32_t* count,
                 int64_t cap, int64_t n_embd) {
    if (cap <= 0 || n_embd <= 0) return;
    Buf pv{}, hv{}, dv{}, cv{};
    if (!arena_resolve(s, parts, (uint64_t) cap * (uint64_t) n_embd * 4, pv) ||
        !arena_resolve(s, hit_out, (uint64_t) cap * (uint64_t) n_embd * 4, hv) ||
        !arena_resolve(s, dst, (uint64_t) cap * 4, dv) || !arena_resolve(s, count, 4, cv))
        refuse("moe_hit_add", "a pointer is not inside this stream's arena");
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/moe_hit_add.spv", 4, 4);
    struct { int32_t n_embd; } pc{(int32_t) n_embd};
    s.ctx->dispatch(pipe, {&pv, &hv, &dv, &cv}, &pc, sizeof(pc), 1u, (uint32_t) cap);
}

// ---- THE PER-HIT CHAIN: gate+up -> SwiGLU -> quantise -> down (s2_expert_grouped.cu:579's four launches) ------
// `n_hits` is a HOST count here (the `_dev` variant resolves it from the device first).  Scratch layout, carved
// exactly as the CUDA does (s2_expert_grouped.cu:592-596): gate_up | h_q8_0 | h_scales, 16-byte aligned.
// `x_scales` (R4.2h) is the fp32 activation scale array from `quantize_q8_0_scaled`; null keeps the fp16 `d` in
// the block.  The two are NEVER mixed: the caller passes it or not and the whole chain follows.
static void moe_hit_chain(Stream& s, const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                          int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, const float* x_scales,
                          void* scratch, float* out) {
    if (n_hits <= 0) return;
    const int64_t H = kH, FF = kFF;
    if (scratch == nullptr) refuse("moe_hit_grouped_s2", "scratch is null; the caller owns it (see the size fn)");
    if (blob_bytes <= 0) refuse("moe_hit_grouped_s2", "blob_bytes must be positive");
    const uint64_t gu_bytes = ((uint64_t) n_hits * (uint64_t) (2 * FF) * 4 + 15) & ~(uint64_t) 15;
    const uint64_t q8_bytes = ((uint64_t) n_hits * (uint64_t) (FF / 32) * 34 + 15) & ~(uint64_t) 15;
    uint8_t* p = reinterpret_cast<uint8_t*>(scratch);
    float* gate_up = reinterpret_cast<float*>(p);
    uint8_t* h_q8_0 = p + gu_bytes;
    float* h_scales = reinterpret_cast<float*>(p + gu_bytes + q8_bytes);

    const bool scaled = (x_scales != nullptr);
    const int32_t use_xs = scaled ? 1 : 0;

    // Resolve every region the two projections bind.  The blob is the SLOT ARENA base (slot s starts at
    // base + s*blob_bytes); one blob's worth of range identifies the buffer, and a slot beyond the first is the
    // SAME buffer.  `out` is (cap, n_embd) with destinations below cap, so one row identifies it.
    Buf bv{}, av{}, xsv{}, sv{}, dv{}, guv{}, hqv{}, hsv{}, ov{};
    if (!arena_resolve(s, blob_base, (uint64_t) blob_bytes, bv) ||
        !arena_resolve(s, x_q8_0, (uint64_t) (H / 32) * 34, av) ||
        !arena_resolve(s, slot_index, (uint64_t) n_hits * 4, sv) ||
        !arena_resolve(s, dst_index, (uint64_t) n_hits * 4, dv) ||
        !arena_resolve(s, scratch, gu_bytes + q8_bytes + (uint64_t) n_hits * (uint64_t) (FF / 32) * 4, guv) ||
        !arena_resolve(s, h_q8_0, q8_bytes, hqv) ||
        !arena_resolve(s, out, (uint64_t) H * 4, ov))
        refuse("moe_hit_grouped_s2", "a pointer is not inside this stream's arena");
    if (scaled && !arena_resolve(s, x_scales, (uint64_t) (H / 32) * 4, xsv))
        refuse("moe_hit_grouped_s2", "the activation scales are not inside this stream's arena");
    // XS is always bound (Vulkan has no null descriptor): a scaled call binds the real scales; an unscaled one
    // reuses the slot list (read-only and never dereferenced with use_xscales == 0 - the port's own idiom).
    const Buf* gu_xs = scaled ? &xsv : &sv;
    const Buf* dn_xs = scaled ? &hsv : &sv;
    if (scaled) {
        if (!arena_resolve(s, h_scales, (uint64_t) n_hits * (uint64_t) (FF / 32) * 4, hsv))
            refuse("moe_hit_grouped_s2", "the intermediate scales are not resolvable");
    }

    // 1. gate + up: one workgroup per row-slot (n_hits * 2 * FF of them), gate-major output.
    {
        struct { int32_t n_hits; int32_t n_embd; int32_t n_ff; int32_t blob_bytes; int32_t use_xscales; } pc{
            (int32_t) n_hits, (int32_t) H, (int32_t) FF, (int32_t) blob_bytes, use_xs};
        VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/s2expert_gu.spv", 5, sizeof(pc));
        s.ctx->dispatch(pipe, {&bv, &av, gu_xs, &sv, &guv}, &pc, sizeof(pc), (uint32_t) (n_hits * 2 * FF));
    }
    // 2. SwiGLU over the n_hits * FF pairs (gate in the first half, up in the second), in place.
    swiglu_f32(s, gate_up, gate_up + (size_t) n_hits * (size_t) FF, gate_up, n_hits * FF);
    // 3. the intermediate's own contract: the down weight is Q2_0 (vec_dot_type Q8_0).  R4.2h: with the CPU's
    //    fp32 scales, quantise SCALED so a hit reproduces a miss; otherwise the plain fp16-d quantiser.
    if (scaled) strata::kernels::quantize_q8_0_scaled(gate_up, h_q8_0, h_scales, n_hits * FF, (void*) &s);
    else        strata::kernels::quantize_q8_0(gate_up, h_q8_0, n_hits * FF, (void*) &s);
    // 4. down: one workgroup per (hit, output row), writing hit h's row at dst_index[h].
    {
        struct { int32_t n_hits; int32_t n_embd; int32_t n_ff; int32_t blob_bytes; int32_t use_xscales; } pc{
            (int32_t) n_hits, (int32_t) H, (int32_t) FF, (int32_t) blob_bytes, use_xs};
        VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/s2expert_down.spv", 6, sizeof(pc));
        s.ctx->dispatch(pipe, {&bv, &hqv, dn_xs, &sv, &dv, &ov}, &pc, sizeof(pc), (uint32_t) (n_hits * H));
    }
}

}  // namespace strata::vulkan

// ============================================================================================================
// THE ENGINE'S OWN SYMBOLS
// ============================================================================================================
namespace strata::kernels {

// `moe_hit_grouped_scratch_bytes` - the caller's carve.  Transcribed from s2_expert_grouped.cu:566-572 so the
// wrapper's layout never exceeds what the caller sized: gate_up | block_q8_0 | fp32 intermediate scales | the
// activation scales.  (A `host` row in PORT-MAP.tsv: a size, not a dispatch, but the symbol the link needs.)
uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    const auto a16 = [](uint64_t x) { return (x + 15) & ~(uint64_t) 15; };
    return a16(gu) + a16(q8) + 2 * a16(hs) + a16(xh);
}

void moe_hit_select(const int32_t* ids, const int32_t* res_row, int k, int n_expert, int32_t* slot, int32_t* dst,
                    int32_t* count, void* stream) {
    strata::vulkan::moe_hit_select(strata::vulkan::need_stream("moe_hit_select", stream), ids, res_row, k, n_expert,
                                   slot, dst, count);
}

void moe_hit_add(float* parts, const float* hit_out, const int32_t* dst, const int32_t* count, int64_t cap,
                 int64_t n_embd, void* stream) {
    if (cap <= 0 || n_embd <= 0) return;
    strata::vulkan::moe_hit_add(strata::vulkan::need_stream("moe_hit_add", stream), parts, hit_out, dst, count, cap,
                                n_embd);
}

// THE PER-HIT ENTRY: `n_hits` is a HOST count (the interface carries it), so the chain RECORDS under capture.
void moe_hit_grouped_s2(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                        int64_t n_hits, int64_t blob_bytes, const uint8_t* x_q8_0, void* scratch, float* out,
                        void* stream, const float* x_scales) {
    if (n_hits <= 0) return;
    strata::vulkan::moe_hit_chain(strata::vulkan::need_stream("moe_hit_grouped_s2", stream), blob_base, slot_index,
                                  dst_index, n_hits, blob_bytes, x_q8_0, x_scales, scratch, out);
}

// THE CAPACITY/DEVICE-COUNT ENTRY.  The port's `s2expert_gu`/`s2expert_down` shaders take the HIT COUNT as a push
// constant, NOT a device pointer, so `_dev`'s "sized for `cap`, reads the real count from `d_count`" contract has
// to be realised by RESOLVING the count on the host: a device->host read, then the same chain with `n_hits = count`.
// That is a HOST ROUND-TRIP INSIDE THE BLOCK, so under stream capture it is REFUSED LOUDLY - the recording is
// invalidated (`Ctx::capture_invalidate`), which makes `cudaStreamEndCapture` return
// `cudaErrorStreamCaptureUnsupported` - rather than replay a count frozen at capture time.  (This is exactly the
// Deliverable-B discipline.  The shipped capture default does not reach it: `--expert-cache 0` leaves `hits`
// null at session.cpp:865, so this symbol is only called with the resident-expert tier switched on.)
void moe_hit_grouped_s2_dev(const uint8_t* blob_base, const int32_t* slot_index, const int32_t* dst_index,
                            const int32_t* d_count, int64_t cap, int64_t blob_bytes, const uint8_t* x_q8_0,
                            void* scratch, float* out, void* stream, const float* x_scales) {
    if (cap <= 0) return;
    strata::vulkan::Stream& s = strata::vulkan::need_stream("moe_hit_grouped_s2_dev", stream);
    if (s.ctx != nullptr && s.ctx->capturing()) {
        // The count must be read on the host to size the launches; that cannot be recorded.  Invalidate so the
        // capture fails LOUDLY at EndCapture instead of replaying a stale count.
        s.ctx->capture_invalidate();
        std::fprintf(stderr,
                     "strata::vulkan::moe_hit_grouped_s2_dev: this entry point reads the live hit count on the "
                     "host and cannot be recorded; refusing under stream capture (cudaErrorStreamCaptureUnsupported)\n");
        return;
    }
    strata::vulkan::Buf cv{};
    if (!arena_resolve(s, d_count, 4, cv))
        strata::vulkan::refuse("moe_hit_grouped_s2_dev", "d_count is not in the arena");
    int32_t count = 0;
    strata::vulkan::stream_read(s, d_count, &count, 4);
    if (count <= 0) return;                       // nothing resident: nothing to run (the chain's own guard)
    if (count > cap) strata::vulkan::refuse("moe_hit_grouped_s2_dev", "the device count exceeds the capacity");
    strata::vulkan::moe_hit_chain(s, blob_base, slot_index, dst_index, count, blob_bytes, x_q8_0, x_scales, scratch,
                                  out);
}

}  // namespace strata::kernels
