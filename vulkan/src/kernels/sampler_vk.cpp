// vulkan/src/kernels/sampler_vk.cpp - `strata::kernels::sample_tokens`, the step that PRODUCES A TOKEN.
//
// ============================================================================================================
// WHY THIS ONE FIRST
// ============================================================================================================
// `generate.cpp:7742` calls `sample_tokens(d_logits, 1, n_vocab, nullptr, 0, sp, d_next, token_stream)` on the
// single-token decode path - the logits the LM head just produced in, one token id out, which the next
// iteration feeds back.  It is the last kernels-namespace symbol a plain decode reaches (the layer graph reaches
// none still undefined), so it is the batch's highest-value wire: without it there is no token at all.
//
// ============================================================================================================
// THE ENGINE'S OWN RULE FOR WHICH PATH A REQUEST TAKES (src/kernels/cuda/sampler.cu:961-1040)
// ============================================================================================================
//   * `p.greedy || p.temperature <= 0` -> the GREEDY argmax (`sampler_greedy_kernel`).  NOTE: a temperature of
//     0 is routed to the argmax HERE, at the entry point; the uniform-over-the-shortlist behaviour at temp 0 is
//     what the sampled kernel does when called DIRECTLY.
//   * otherwise the SPLIT sampler (the default): each 4096-logit partition keeps its top_k and the ordered
//     lists are merged, then the tail (top_p / min_p / temperature / one Philox draw) runs.  The engine falls
//     back to the one-block `sampler_kernel` when the merge cannot hold the width, there are more than 64 rows,
//     a stream is under capture (its split needs a `cudaMalloc`), or there is no scratch.
// The engine asserts its sampled paths "pick the same token, bit for bit" - the gate's `case_sampler_split` pins
// that parity against `sampler_kernel`, and `case_coupled_draft` the coupled tail.
//
// ============================================================================================================
// THE PORT'S RENDERING, AND THE ONE PORT DECISION STATED RATHER THAN HIDDEN
// ============================================================================================================
// The port runs the split as ONE WORKGROUP PER ROW with the partition walk and the merge in-shader
// (`sampler_split.comp`, `common/sampler_select.glsl`) - it needs NO scratch buffer and NO `cudaMalloc`.  So the
// CUDA's "a stream under capture falls back to the one-block kernel" condition does NOT apply here: the split
// records under this backend's capture (a pure dispatch), and the replay picks the same token.  Every dispatch
// below is a plain `Ctx::dispatch`, so an active capture RECORDS it (the graph batch's mapping) - there is no
// host round-trip and no submit-and-wait anywhere in this file.
//
// THE FP64 SPLIT.  `sampler_split.comp` and `sampler_kernel.comp` accumulate the top_p cut and the softmax in
// DOUBLE (the engine's arithmetic).  Intel Arc has no `shaderFloat64`; this box's ANV DOES report it, so the
// faithful path runs here.  For a device that does not, the port ships `sampler_kernel_f32.comp` (the f32
// sibling, gated by `case_sampler_kernel_f32`), and this wrapper takes it whenever the device reports no
// float64 - the same `*_enabled`-style capability answer the rest of the backend uses: the backend reports what
// it implements.
#if !defined(STRATA_ENABLE_VULKAN)
#error "sampler_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/sampler.hpp"        // the engine wrapper this file answers
#include "strata/vulkan/vk_backend.hpp"      // Stream, stream_of
#include "vk_arena.hpp"                      // Buf, arena_resolve

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static Stream& need_stream(const char* who, void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) {
        std::fprintf(stderr, "%s: the stream handle is not a live Vulkan stream; refusing\n", who);
        std::exit(2);
    }
    return *s;
}

// The push constant block `sampler_split.comp` / `sampler_kernel.comp` / `sampler_kernel_f32.comp` share (the
// gate's `case_sample_tokens`'s `Pc`, byte for byte).
struct SamplerPc {
    int32_t n_vocab, n_tokens, history_len, penalty_last_n, top_k, min_keep;
    float temperature, top_p, min_p, penalty_repeat, penalty_freq, penalty_present;
    uint32_t seed_lo, seed_hi, counter_lo, counter_hi;
};

// `sampler_greedy.comp`'s push constant (28 bytes; the gate's `case_sampler_greedy`).
struct GreedyPc {
    int32_t n_vocab, n_tokens, history_len, plen;
    float penalty_repeat, penalty_freq, penalty_present;
};

// A Vulkan descriptor cannot be null, and the greedy/sampled shaders ALWAYS bind a HIST buffer even when there
// is no penalty window.  The sentinel lives with the stream (the qsa/iq grid precedent), so a per-dispatch
// allocation cannot exhaust the never-decreasing arena.
static Buf& dummy_hist(Stream& s) {
    if (s.dummy.buffer == VK_NULL_HANDLE) {
        s.dummy = s.ctx->alloc(64);
        const int32_t zero[16] = {0};
        s.ctx->write(s.dummy, zero, sizeof(zero));
    }
    return s.dummy;
}

// The port's rendering: greedy -> the argmax shader; sampled -> the split (default), or the one-block kernel
// when the width/rows exceed the split's merge, or the f32 one-block sibling on a device without float64.
void sample_tokens(Stream& s, const float* logits, int n_tokens, int n_vocab, const int* history,
                   int history_len, const strata::kernels::SamplerParams& p, int* out) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    // The engine's own validation: a penalty window with no history is a caller error, not a token.
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    // THE OPERANDS ARE EITHER ARENA (DEVICE) BUFFERS OR REGISTERED MAPPED REGIONS - the same two kinds the CUDA
    // `sampler.cu` kernel accepts (it writes `out` through a plain device store, and the engine's `out` may be
    // either kind), and the same two `copy_from_mapped` accepts.  **THE MAPPED CASE IS NOT OPTIONAL HERE:** the
    // P6 verify window's own sampling step is `verify.cpp:1170`
    // `sample_tokens(head_logits_, T, n_vocab_, nullptr, 0, sp, m_out_, cs)` - `head_logits_` is an arena buffer
    // (`verify.cpp:437`) but `m_out_` is the MAPPED twin of `h_out_` (`:371` `mapped(T*4+16, &h_out_, &m_out_)`),
    // the token id the ENGINE reads back on the HOST.  Refusing it as "not in the arena" was the port demanding a
    // kind the CUDA never demanded; the mapped region's buffer IS device-visible, so binding it is exact.  A
    // pointer that is NEITHER is still refused, so the wrong-view guard stands.
    auto resolve_operand = [&](const void* q, uint64_t bytes, Buf& b) -> bool {
        if (q == nullptr) return false;
        if (arena_resolve(s, q, bytes, b)) return true;
        return strata::vulkan::mapped_resolve(q, bytes, b);
    };
    Buf lv{}, ov{};
    const bool lok = resolve_operand(logits, (uint64_t) n_tokens * (uint64_t) n_vocab * 4, lv);
    const bool ook = resolve_operand(out, (uint64_t) n_tokens * 4, ov);
    if (!lok || !ook) {
        std::fprintf(stderr,
                     "sample_tokens: logits %p (%s, %llu bytes) or out %p (%s, %llu bytes) is neither in this "
                     "stream's arena nor a live mapped region; refusing rather than binding a wrong view\n",
                     (const void*) logits, lok ? "ok" : "BAD",
                     (unsigned long long) ((uint64_t) n_tokens * (uint64_t) n_vocab * 4),
                     (const void*) out, ook ? "ok" : "BAD",
                     (unsigned long long) ((uint64_t) n_tokens * 4));
        std::exit(2);
    }
    const Buf* hv = nullptr;
    Buf hb{};
    if (history != nullptr && history_len > 0) {
        if (!resolve_operand(history, (uint64_t) n_tokens * (uint64_t) history_len * 4, hb)) {
            std::fprintf(stderr, "sample_tokens: the history is neither in this stream's arena nor a live mapped "
                                 "region; refusing\n");
            std::exit(2);
        }
        hv = &hb;
    } else {
        hv = &dummy_hist(s);
    }

    if (p.greedy || p.temperature <= 0.0f) {
        GreedyPc pc{n_vocab, n_tokens, history_len, p.penalty_last_n, p.penalty_repeat, p.penalty_freq,
                    p.penalty_present};
        VkPipeline pipe = s.ctx->pipeline(s.spv_dir + "/sampler_greedy.spv", 3, (uint32_t) sizeof(pc));
        s.ctx->dispatch(pipe, {&lv, hv, &ov}, &pc, sizeof(pc), (uint32_t) n_tokens);
        return;
    }

    SamplerPc pc{};
    pc.n_vocab = n_vocab;
    pc.n_tokens = n_tokens;
    pc.history_len = history_len;
    pc.penalty_last_n = p.penalty_last_n;
    pc.top_k = p.top_k;
    pc.min_keep = p.min_keep;
    pc.temperature = p.temperature;
    pc.top_p = p.top_p;
    pc.min_p = p.min_p;
    pc.penalty_repeat = p.penalty_repeat;
    pc.penalty_freq = p.penalty_freq;
    pc.penalty_present = p.penalty_present;
    pc.seed_lo = (uint32_t) (p.seed & 0xffffffffu);
    pc.seed_hi = (uint32_t) (p.seed >> 32);
    pc.counter_lo = (uint32_t) (p.counter & 0xffffffffu);
    pc.counter_hi = (uint32_t) (p.counter >> 32);

    const int n_blocks = (n_vocab + 4095) / 4096;      // kSplitBlockSpan, the merge's 4096-logit partitions
    // #3838a40 (upstream v0.1.40.2, SYCL): a device can ADVERTISE shaderFloat64 and still have no FP64 hardware -
    // Intel Arc reports `fp64 1` through ANV while Intel's own article 000089817 says Arc has no shaderFloat64, so
    // the tail is EMULATED (upstream measured 451 us against 20 us in float for one row on an A750, sampled decode
    // 7.62 -> 8.10 tok/s).  The port already ships the portable sibling; this lets the emulated double tail be
    // measured against it without a rebuild.  STRATA_VK_SAMPLER_F32=1 forces the f32 sibling; a device rule (Intel
    // vendor id + no native fp64) is the alternative once the number is in.
    //
    // MEASURED 2026-10-07 (Arc Pro B70, the CLI verify-window arms, n=2 per tail): THIS SWITCH IS A NO-OP FOR THESE
    // ARMS, because a `strata --tokens ...` run never dispatches the sampled path at all - the per-shader census of
    // every arm lists `sampler_greedy.spv` (2 dispatches) and NO `sampler_split/kernel/kernel_f32`, since the P6
    // verify window accepts a draft only if it equals the verifier's own GREEDY token.  Both tails returned the same
    // id (`3aed108cceee`, 19 rounds) and the same rate, which is what a no-op switch must return.  The
    // f32-vs-emulated-double question therefore needs the SERVER/bench sampled path (W24/W25 measured
    // `sampler_split_def` = 2.8448 ms of a ~238 ms round and a k-round scan of 0.124 ms/round, so the tail's share
    // is small at the shipped defaults).
    //
    // AND THE PREMISE IS WITHDRAWN: an all-zeros decode IS non-finite HEAD logits (`sampler_greedy.comp:106` returns
    // token 0 when nothing beats `-inf`), and a sampler only READS the logits - it cannot create them.  The nan
    // instrument's S1-S4 stages (all upstream of this call) are finite on every valid arm, so this tail is not the
    // fault's cause and was never able to be.
    static const bool prefer_f32 = [] {
        const char* e = std::getenv("STRATA_VK_SAMPLER_F32");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    const bool fp64 = s.ctx->info().shader_float64 && !prefer_f32;
    const char* which = nullptr;
    if (fp64 && n_blocks <= 64 && n_tokens <= 64) which = "/sampler_split.spv";        // the default
    else if (fp64)                                which = "/sampler_kernel.spv";       // the f64 one-block fallback
    else                                          which = "/sampler_kernel_f32.spv";  // the portable sibling
    VkPipeline pipe = s.ctx->pipeline(s.spv_dir + which, 3, (uint32_t) sizeof(pc));
    s.ctx->dispatch(pipe, {&lv, hv, &ov}, &pc, sizeof(pc), (uint32_t) n_tokens);
}

}  // namespace strata::vulkan

// ============================================================================================================
// THE ENGINE'S OWN SYMBOL
// ============================================================================================================
namespace strata::kernels {

// sampler.hpp: `void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int
// history_len, const SamplerParams& p, int* out, void* stream);`  Device pointers throughout; the engine's
// decode call is n_tokens == 1.
void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    strata::vulkan::sample_tokens(strata::vulkan::need_stream("sample_tokens", stream), logits, n_tokens, n_vocab,
                                  history, history_len, p, out);
}

}  // namespace strata::kernels
