// vulkan/src/kernels/shared_expert_vk.cpp - the shared expert (`shared_expert`) and its two host getters.
//
// WHY THIS FILE.  `layer.cpp`'s `moe_shared` (`:417`) calls `shared_expert` on EVERY MoE block, so it is on the
// forward path of every token.  The previous batch reported it UN-WIRABLE because its CANONICAL path dispatches
// `s_gemv_q8_0_split` / `s_gemv_q8k_split`, "and neither has a shader in this tree".  THIS BATCH PORTS that pair
// (matvec_vk.cpp) - and, reading the engine's own code for the wiring, found the shader `s_gemv_q8_split` was
// ALREADY IN THIS TREE (built by run_gate.sh, gated by `case_s_gemv_q8_split`); what was missing was the
// ENGINE-SIDE definition, exactly the `indexer_key_append` shape the triage records ("ported" is a claim about
// the SYMBOL the layer links against, not about a shader or a plan row).
//
// THE CANONICAL PATH, transcribed from `src/kernels/cuda/shared_expert.cu:242-361`:
//
//     gate = gemv(gate_form, x)          # s2_gemv_q8 if the form is S2, else s_gemv_q8{_0,k}_split
//     up   = gemv(up_form,   x)          #   by the form's own `act_kind` (Q8_0 image vs Q8_K image)
//     gate = silu(gate) * up             # SILU GOES ON THE GATE, and the multiply is the reference's order
//     h    = quantize_to(down's contract)(gate); out = gemv(down_form, h)   # down: n_ff -> n_embd
//     g    = sigmoid(dot(x_bf16, ffn_gate_inp_shexp))   # ONE SCALAR per token
//     out *= g
//
// The result is ADDED to the routed output by `moe_combine` - not router-weighted here.
//
// THE TWO BRANCHES THE BACKEND DOES NOT TAKE, and why that is stated rather than silently handled:
//   * `native_bf16` (the native scalar gate, `shared_expert.cu:346-348`) reads `x_f32` and uses the pinned
//     CUDA fast-math MMVF+sigmoid.  The backend OWNS the getter `shared_expert_native_bf16_enabled()` and
//     answers FALSE (below), so this branch is not reachable; were it somehow entered the wrapper REFUSES.
//   * the native PROJECTIONS (`native->gate_data`/`up_data`/`down_data`, selected per weight by
//     `native_mmvq_supported(type)`) ARE implemented, by delegating to this port's already-wired
//     `native_quantize_q8_1` + `native_mmvq` - the same composite the CUDA calls - because the shipped
//     `--native` launch CAN make them non-null (ffn_down_shexp is IQ4_NL on some layers, one of the six types
//     this port supports).  They are not a dodge: the port's native member is real.
//
// ARITHMETIC NOTE, stated: `shared_expert.cu` runs its swiglu in DOUBLE (the default `swiglu_kernel`) or in
// CUDA fast-math (the native one).  The port dispatches `swiglu_f32.comp`, the correctly-rounded FLOAT
// expression - neither, and documented as such in that shader's own header.  The scalar gate is `scalar_gate_f32`
// (bf16 products are exact in f32, so only the summation matters and that shader compensates with Kahan).
#if !defined(STRATA_ENABLE_VULKAN)
#error "shared_expert_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/shared_expert.hpp"   // shared_expert, NativeSharedWeights, the two host getters
#include "strata/kernels/s_gemv.hpp"          // s_gemv_q8k_split / s_gemv_q8_0_split (the newly-ported pair)
#include "strata/kernels/s2_gemv_q8.hpp"      // s2_gemv_q8 (the S2 gate/up)
#include "strata/kernels/quantize_act.hpp"    // quantize_q8_0 / quantize_q8_K (the down's activation image)
#include "strata/kernels/native_mmvq.hpp"     // native_mmvq / native_quantize_q8_1 / *_supported

#include "strata/vulkan/vk_backend.hpp"       // Stream, stream_of
#include "vk_arena.hpp"                       // arena_resolve, Buf

#include <cstdio>
#include <cstdlib>

namespace strata::vulkan {

static constexpr uint32_t kLocalSize = 256;
static uint32_t groups_for(uint64_t n) { return (uint32_t) ((n + kLocalSize - 1) / kLocalSize); }

[[noreturn]] static void refuse(const char* what) {
    std::fprintf(stderr, "strata::vulkan::shared_expert: %s - refusing rather than dispatching a wrong view\n", what);
    std::exit(2);
}

static Stream& need_stream(void* stream) {
    Stream* s = stream_of(stream);
    if (s == nullptr) refuse("the engine stream is not a live Vulkan Stream");
    return *s;
}

// A Vulkan descriptor cannot be null; a buffer bound but never read reuses an already-resolved view.

// `swiglu_f32` -> swiglu_f32.spv (G read, U read, O write; push {int n}): out[i] = (g/(1+exp(-g)))*u[i].
static void swiglu_f32(Stream& s, const float* gate, const float* up, float* out, int64_t n) {
    Buf gv{}, uv{}, ov{};
    if (!arena_resolve(s, gate, (uint64_t) n * 4, gv) || !arena_resolve(s, up, (uint64_t) n * 4, uv) ||
        !arena_resolve(s, out, (uint64_t) n * 4, ov))
        refuse("the swiglu operand is not inside this stream's arena");
    struct { int32_t n; } pc{(int32_t) n};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/swiglu_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&gv, &uv, &ov}, &pc, sizeof(pc), groups_for((uint64_t) n));
}

// `scalar_gate_f32` -> scalar_gate_f32.spv (X bf16 pairs, W bf16 pairs, OUT; push {int n_tokens; int n_embd}):
// one workgroup per token, `g[t] = sigmoid(dot(x_bf16[t], w_bf16))`.  ONE TOKEN here (the decode path).
static void scalar_gate_f32(Stream& s, const uint16_t* x_bf16, const uint16_t* w_bf16, float* g, int64_t n_embd) {
    Buf xv{}, wv{}, ov{};
    if (!arena_resolve(s, x_bf16, (uint64_t) n_embd * 2, xv) || !arena_resolve(s, w_bf16, (uint64_t) n_embd * 2, wv) ||
        !arena_resolve(s, g, 4, ov))
        refuse("the scalar-gate operand is not inside this stream's arena");
    struct { int32_t n_tokens; int32_t n_embd; } pc{1, (int32_t) n_embd};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/scalar_gate_f32.spv", 3, sizeof(pc));
    s.ctx->dispatch(p, {&xv, &wv, &ov}, &pc, sizeof(pc), 1);
}

// `scale_rows` -> scale_rows.comp (O rw, G read; push {int n_tokens; int n}): out[t][i] *= g[t].  One row here.
static void scale_rows(Stream& s, float* out, const float* g, int64_t n_embd) {
    Buf ov{}, gv{};
    if (!arena_resolve(s, out, (uint64_t) n_embd * 4, ov) || !arena_resolve(s, g, 4, gv))
        refuse("the per-row scale operand is not inside this stream's arena");
    struct { int32_t n_tokens; int32_t n; } pc{1, (int32_t) n_embd};
    VkPipeline p = s.ctx->pipeline(s.spv_dir + "/scale_rows.spv", 2, sizeof(pc));
    s.ctx->dispatch(p, {&ov, &gv}, &pc, sizeof(pc), 1);
}

}  // namespace strata::vulkan

namespace strata::kernels {

// THE BACKEND'S OWN ANSWER for the native-BF16 scalar gate (a `host` row).  This backend implements the
// DEFAULT (BF16-input, portable) scalar gate (`scalar_gate_f32`), so it reports the native one NOT selected -
// the `native_gdn_enabled()` pattern: the backend reports what it implements.  `set` is a no-op for the same
// reason the `native_*_set_enabled` setters are (the answer is the backend's, not the caller's).
void shared_expert_set_native_bf16(bool) { /* the backend answers for itself; see the header note */ }
bool shared_expert_native_bf16_enabled() { return false; }

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    (void) tpr;   // the port's split GEMVs are one workgroup per row: the CUDA knob is not connected here

    const bool use_native = shared_expert_native_bf16_enabled();
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;

    if ((use_native || native_gate || native_up) && x_f32 == nullptr)
        strata::vulkan::refuse("the native input projection requires the unrounded x_f32 (shared_expert.cu throws)");
    if (native_projection && (native->q8_1 == nullptr || stream == nullptr))
        strata::vulkan::refuse("the native projection requires the caller's native->q8_1 scratch and a stream");
    if (scratch == nullptr)
        strata::vulkan::refuse("scratch is null; the caller owns it (see shared_expert_scratch_bytes)");

    strata::vulkan::Stream& s = strata::vulkan::need_stream(stream);

    // CARVED FROM THE CALLER'S SCRATCH, in the SAME layout `shared_expert_scratch_bytes` sizes (ple_vk.cpp):
    // gate(n_ff f32) | up(n_ff f32) | q8_0(n_ff/32*34) | q8k(n_ff/256*292) | g(1 f32), 16-byte aligned.  The
    // CUDA used four `cudaMalloc`s per call (illegal in a capture and a token-path allocation); the port
    // carves, like the CUDA does now.
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~(uint64_t) 15;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~(uint64_t) 15;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~(uint64_t) 15;
    uint8_t* p = reinterpret_cast<uint8_t*>(scratch);
    float* gate = reinterpret_cast<float*>(p);
    float* up = reinterpret_cast<float*>(p + a);
    uint8_t* h_q8_0 = p + a * 2;
    uint8_t* h_q8k = p + a * 2 + q0;
    float* g = reinterpret_cast<float*>(p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, read from the weight's own form - `SForm::act_kind` is CARRIED
    // (Q8_0 vs Q8_K cannot be told apart from the other fields); `code_bits == 2` is the S2 path (its own
    // activation is Q8_0 and `s2_gemv_q8` hardcodes the S2 attributes).
    //
    // **THE ACTIVATION IS A PARAMETER, NOT A CAPTURE - AND THAT IS A DEFECT THIS CASE CAUGHT.**  The CUDA's
    // `gemv` lambda takes `act80`/`actq8k` as arguments: gate and up read the INPUT images (`x_q8_0`/`x_q8k`),
    // but the DOWN projection reads the QUANTISED SWIGLU INTERMEDIATE (`h_q8_0`/`h_q8k`).  A first version of
    // this lambda closed over `x_q8_0`/`x_q8k`, so the down GEMV read the input activation instead of the buffer
    // the two lines above it had just produced - a plausible, finite, WRONG answer (the case measured it at
    // ~1e3x the oracle).  The lambda now takes the activation pointers explicitly, exactly as the CUDA does.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    // gate and up, then silu(gate) * up written back into `gate`.
    if (native_gate || native_up) native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    strata::vulkan::swiglu_f32(s, gate, up, gate, n_ff);

    // down: quantize the intermediate to the DOWN weight's OWN contract, then the projection (n_ff -> n_embd).
    // THE ACTIVATION IS `h_q8_0`/`h_q8k` - the buffer this function just produced - NOT the input image.
    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else if (down_form.act_kind == 1) {
        if (n_ff % 256 != 0)
            strata::vulkan::refuse("the down weight wants Q8_K but n_ff is not a multiple of 256 (structurally impossible)");
        quantize_q8_K(gate, h_q8k, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    // the per-token scalar gate, computed from `x` (the ORIGINAL hidden state), then the multiply.
    if (use_native)
        strata::vulkan::refuse("native_bf16 scalar gate is not selected (shared_expert_native_bf16_enabled() is false)");
    strata::vulkan::scalar_gate_f32(s, x_bf16, gate_inp_bf16, g, n_embd);
    strata::vulkan::scale_rows(s, out, g, n_embd);
}

}  // namespace strata::kernels
