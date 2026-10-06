// vulkan/src/kernels/native_expert_vk.cpp - the NATIVE (IQ) EXPERT GEOMETRY a Vulkan build must supply.
//
// WHY THIS FILE EXISTS.  `src/kernels/cpu/expert_layout.cpp` parses `<pack>/native_experts.txt` and, for every
// layer, calls `native_fmt(gu_type, d_type, n_embd, n_ff)` and REQUIRES `f.bytes == the pack's blob column`
// (expert_layout.cpp:284-289).  On a CUDA/HIP build `native_fmt` is ggml-cpu's: it asks
// `ggml_get_type_traits_cpu` for the type's row size and its `vec_dot_type`.  The Vulkan configuration builds
// NONE of the ggml half (`src/kernels/cpu/native_expert.cpp` + `iq_avx*.cpp` are excluded - no ggml sources here),
// so that definition is absent.  Without it `expert_layout_load` fails with "built without STRATA_NATIVE_EXPERTS"
// and every native (IQ) pack - including `coder-iq1_m` - is refused at load.
//
// THE FIX IS THE GEOMETRY, NOT A SUBSTITUTE.  `native_fmt` is arithmetic on block sizes the engine ALREADY
// states, byte for byte, in the port's own `iq_row_bytes` (`vulkan/src/kernels/matvec_vk.cpp`, transcribed from
// ggml) - so this file does NOT carry a second copy of the table and cannot drift from the `*_mmvq` shaders'
// row stride.  The `vec_dot_type` map is transcribed from the engine's pinned ggml
// (`ggml/src/ggml-cpu/ggml-cpu.c`'s type-traits table): the i-quants' `vec_dot_type` is Q8_K, IQ4_NL's and
// Q2_0's is Q8_0, BF16's is BF16.  The load path's own blob-size check is the measurement: if any of this is
// wrong for any of the 48 layers, `expert_layout_load` refuses by name instead of running.
//
// WHAT THIS FILE DOES **NOT** DO, AND SAYS SO.  The four row/activation kernels (`native_quant_act`,
// `native_quant_h`, `native_gu_rows`, `native_down_rows`) are the CPU expert COMPUTE - ggml-cpu's vec_dot rows,
// the engine's CPU expert pool's work.  This port has no CPU-hybrid execution path (the token must come from the
// GPU backend), and the CUDA configuration's own single-token expert adapter REFUSES a native pack
// (`expert_source.cpp:1868`).  So each of the four is a LOUD REFUSAL naming the GPU kernel that would replace it,
// never a silent zero.
#if !defined(STRATA_ENABLE_VULKAN)
#error "native_expert_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/cpu/native_expert.hpp"   // NativeFmt, native_fmt and the four row kernels
#include "strata/kernels/iq_kernels.hpp"         // iq_row_bytes (the port's ONE row-size table)

#include <cstdio>
#include <cstdlib>
#include <string>

namespace strata::kernels::cpu {

// ggml's `vec_dot_type` per weight type (ggml-cpu.c's type-traits table): the activation format a row dot reads.
// -1: a type this engine has no row layout for.
static int vec_dot_type_of(int t) noexcept {
    switch (t) {
    case 42:                       // Q2_0
    case 20:                       // IQ4_NL
    case 2:                        // Q4_0
    case 3:                        // Q4_1
    case 6:                        // Q5_0
    case 7:                        // Q5_1
    case 8:                        // Q8_0
    case 1:                        // F16
        return 8;                  // -> Q8_0
    case 29:                       // IQ1_M
    case 16:                       // IQ2_XXS
    case 17:                       // IQ2_XS
    case 22:                       // IQ2_S
    case 18:                       // IQ3_XXS
    case 21:                       // IQ3_S
    case 19:                       // IQ1_S
    case 23:                       // IQ4_XS
    case 10:                       // Q2_K
    case 11:                       // Q3_K
    case 12:                       // Q4_K
    case 13:                       // Q5_K
    case 14:                       // Q6_K
        return 15;                 // -> Q8_K
    case 30:                       // BF16
        return 30;                 // -> BF16
    default:
        return -1;
    }
}

// ggml's block size (values per block) for the types `iq_row_bytes` lays out.
static int block_size_of(int t) noexcept {
    switch (t) {
    case 42: return 64;            // Q2_0
    case 20: return 32;            // IQ4_NL
    case 2: case 6: case 7: case 8: case 1: return 32;
    case 30: return 1;             // BF16
    default: return 256;           // every i-quant and K-quant superblock
    }
}

/// The ggml row size the pack was built with, from the port's ONE table.  `iq_row_bytes` is a LOUD REFUSAL for a
/// type it has no layout for, which is the right answer here too.
static size_t row_bytes_of(int t, int64_t n) {
    return strata::kernels::iq_row_bytes(t, n);
}

// `native_experts_available()` - "this build has the native expert path".  TRUE: the LAYOUT half is present and
// real (above).  It says nothing about the CPU rows, which refuse below by design.
bool native_experts_available() noexcept { return true; }

bool native_fmt(int gu_type, int d_type, int64_t n_embd, int64_t n_ff, NativeFmt& f, std::string& err) {
    const int gu_act = vec_dot_type_of(gu_type);
    const int d_act = vec_dot_type_of(d_type);
    if (gu_act < 0 || d_act < 0) {
        err = "native experts: ggml-cpu has no dot product for type " + std::to_string(gu_act < 0 ? gu_type : d_type);
        return false;
    }
    // The activation quantisers the CUDA's `native_fmt` also requires (`traits(vec_dot_type)->from_float`): Q8_0
    // and Q8_K both have one; anything else here is a geometry this port would have to port first.
    if ((gu_act != 8 && gu_act != 15) || (d_act != 8 && d_act != 15)) {
        err = "native experts: ggml-cpu cannot quantize an activation for this layer";
        return false;
    }
    const int bgu = block_size_of(gu_type), bd = block_size_of(d_type);
    if (n_embd % bgu || n_ff % bd || n_embd % block_size_of(gu_act) || n_ff % block_size_of(d_act)) {
        err = "native experts: expert geometry is not whole blocks";
        return false;
    }
    f.gu_type = gu_type;
    f.d_type = d_type;
    f.gu_act = gu_act;
    f.d_act = d_act;
    f.n_embd = n_embd;
    f.n_ff = n_ff;
    f.gu_row = row_bytes_of(gu_type, n_embd);
    f.d_row = row_bytes_of(d_type, n_ff);
    f.up_off = f.gu_row * (size_t) n_ff;
    f.down_off = 2 * f.up_off;
    f.bytes = f.down_off + f.d_row * (size_t) n_embd;
    f.act_bytes = row_bytes_of(gu_act, n_embd);
    f.h_bytes = row_bytes_of(d_act, n_ff);
    if (f.act_bytes > kNativeActBytes || f.h_bytes > kNativeHBytes) {
        err = "native experts: activation larger than the pool's buffers";
        return false;
    }
    return true;
}

// ---- THE CPU-HYBRID ROWS: LOUD REFUSALS --------------------------------------------------------------------
// Named by the flag chain and the GPU kernel that replaces each, so a run that reaches one stops with a reason
// rather than a silently zero expert contribution (which is a wrong token, not a slow one).
[[noreturn]] static void refuse_cpu_row(const char* who) {
    std::fprintf(stderr,
                 "strata::kernels::cpu::%s: the ggml-cpu native (IQ) expert rows are NOT built in the Vulkan "
                 "configuration - this port has no CPU-hybrid execution path, and the token must come from the GPU "
                 "backend.  REACHED BECAUSE THE PLAN BRANCH LEFT `kind[i] == -1` FOR A MISSED EXPERT "
                 "(expert_source.cpp:2043-2067): a missed expert becomes GPU work (`kind = 1`) only when BOTH "
                 "`d.pcie_num > 0` AND `d.src->pinned(l, e)` holds (a blob in the registered/pinned host tier).  "
                 "On this port NOTHING is registered: `cudaHostRegister` is `cudaErrorNotSupported` (the port has "
                 "no shader-addressable view of the engine's host expert arena - a shader binds only the single "
                 "arena VkBuffer, and `fetch_blobs`/`native_expert_grouped` rebase every source by "
                 "`ptr - kArenaBase`), so `pinned()` is false for every expert and `d.src->pcie_layer()` is false "
                 "too, which pins `m = 0` as well.  `native_expert_grouped` IS ported (PORT-MAP.tsv `kernel`); "
                 "the P6 verify window RUNS and RECORDS to its capture.  What is missing is the pinned host tier, "
                 "not the grouped kernels.  Refusing rather than computing a subset of the experts.\n",
                 who);
    std::exit(2);
}

void native_quant_act(const NativeFmt&, const float*, void*) { refuse_cpu_row("native_quant_act"); }
void native_quant_h(const NativeFmt&, const float*, void*) { refuse_cpu_row("native_quant_h"); }
void native_gu_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    refuse_cpu_row("native_gu_rows");
}
void native_down_rows(const NativeFmt&, const uint8_t*, const void* const*, int, float* const*, int, int) {
    refuse_cpu_row("native_down_rows");
}

}  // namespace strata::kernels::cpu
