// vulkan/src/kernels/refusals_prefill_vk.cpp - THE PROMPT-PATH LOUD REFUSALS THAT REMAIN.
//
// `src/prefill/prefill.cpp` is HOST orchestration that COMPILES against the CUDA-runtime shim, so it joins the
// Vulkan engine target and lets the program LINK.  The batched KERNELS it drives live in `src/prefill/*.cu`
// (`kernels.cu`, `gemm.cu`, `moe_fused*.cu`, `moe_mmq.cu`).  The subset a chunk reaches FIRST is now PORTED
// (vulkan/src/kernels/prefill_vk.cpp: the whole `Gemm` class, the hyper-connection family, and the thin
// elementwise/copy set).  This file answers the REMAINING entry points with a real definition whose only
// behaviour is to REFUSE, naming the prompt path.  Same discipline as `refusals_vk.cpp`.
#if !defined(STRATA_ENABLE_VULKAN)
#error "refusals_prefill_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/prefill/kernels.hpp"           // the batched prompt kernels still unported
#include "strata/kernels/iq_kernels.hpp"        // iq_dequant_f16 / iq_dequant_gu_f16 (the batched IQ readers)
#include "strata/kernels/kv_q4.hpp"             // kv_append_q4
#include "strata/kernels/kv_stream.hpp"         // kv_stage_from_host
#include "strata/kernels/native_ple_postops.hpp"// native_ple_postops_batch (PleWeights)
#include "strata/kernels/native_qsa_indexer.hpp"// native_qsa_indexer_append_batch
#include "strata/kernels/qsa_prompt_attn.hpp"   // qsa_prompt_attn_batch

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
[[noreturn]] void refuse_prompt(const char* sym) {
    std::fprintf(stderr,
                 "%s: NOT PORTED on the Vulkan backend - REFUSING.\n"
                 "  Reached only by the batched PROMPT path (--prefill / a request with a prompt); the remaining\n"
                 "  src/prefill entry points are the DELIVERABLE-B list in the port's report.  A bare single-token\n"
                 "  decode does not run it.  This definition exists so the `strata` program LINKS; it is a LOUD\n"
                 "  REFUSAL, never a silent fallback.\n",
                 sym);
    std::exit(2);
}
}  // namespace

namespace strata::prefill {

// `native_qsa_indexer_append` (layer.cpp:945) is PORTED (qsa_vk.cpp) and the flag is TRUE; `kv_append` and
// `qsa_prompt_attn_batch` are PORTED in prefill_vk.cpp.  The remaining prompt-path refusals are the pack/flag
// cases this backend is not running (kv_q4, kv_mode 1) plus the two diagnostics and the batched PLE block.
// `blob_dequant_f16` is the Q2_0 pack's expert blob -> FP16 reader (a pack variant this port is not running, and
// the engine's own `native` expert route is a different kernel).  `round_f16` is only the
// STRATA_IDX_FP16_CHECK diagnostic (kernels.cu:813).
void blob_dequant_f16(const uint8_t*, uint16_t*, uint16_t*, void*) { refuse_prompt("strata::prefill::blob_dequant_f16"); }
void round_f16(const float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::round_f16"); }

}  // namespace strata::prefill

// ---- the prefill-path kernels-NAMESPACE symbols still unported ---------------------------------------------
namespace strata::kernels {

void kv_append_q4(uint8_t*, uint8_t*, const int32_t*, int64_t, int64_t, const float*, const float*, const QsaShapes&, void*,
                  const KvHostPools*, const KvHostPools*) { refuse_prompt("strata::kernels::kv_append_q4"); }
void kv_stage_from_host(const QsaAttnPools&, const KvHostPools&, int, int64_t, const QsaShapes&, void*) { refuse_prompt("strata::kernels::kv_stage_from_host"); }
void native_ple_postops_batch(float*, float*, const float*, float*, const PleWeights&, float*, float*, float*, int, void*) { refuse_prompt("strata::kernels::native_ple_postops_batch"); }

}  // namespace strata::kernels
