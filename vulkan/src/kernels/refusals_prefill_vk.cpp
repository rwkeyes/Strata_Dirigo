// vulkan/src/kernels/refusals_prefill_vk.cpp - THE PROMPT-PATH LOUD REFUSALS.
//
// `src/prefill/prefill.cpp` is HOST orchestration that COMPILES against the CUDA-runtime shim (it only needed
// `cudaMemcpyPeerAsync`, now refused in the shim), so it joins the Vulkan engine target and lets the program LINK.
// The batched KERNELS it drives live in `src/prefill/*.cu` (`kernels.cu`, `gemm.cu`, `moe_fused*.cu`, `moe_mmq.cu`)
// and are NOT ported: this file gives each a real definition whose only behaviour is to REFUSE, naming the prompt
// path.  A bare single-token decode never enters `Prefill`; the symbols exist so the executable links and so a
// prompt-bearing run is a NAMED error rather than a link failure.  Same discipline as `refusals_vk.cpp`.
//
// THE GEMM CONSTRUCTOR AND DESTRUCTOR ARE REAL, EMPTY BODIES.  `Gemm` is default-constructed and destroyed on the
// request path even when no prompt is run, and it has no opaque `Impl` (only raw pointers + ints), so an empty
// destructor is exact.  Refusing it would abort a decode that never projected a prompt.
#if !defined(STRATA_ENABLE_VULKAN)
#error "refusals_prefill_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/prefill/gemm.hpp"              // Gemm (bf16/f16/native/init_external/rebind)
#include "strata/prefill/kernels.hpp"           // the batched prompt kernels
#include "strata/kernels/iq_kernels.hpp"        // iq_dequant_f16 / iq_dequant_gu_f16 (the batched IQ readers)
#include "strata/kernels/kv_q4.hpp"             // kv_append_q4
#include "strata/kernels/kv_stream.hpp"         // kv_stage_from_host
#include "strata/kernels/native_ple_postops.hpp"// native_ple_postops_batch (PleWeights)
#include "strata/kernels/native_qsa_indexer.hpp"// native_qsa_indexer_append_batch
#include "strata/kernels/qsa_select.hpp"        // qsa_block_scores_tc
#include "strata/kernels/qsa_prompt_attn.hpp"   // qsa_prompt_attn_batch

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
[[noreturn]] void refuse_prompt(const char* sym) {
    std::fprintf(stderr,
                 "%s: NOT PORTED on the Vulkan backend - REFUSING.\n"
                 "  Reached only by the batched PROMPT path (--prefill / a request with a prompt); src/prefill/\n"
                 "  is not ported (BACKEND-INTEGRATION.md §4).  A bare single-token decode does not run it.\n"
                 "  This definition exists so the `strata` program LINKS; it is a LOUD REFUSAL, never a silent\n"
                 "  fallback.\n",
                 sym);
    std::exit(2);
}
}  // namespace

namespace strata::prefill {

Gemm::~Gemm() {}

void blob_dequant_f16(const uint8_t*, uint16_t*, uint16_t*, void*) { refuse_prompt("strata::prefill::blob_dequant_f16"); }
void copy_f32_wide(float*, const float*, int64_t, void*) { refuse_prompt("strata::prefill::copy_f32_wide"); }
void copy_i32(int32_t*, const int32_t*, int64_t, void*) { refuse_prompt("strata::prefill::copy_i32"); }
void gate_attn(const float*, const float*, uint16_t*, int64_t, void*) { refuse_prompt("strata::prefill::gate_attn"); }
void gather_rows16(const uint16_t*, const int32_t*, uint16_t*, int64_t, int64_t, void*) { refuse_prompt("strata::prefill::gather_rows16"); }
void gdn_conv(float*, const float*, const float*, float*, int64_t, float, void*) { refuse_prompt("strata::prefill::gdn_conv"); }
void gdn_gates(const float*, const float*, const float*, float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::gdn_gates"); }
void gdn_recurrence(float*, const float*, const float*, const float*, const float*, const float*, float, float*, uint16_t*,
                    int64_t, void*) { refuse_prompt("strata::prefill::gdn_recurrence"); }
void gr_broadcast(const float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::gr_broadcast"); }
void gr_mix(const float*, const float*, float*, uint16_t*, int64_t, void*, uint16_t*, uint16_t*) { refuse_prompt("strata::prefill::gr_mix"); }
void gr_mix_r(const float*, const float*, const float*, const float*, float*, uint16_t*, int64_t, void*, uint16_t*,
              uint16_t*) { refuse_prompt("strata::prefill::gr_mix_r"); }
void gr_norm(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*) { refuse_prompt("strata::prefill::gr_norm"); }
void gr_norm_rs(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*) { refuse_prompt("strata::prefill::gr_norm_rs"); }
void gr_silu(const float*, uint16_t*, int64_t, void*, uint16_t*) { refuse_prompt("strata::prefill::gr_silu"); }
void gr_write(float*, const float*, const float*, int64_t, int64_t, void*) { refuse_prompt("strata::prefill::gr_write"); }
void gr_write_norm_rs(float*, const float*, const float*, int64_t, const float*, float, float*, uint16_t*, int64_t, void*,
                      uint16_t*) { refuse_prompt("strata::prefill::gr_write_norm_rs"); }
void kv_append(const float*, const float*, int64_t, int64_t, const int32_t*, int64_t, uint16_t*, uint16_t*, int8_t*,
               int8_t*, uint16_t*, uint16_t*, void*, const strata::kernels::KvHostPools*,
               const strata::kernels::KvHostPools*) { refuse_prompt("strata::prefill::kv_append"); }
void moe_combine(const float*, const int32_t*, const float*, const float*, const float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::moe_combine"); }
void rms_rows(float*, const float*, int64_t, int64_t, int64_t, float, void*) { refuse_prompt("strata::prefill::rms_rows"); }
void rope(float*, int64_t, int64_t, int64_t, int64_t, int64_t, const strata::kernels::RopeScaling&, void*) { refuse_prompt("strata::prefill::rope"); }
void round_f16(const float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::round_f16"); }
void route(const float*, int32_t*, float*, int64_t, int64_t, void*) { refuse_prompt("strata::prefill::route"); }
void split_q(const float*, float*, int64_t, void*) { refuse_prompt("strata::prefill::split_q"); }
void swiglu_interleaved(const float*, uint16_t*, int64_t, void*) { refuse_prompt("strata::prefill::swiglu_interleaved"); }
void swiglu_pair(const float*, const float*, uint16_t*, int64_t, void*) { refuse_prompt("strata::prefill::swiglu_pair"); }
void to_bf16(const float*, uint16_t*, int64_t, void*, uint16_t*) { refuse_prompt("strata::prefill::to_bf16"); }
void to_f16(const float*, uint16_t*, int64_t, void*) { refuse_prompt("strata::prefill::to_f16"); }

bool Gemm::init_external(void*, uint16_t*, int64_t, void*, size_t, std::string&) { refuse_prompt("strata::prefill::Gemm::init_external"); }
void Gemm::bf16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float) { refuse_prompt("strata::prefill::Gemm::bf16"); }
void Gemm::f16(const uint16_t*, const uint16_t*, float*, int64_t, int64_t, int64_t, int64_t, float) { refuse_prompt("strata::prefill::Gemm::f16"); }
void Gemm::native(const uint16_t*, int, const void*, float*, int64_t, int64_t, int64_t, int64_t, float) { refuse_prompt("strata::prefill::Gemm::native"); }
void Gemm::rebind(uint16_t*, int64_t, void*, size_t) { refuse_prompt("strata::prefill::Gemm::rebind"); }

}  // namespace strata::prefill

// ---- the prefill-path kernels-NAMESPACE symbols prefill.cpp also reaches --------------------------------------
namespace strata::kernels {

void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*) { refuse_prompt("strata::kernels::iq_dequant_f16"); }
void iq_dequant_gu_f16(int, const void*, const void*, int64_t, int64_t, uint16_t*, void*) { refuse_prompt("strata::kernels::iq_dequant_gu_f16"); }
void kv_append_q4(uint8_t*, uint8_t*, const int32_t*, int64_t, int64_t, const float*, const float*, const QsaShapes&, void*,
                  const KvHostPools*, const KvHostPools*) { refuse_prompt("strata::kernels::kv_append_q4"); }
void kv_stage_from_host(const QsaAttnPools&, const KvHostPools&, int, int64_t, const QsaShapes&, void*) { refuse_prompt("strata::kernels::kv_stage_from_host"); }
void native_ple_postops_batch(float*, float*, const float*, float*, const PleWeights&, float*, float*, float*, int, void*) { refuse_prompt("strata::kernels::native_ple_postops_batch"); }
void native_qsa_indexer_append_batch(const float*, int64_t, int64_t, int32_t, const float*, float, const QsaIndexerBuffers&,
                                     const QsaShapes&, int64_t, const RopeScaling&, void*) { refuse_prompt("strata::kernels::native_qsa_indexer_append_batch"); }
bool qsa_block_scores_tc(const float*, const float*, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&,
                         float*, void*, int64_t) { refuse_prompt("strata::kernels::qsa_block_scores_tc"); }
bool qsa_prompt_attn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                           float*, int64_t, void*) { refuse_prompt("strata::kernels::qsa_prompt_attn_batch"); }

}  // namespace strata::kernels
