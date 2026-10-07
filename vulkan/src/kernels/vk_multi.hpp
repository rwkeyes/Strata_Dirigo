// vulkan/src/kernels/vk_multi.hpp - INTERNAL: the ONE-DISPATCH-PER-WINDOW forms of the verify window's
// per-token kernels.
//
// WHY THIS EXISTS.  The P6 verify window is a native (IQ) pack's ONLY decode path, and its cost is a DISPATCH
// COUNT: ~4,792 recorded dispatches per round at this port's measured ~46-54 us small-dispatch cost.  The
// per-token term inside it was a host-side C++ loop that dispatched the single-token kernel once per draft
// (`gdn_conv_l2_multi`, `gdn_ab_multi`, `native_router_top10_multi`, `shared_expert_multi`,
// `native_moe_combine_multi`).  Each `_multi` now issues ONE dispatch that covers every token of the round.
//
// THE CONTRACT KEPT.  `verify_kernels.hpp` states each `_multi` is "bitwise <single-token kernel> per token".
// Every form here keeps that BY CONSTRUCTION: the per-token arithmetic is untouched, only the index base moves
// (a token dimension in the shader's grid, or the module's own `ncols` column dimension).  The single-token
// entry points are unchanged - they call these with n_tok == 1.
#pragma once

#include <cstdint>

#include "strata/vulkan/vk_backend.hpp"   // strata::vulkan::Stream

namespace strata::vulkan {

// `native_router_top10.spv` already carries a token dimension (`gl_WorkGroupID.x` is the token, `n_tokens` the
// count): one workgroup per token.  n_tok workgroups = one dispatch for the window.
void router_top10_n(Stream& s, const float* logits, int32_t* ids, float* weights, int64_t n_tok, int n_expert);

// `native_moe_combine.spv` gains the token as `gl_WorkGroupID.y`; one flat grid over n_embd*n_tok elements.
void native_moe_combine_n(Stream& s, const float* parts, const float* weights, const float* shared, float* output,
                          int64_t n_embd, int64_t k, int64_t n_tok);

// `fused_gdn_ab.spv` gains the token as `gl_WorkGroupID.y`; the BF16 weights are token-invariant.
void fused_gdn_ab_n(Stream& s, const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt,
                    const float* ssm_a, float* gate, float* beta, int64_t n_embd, int64_t h_v, int64_t n_tok);

// `fused_gdn_conv_l2.spv` gains the token as `gl_WorkGroupID.y` AND a token-invariant formulation of the
// running window: token t's four taps are stream[t..t+3] of the concatenation [history(3) | qkv], read from
// `history`/`qkv` directly instead of from a slid working copy.  `write_hist == 0` is the multi's contract
// ("history is NOT written"); the single-token entry passes 1.
void fused_gdn_conv_l2_n(Stream& s, const float* history, const float* qkv, const float* conv_w, float* h,
                         int64_t channels, int64_t qk_heads, float eps, int64_t t_begin, int64_t n_tok,
                         int write_hist);

}  // namespace strata::vulkan
