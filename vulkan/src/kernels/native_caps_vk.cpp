// vulkan/src/kernels/native_caps_vk.cpp - THE VULKAN BACKEND'S OWN ANSWER TO THE NATIVE CAPABILITY CHECKS.
//
// WHY THIS FILE EXISTS.  include/strata/kernels/native_*.hpp declare `native_X_enabled()` / `native_X_set_enabled()`
// and an entry point per symbol.  On a CUDA/HIP build those are answered from src/kernels/cuda/native_*.cu; the
// engine's option resolution (`src/program/generate.cpp` ~2290) turns the flags on for a `--native` launch.  A
// Vulkan build compiles none of the .cu files, so it must answer the checks ITSELF - and the answer is a
// property of what this backend has actually implemented, not of a CLI flag.  That is the "capability contract"
// the port's plan describes: the backend reports the branch the engine may take.
//
// THE RULE HERE, and it is a SYMBOL-AT-A-TIME truth rather than a blanket `true`:
//
//   a capability check answers `true` only when EVERY kernels-namespace symbol that this check selects has a
//   shader in this port.  If a check also gates a symbol with no shader, the check answers `false` - because the
//   engine would otherwise dispatch an unimplemented symbol.
//
// Per check, from the call sites in src/core/ (the lines are the evidence):
//
//   native_rope_enabled()        -> native_rope_apply  (layer.cpp:881, mtp.cpp:515, verify.cpp:769).  Every
//                                   caller dispatches ONLY native_rope_apply, which this increment ports
//                                   (`native_rope_apply.comp`, gated by case_native_rope_apply).  ANSWER: true.
//
//   native_router_enabled()      -> native_router_top10 (layer.cpp:370, mtp.cpp:576) on the forward path, and
//                                   native_router_top10_multi in verify.cpp:916.  The `_multi` symbol is not
//                                   ported, BUT the verifier cannot run under this port's contract:
//                                   `Verifier::init` refuses unless `layer_verify_compatible()` holds
//                                   (layer.cpp:476-491), which demands the native GDN and the native QSA
//                                   indexer - both answered false.  So the reachable set is {native_router_top10},
//                                   which is ported.  ANSWER: true.
//
//   native_moe_combine_enabled() -> native_moe_combine (layer.cpp:463, mtp.cpp:601) on the forward path, and
//                                   native_moe_combine_multi (verify.cpp:1081).  Same argument as the router:
//                                   the `_multi` variant is verifier-only and the verifier is unreachable.
//                                   ANSWER: true.
//
//   native_qsa_enabled()         -> native_qsa_rms_norm_weighted (layer.cpp:879, mtp.cpp:488, verify.cpp:767)
//                                   AND native_qsa_gate_apply (layer.cpp:1010, verify.cpp:883) - ONE flag
//                                   selects BOTH, and `native_qsa_gate_apply` has NO shader in this tree and is
//                                   on the MAIN forward path (every QSA layer, 12 of 48).  Answering true would
//                                   make layer.cpp:1010 call an unimplemented symbol.  ANSWER: false, and this
//                                   is the honest reading of "report what it has actually implemented": the
//                                   ported symbol cannot be selected on its own while an unported sibling
//                                   shares its switch.  `native_qsa_rms_norm_weighted.comp` is still ported and
//                                   gated (case_native_qsa_rms_norm_weighted) so the symbol is READY - the flag
//                                   turns on with `native_qsa_gate_apply`'s shader, not before.
//
//   native_gdn_enabled()         -> SIX native GDN kernels (`native_gdn_conv_silu`, `native_gdn_l2_norm`,
//                                   `native_gdn_beta_gate` at layer.cpp:253/266-267/296; `native_gdn_gate`,
//                                   `native_gdn_step`, `native_gdn_out_norm` at layer.cpp:297/308/324) AND the
//                                   THREE fused paths `fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm`
//                                   (layer.cpp:250/287/322, additionally gated on `g_fused_gdn` and
//                                   `native_bf16_projections`).  This backend has ported the first three
//                                   (`gdn_conv_silu`, `gdn_l2_norm`, `beta_gate` shaders); the other THREE GDN
//                                   kernels and ALL THREE fused paths have no shader.  Answering true would
//                                   dispatch them.  ANSWER: false - and the answer is not "nothing is
//                                   implemented": the three ported shaders exist and are gated.  The flag stays
//                                   false until EVERY symbol it selects has a shader, which is what
//                                   `case_native_capabilities`'s gdn arm enforces (it requires the flag to equal
//                                   "every gated symbol has a built shader").
//
// THE SETTERS are the engine's option plumbing (`generate.cpp` calls `native_X_set_enabled(o.native_X)`).  On this
// backend they do NOT decide the answer: the backend reports its own implementation, so a `--native` launch
// cannot talk it into selecting a symbol it has not ported.  They are no-ops that exist so the engine's option
// resolution links once it is compiled under STRATA_ENABLE_VULKAN (a later increment), and so the single writer
// of these answers is this file.
//
// HOW TO CHECK IT: the numeric gate's `case_native_capabilities` includes these same headers, calls the five
// getters, requires the exact answers above, and requires each ported symbol's .spv to be present - so a
// capability cannot answer true for a symbol whose shader was deleted.  Its GDN arm is the invariant rather
// than a hard-coded boolean: `native_gdn_enabled()` must equal "every symbol this flag gates has a built
// shader" (currently false, because the six unported GDN symbols and the three fused paths have none).
#if !defined(STRATA_ENABLE_VULKAN)
#error "native_caps_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/native_router.hpp"

namespace strata::kernels {

// ---- native RoPE: native_rope_apply (ported) -----------------------------------------------------------------
void native_rope_set_enabled(bool) { /* the backend answers for itself; see the header note */ }
bool native_rope_enabled() { return true; }        // native_rope_apply.comp, gated

// ---- native router: native_router_top10 (ported; `_multi` is verifier-only and unreachable) ------------------
void native_router_set_enabled(bool) { /* see the header note */ }
bool native_router_enabled() { return true; }      // native_router_top10.comp, gated

// ---- native MoE combine: native_moe_combine (ported; `_multi` is verifier-only and unreachable) --------------
void native_moe_combine_set_enabled(bool) { /* see the header note */ }
bool native_moe_combine_enabled() { return true; } // native_moe_combine.comp, gated

// ---- native QSA: the flag also gates the UNPORTED native_qsa_gate_apply (layer.cpp:1010) ---------------------
void native_qsa_set_enabled(bool) { /* see the header note */ }
bool native_qsa_enabled() { return false; }        // see the header note: shared switch, sibling unported

// ---- native GDN: the flag also gates the UNPORTED three GDN kernels + three fused paths -----------------------
// `native_gdn_conv_silu` / `native_gdn_l2_norm` / `native_gdn_beta_gate` ARE ported and gated, but the SAME flag
// also selects `native_gdn_gate` / `native_gdn_step` / `native_gdn_out_norm` and the three `fused_gdn_*` paths,
// none of which has a shader - so the answer is false and must stay false while any of them is missing.  This is
// the "symbol-at-a-time truth" rule; `case_native_capabilities`'s gdn arm asserts the invariant directly.
void native_gdn_set_enabled(bool) { /* see the header note */ }
bool native_gdn_enabled() { return false; }        // see the header note: six gated symbols unported

}  // namespace strata::kernels
