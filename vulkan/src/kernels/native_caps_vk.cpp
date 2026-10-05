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
//                                   (layer.cpp:476-491), which demands the native BF16 projections, the fused
//                                   hyper-connection read, the fused native GDN kernels, the split-K decode
//                                   attention, the block top-k selection AND the native QSA indexer, ALL answering
//                                   true.  After batch 4 flipped `native_gdn_enabled()` true, that conjunction is
//                                   STILL false: `native_bf16_projections` is a setting that defaults false,
//                                   `g_fused_gr` defaults false, and `native_qsa_indexer_enabled()` is false
//                                   (unported).  So the verifier is unreachable and the reachable set is
//                                   {native_router_top10}, which is ported.  ANSWER: true.
//
//   native_moe_combine_enabled() -> native_moe_combine (layer.cpp:463, mtp.cpp:601) on the forward path, and
//                                   native_moe_combine_multi (verify.cpp:1081).  Same argument as the router:
//                                   the `_multi` variant is verifier-only and the verifier is unreachable.
//                                   ANSWER: true.
//
//   native_qsa_enabled()         -> native_qsa_rms_norm_weighted (layer.cpp:879, mtp.cpp:488/491/514, verify.cpp:767)
//                                   AND native_qsa_gate_apply (layer.cpp:1010, verify.cpp:883/887) - ONE flag
//                                   gates BOTH.  Both now have a shader (`native_qsa_rms_norm_weighted.comp` from
//                                   batch 1; `native_qsa_gate_apply.comp` lands with this batch), so every gated
//                                   symbol is built and the flag answers TRUE.  The flip was checked call site by
//                                   call site: the verify.cpp sites are owned by the P6 verifier, which cannot
//                                   init under this contract (`layer_verify_compatible`, layer.cpp:476-491, also
//                                   demands `native_bf16_projections`, `g_fused_gr` and `native_qsa_indexer_enabled()`,
//                                   all false here), and even if reached they dispatch only ported symbols.
//                                   ANSWER: true.
//
//   native_qsa_indexer_enabled() -> native_qsa_indexer_append (layer.cpp:945; verify.cpp:819/1305/1859) when TRUE,
//                                   else the ported `indexer_key_append` (layer.cpp:948).  The indexer runs on the
//                                   main forward path of all 12 QSA layers EVERY token, and the shipped launch
//                                   (`--native`, generate.cpp:1807) sets the option TRUE.  `native_qsa_indexer_append`
//                                   has NO shader in this tree, so the backend MUST answer FALSE - and until this
//                                   batch it answered NOTHING: the getter/setter were declared
//                                   (`native_qsa_indexer.hpp:9-10`), called from src/core (layer.cpp:483,
//                                   layer.cpp:944) and set from the option plumbing (generate.cpp:2294), but no
//                                   Vulkan definition existed.  That is the reachability hole this batch closes (see
//                                   plan/DECODE-PATH-TRIAGE.md's REACHABILITY AUDIT): without the answer, the
//                                   engine's `--native` option would select the UNPORTED symbol.  ANSWER: false,
//                                   which selects the ported `indexer_key_append`.
//
//   native_gdn_enabled()         -> SIX native GDN kernels (`native_gdn_conv_silu`, `native_gdn_l2_norm`,
//                                   `native_gdn_beta_gate` at layer.cpp:253/266-267/296; `native_gdn_gate`,
//                                   `native_gdn_step`, `native_gdn_out_norm` at layer.cpp:297/308/324) AND the
//                                   THREE fused paths `fused_gdn_conv_l2` / `fused_gdn_ab` / `fused_gdn_step_norm`
//                                   (layer.cpp:250/287/322).  This backend has ported ALL NINE (batch 4 lands the
//                                   three fused paths), so EVERY gated symbol has a shader and the flag answers
//                                   TRUE - the invariant `case_native_capabilities` enforces.  The fused paths'
//                                   extra runtime conditions are SETTINGS, not capabilities this backend answers:
//                                   `g_fused_gdn` defaults true (layer.cpp:42), and `native_bf16_projections`
//                                   defaults false and is set from `--native-bf16`/`--native` (generate.cpp:2286).
//                                   Both of their dependencies are ported: the fused step+norm is self-contained,
//                                   and the conv_l2/ab paths' bf16 dependency is `bf16_mmvf_f32`, which is
//                                   ported and gated.  So a flag flip cannot route the engine at an unported
//                                   symbol.  ANSWER: true.
//
// THE SETTERS are the engine's option plumbing (`generate.cpp` calls `native_X_set_enabled(o.native_X)`).  On this
// backend they do NOT decide the answer: the backend reports its own implementation, so a `--native` launch
// cannot talk it into selecting a symbol it has not ported.  They are no-ops that exist so the engine's option
// resolution links once it is compiled under STRATA_ENABLE_VULKAN (a later increment), and so the single writer
// of these answers is this file.
//
// HOW TO CHECK IT: the numeric gate's `case_native_capabilities` includes these same headers, calls the six
// getters (rope / router / moe_combine / qsa / qsa_indexer / gdn), requires the exact answers above, and requires
// each ported symbol's .spv to be present - so a capability cannot answer true for a symbol whose shader was
// deleted.  The `qsa` and `gdn` flags are checked as INVARIANTS rather than hard-coded booleans: each must equal
// "every symbol that flag gates has a built shader" (both TRUE now - the two QSA gated symbols, and the nine GDN
// gated symbols: six native kernels + three fused paths).  The invariant is falsified in BOTH directions (a flag
// that over-claims AND one that under-claims must each fail the arm) - `inject-verify.sh native-caps-qsa-false` /
// `native-caps-gdn-false` answer a true flag FALSE while every gated shader exists.
#if !defined(STRATA_ENABLE_VULKAN)
#error "native_caps_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
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

// ---- native QSA: both gated symbols are now ported, so the flag answers TRUE -------------------------------
void native_qsa_set_enabled(bool) { /* see the header note */ }
bool native_qsa_enabled() { return true; }         // native_qsa_rms_norm_weighted + native_qsa_gate_apply, both built

// ---- native QSA INDEXER: the flag gates the UNPORTED native_qsa_indexer_append - answer FALSE ----------------
// This getter/setter pair is the reachability hole closed by this batch (plan/DECODE-PATH-TRIAGE.md's
// REACHABILITY AUDIT): src/core calls it (layer.cpp:483, :944) and the option plumbing sets it
// (generate.cpp:2294) but NO Vulkan definition existed, so the shipped `--native` option (which sets it TRUE,
// generate.cpp:1807) would have selected `native_qsa_indexer_append` - a symbol with no shader - on every QSA
// layer.  Answering FALSE selects the ported legacy `indexer_key_append` (layer.cpp:948).
void native_qsa_indexer_set_enabled(bool) { /* see the header note */ }
bool native_qsa_indexer_enabled() { return false; } // native_qsa_indexer_append unported; indexer_key_append is ported

// ---- native GDN: ALL NINE gated symbols (six native kernels + three fused paths) are now ported -------------
// `native_gdn_conv_silu`, `native_gdn_l2_norm`, `native_gdn_beta_gate`, `native_gdn_gate`, `native_gdn_step`,
// `native_gdn_out_norm` (batches 2-3) AND `fused_gdn_conv_l2`, `fused_gdn_ab`, `fused_gdn_step_norm` (batch 4)
// all have shaders and gated cases, so the flag answers TRUE.  The fused paths' extra gates (`g_fused_gdn`,
// `native_bf16_projections`) are engine SETTINGS whose dependencies are ported - see the header note.
void native_gdn_set_enabled(bool) { /* see the header note */ }
bool native_gdn_enabled() { return true; }         // all nine gated symbols have shaders (six native + three fused)

}  // namespace strata::kernels
