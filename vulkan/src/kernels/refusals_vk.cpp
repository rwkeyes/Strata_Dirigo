// vulkan/src/kernels/refusals_vk.cpp - THE LOUD REFUSALS: the engine symbols the SHIPPED configuration
// cannot reach, defined so the layer body LINKS and so a reach is a named, fatal error rather than a silent
// fallback.
//
// WHY THIS FILE EXISTS.  `ports/vulkan/NEXT.md` measured a one-layer-body link of 64 undefined references.
// Some of those references are on symbols whose ONLY reaching configuration is one the shipped launch (and the
// backend's own capability answers) does NOT select.  A `todo` row cannot simply be left undefined - the linker
// needs a definition - and it must NOT be answered with a plausible wrong body either: a wrong body is a wrong
// token that looks like a right one.  So each such symbol gets a definition whose only behaviour is to NAME the
// flag chain that would reach it and REFUSE.
//
// THE RULE EACH ONE WAS CLASSIFIED UNDER (the audit is in `ports/vulkan/plan/DECODE-PATH-TRIAGE.md`, "THE
// REACHABILITY AUDIT", and each entry below quotes the call site, the enclosing condition, every flag, each
// flag's DEFAULT, and what the shipped launch sets).  **A CLASS LABEL IS A ROUTING DECISION, and the last batch
// learned that the expensive way: `qsa_decode_attn_step` was called unreachable and WAS reachable, which cost a
// shader port found only later.**  So the classification was re-derived from the engine's own code this batch,
// and the ONE row that was wrong (`fused_gr_read`) is fixed at its cause rather than papered over - see the
// `fused_gr_supported` note in `ple_vk.cpp`.
//
// WHAT IS NOT HERE: a symbol is only in this file if it is unreachable under EVERY shipped configuration, AFTER
// the backend's own answers are taken into account.  A reachable symbol is a HOLE and gets a shader; a symbol
// whose reachability depends on a backend answer gets that answer fixed instead (the `fused_gr_supported` case).
#if !defined(STRATA_ENABLE_VULKAN)
#error "refusals_vk.cpp is the Vulkan backend: compile it only in a -DSTRATA_ENABLE_VULKAN=1 build"
#endif

// The engine headers that DECLARE each symbol - included so the definitions below are checked against the
// engine's own signatures (a mismatch is a compile error here, not a link-time surprise).
#include "strata/kernels/fused_gr.hpp"          // fused_gr_read (FusedGrArgs)
#include "strata/kernels/kv_stream.hpp"         // kv_ring_table / kv_stream_reset / kv_stream_resolve
#include "strata/kernels/native_flash_attn.hpp" // native_flash_attn_short_step
#include "strata/kernels/native_qsa_indexer.hpp"// native_qsa_indexer_append (QsaIndexerBuffers, RopeScaling)
#include "strata/kernels/qsa.hpp"               // qsa_attend_step / qsa_index_step / topk_512_step

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

// ONE refusal body: it names the symbol, the SHIPPED configuration that does NOT reach it, and the exact flag
// chain that WOULD.  `std::exit(2)` (not an exception): a reach here is a configuration the backend does not
// implement, and the port's rule is to refuse loudly rather than degrade.
[[noreturn]] static void refuse_unreachable(const char* sym, const char* chain) {
    std::fprintf(stderr,
                 "strata::kernels::%s: NOT PORTED and NOT REACHED by the shipped configuration - REFUSING.\n"
                 "  The only configuration that reaches it: %s\n"
                 "  This definition exists so the layer body LINKS; it is a LOUD REFUSAL, never a silent\n"
                 "  fallback.  The reachability audit is in ports/vulkan/plan/DECODE-PATH-TRIAGE.md.\n",
                 sym, chain);
    std::exit(2);
}

// ---- the QSA / flash-attention tail ------------------------------------------------------------------------
// `native_flash_attn_short_step` (layer.cpp:995).  Reached only inside `if (native_flash_attn_short)` (:989),
// itself in the `else` of `if (g_fast_attn && !native_flash_attn_short && dump == nullptr)` (:978).
// `native_flash_attn_short` DEFAULTS false (layer.cpp:92) and is set by `--native-flash-attn-short` ONLY
// (generate.cpp:2290 `layer_set_native_flash_attn_short(o.native_flash_attn_short)`), which `--native` does NOT
// set.  With it false the :978 `if` branch runs - `qsa_decode_attn_step`, ported.
void native_flash_attn_short_step(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, int,
                                  const QsaShapes&, float*, int32_t*, const uint16_t*, void*) {
    refuse_unreachable("native_flash_attn_short_step",
                       "--native-flash-attn-short (layer.cpp:92 default false; generate.cpp:2290; NOT set by --native)");
}

// `qsa_attend_step` (layer.cpp:1002): the final `else` of the `native_flash_attn_short` test (:989), inside the
// `else` of the fast-attention test (:978).  Reached only when NOT(g_fast_attn && !native_flash_attn_short &&
// dump==nullptr) AND !native_flash_attn_short: i.e. `--no-fast-attn` (g_fast_attn default true, layer.cpp:42),
// OR `native_flash_attn_short` true, OR a non-null dump.  Shipped: fast attn on, nfas off, dump null -> the
// ported `qsa_decode_attn_step` runs.
void qsa_attend_step(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, const QsaShapes&,
                     float*, float*, void*) {
    refuse_unreachable("qsa_attend_step",
                       "--no-fast-attn (g_fast_attn default true, layer.cpp:42) OR --native-flash-attn-short OR a "
                       "non-null dump - the `else` of `if (g_fast_attn && !native_flash_attn_short && dump == nullptr)` "
                       "(layer.cpp:978/1002); the selected branch is the ported qsa_decode_attn_step");
}

// `qsa_index_step` (layer.cpp:973) and `topk_512_step` (layer.cpp:973): both in the `else` of
// `if (g_fast_select)` (:968).  `g_fast_select` DEFAULTS true (layer.cpp:42), set by
// `layer_set_fast_select(!o.no_fast_select)` (generate.cpp:2282) where `--no-fast-select` defaults false.  With
// it true the selected branch is the ported `qsa_block_scores` + `qsa_block_topk`.
void qsa_index_step(const float*, const float*, const float*, const QsaShapes&, const int32_t*, int64_t, float*,
                    void*) {
    refuse_unreachable("qsa_index_step",
                       "--no-fast-select (g_fast_select default true, layer.cpp:42; generate.cpp:2282) - the `else` "
                       "of `if (g_fast_select)` (layer.cpp:968); the selected branch is the ported qsa_block_scores");
}
void topk_512_step(const float*, const QsaShapes&, int64_t, const int32_t*, int32_t*, void*) {
    refuse_unreachable("topk_512_step",
                       "--no-fast-select (g_fast_select default true, layer.cpp:42; generate.cpp:2282) - the `else` "
                       "of `if (g_fast_select)` (layer.cpp:968); the selected branch is the ported qsa_block_topk");
}

// `native_qsa_indexer_append` (layer.cpp:945): inside `if (native_qsa_indexer_enabled())` (:944).  The backend
// ANSWERS that getter FALSE (`native_caps_vk.cpp`, with `native_qsa_indexer_set_enabled` a no-op), which selects
// the ported `indexer_key_append` (:948).  So the reaching configuration is the BACKEND ANSWERING TRUE, which it
// does not - see the capability contract.
void native_qsa_indexer_append(const float*, const int32_t*, int32_t, const float*, float,
                               const QsaIndexerBuffers&, const QsaShapes&, int64_t, const RopeScaling&, void*) {
    refuse_unreachable("native_qsa_indexer_append",
                       "native_qsa_indexer_enabled() == true (layer.cpp:944); the backend ANSWERS it false "
                       "(native_caps_vk.cpp; native_qsa_indexer_set_enabled is a no-op), selecting the ported "
                       "indexer_key_append (layer.cpp:948)");
}

// ---- the fused hyper-connection read -------------------------------------------------------------------------
// `fused_gr_read` (layer.cpp:1253/1276): inside `if (fused)` where `fused = g_fused_gr &&
// fused_gr_supported(g.n_embd, g.hc, g.hc_lr)` (layer.cpp:1188/1328).  `g_fused_gr` IS TRUE under the shipped
// `--native` launch (generate.cpp:1804 sets `gr_native_mmvf`, :2284 `layer_set_fused_gr(...)`) - the earlier
// note that "the backend forces g_fused_gr false" was FALSE of the code (there is no such call).  So the branch
// is selected by the OTHER input, `fused_gr_supported`, which the BACKEND defines: this backend has no fused_gr
// shader, so it now answers FALSE (ple_vk.cpp), keeping the ported `gr_read` on the path.  This is the
// `native_mmvq_supported` shape - "the backend reports what it implements".
void fused_gr_read(const FusedGrArgs&, void*) {
    refuse_unreachable("fused_gr_read",
                       "fused_gr_supported(n_embd,hc,hc_lr) == true (layer.cpp:1188/1328) with g_fused_gr true "
                       "(--native, generate.cpp:1804/2284); the backend ANSWERS fused_gr_supported false (no "
                       "fused_gr shader in this tree), selecting the ported gr_read (layer.cpp:1255)");
}

// ---- the KV streaming resident tier -------------------------------------------------------------------------
// `--kv-resident N` (generate.cpp:1319 -> qsa_set_kv_resident, layer.cpp:554) sets a nonzero resident-cell
// count; the residency PLAN (layer.cpp:528) then sets `p.mode` (1 = streamed, 2 = ring) ONLY when the
// resident count is positive and the ring is smaller than the page count.  `g_kv_resident` DEFAULTS 0
// (layer.cpp:515, generate.cpp:311), so `qsa_residency_plan` returns `p.mode == 0` and the state init takes the
// IDENTITY page table (layer.cpp:702-705).  `kv_mode == 0` everywhere, so all three are unreachable.
void kv_stream_reset(const KvStreamMap&, void*) {
    refuse_unreachable("kv_stream_reset",
                       "--kv-resident N>0 (g_kv_resident default 0, layer.cpp:515; generate.cpp:1319/1773) making "
                       "qsa_residency_plan set p.mode==1 (layer.cpp:538), so layer.cpp:707/:738 (kv_mode==1) run");
}
void kv_ring_table(int32_t*, int64_t, int64_t, void*) {
    refuse_unreachable("kv_ring_table",
                       "--kv-resident N>0 with a ring smaller than the page count (layer.cpp:535 sets p.mode==2); "
                       "layer.cpp:709 is the `else` of the mode test, with g_kv_resident default 0 (layer.cpp:515)");
}
void kv_stream_resolve(const KvStreamMap&, const QsaAttnPools&, const KvHostPools&, int, const int32_t*,
                       const int32_t*, int64_t, int64_t, const QsaShapes&, void*) {
    refuse_unreachable("kv_stream_resolve",
                       "--kv-resident N>0 (kv_mode==1); qsa_kv_resolve returns early `if (st.kv_mode != 1)` "
                       "(layer.cpp:756), and kv_mode is p.mode which is 0 by default (layer.cpp:515)");
}

}  // namespace strata::kernels
