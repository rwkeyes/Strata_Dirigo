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
#include "strata/kernels/fused_gr.hpp"          // fused_gr_read (FusedGrArgs) + fused_gr_read_multi / fused_gr_check
#include "strata/kernels/kv_stream.hpp"         // kv_ring_table / kv_stream_reset / kv_stream_resolve / kv_stream_counters
#include "strata/kernels/native_flash_attn.hpp" // native_flash_attn_short_step
#include "strata/kernels/native_qsa_indexer.hpp"// native_qsa_indexer_append (QsaIndexerBuffers, RopeScaling)
#include "strata/kernels/qsa.hpp"               // qsa_attend_step / qsa_index_step / topk_512_step
// The engine-executable block's headers (see the section at the end of this file).
#include "strata/kernels/verify_kernels.hpp"    // the P6 verifier + drafter kernels
#include "strata/kernels/s2_expert_grouped.hpp" // moe_grouped_s2 / moe_group_resident / moe_hit_grouped_s2_cpu_order
#include "strata/kernels/sampler.hpp"           // coupled_draft_sample / _stage / _scratch_bytes (SamplerParams)
#include "strata/kernels/qsa_decode_attn.hpp"   // qsa_decode_attn_batch
#include "strata/kernels/iq_kernels.hpp"        // native_expert_grouped / _layout / _scratch_bytes / _supported
#include "strata/kernels/native_moe.hpp"        // native_moe_combine_multi
#include "strata/kernels/native_router.hpp"     // native_router_top10_multi
#include "strata/kernels/ple.hpp"               // ple_block_projected (PleWeights/PleOut)
#include "strata/kernels/shared_expert.hpp"     // shared_expert_multi (NativeSharedWeights)
#include "strata/kernels/elementwise.hpp"       // copy_rows_from_mapped
#include "strata/kernels/cpu/kq_avx1.hpp"       // cpu::bf16_rows_dot_multi_avx1
#include "strata/kernels/cpu/kq_avx2.hpp"       // cpu::bf16_rows_dot_multi

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

// ============================================================================================================
// THE ENGINE-EXECUTABLE REFUSALS - the ~40 kernels-namespace symbols the `strata` PROGRAM still wants at LINK
// time, every one of them OFF a single-token decode.
// ============================================================================================================
// This block exists because the `strata` executable (generate.cpp) whole-archives more of the engine than the
// one-layer-body link did, so its undefined list is a SUPERSET: the 9 verifier kernels + 24 more, plus the
// `kernels_cpu` half and the multi-GPU `native_expert_*` tier.  Each is answered with the ONE body this
// backend has for it - a LOUD REFUSAL naming the flag chain that reaches it - so the program LINKS and refuses
// loudly when a non-selected path is walked, rather than silently degrading.  The triage classes are in
// `plan/DECODE-PATH-TRIAGE.md` ("THE ORDERED DECODE-PATH LIST"): NONE of the symbols below is on a plain
// single-token decode under the shipped `--native` launch (the two that ARE - `sample_tokens` and
// `copy_from_mapped` - each have a real definition and a gate case).  A `native_expert_supported` /
// `embed_type_supported` CAPABILITY is ANSWERED rather than refused (the backend reports what it implements);
// those live in `native_caps_vk.cpp` / `iq_vk.cpp` respectively.  The map's vocabulary stays honest: these
// stay `todo` in PORT-MAP.tsv, and a `todo` row is explicitly NOT "the backend defines and works".

// ---- CLASS C: the speculative DRAFTER (src/core/mtp.cpp), a config the port does not select ------------------
// `setup.py` writes `--spec 4 --mtp`, but the draft loop needs `Verifier::init` to succeed and
// `layer_verify_compatible()` (layer.cpp:476-491) demands a conjunction the contract leaves false.  The port's
// selected branch is a `--spec 0` run.
size_t coupled_draft_scratch_bytes(int) {
    refuse_unreachable("coupled_draft_scratch_bytes", "--spec 4 --mtp (mtp.cpp), a drafter config the contract refuses at Verifier::init");
}
void coupled_draft_stage(const SamplerParams*, const int32_t*, SamplerParams*, int32_t*, int, void*) {
    refuse_unreachable("coupled_draft_stage", "--spec 4 --mtp (mtp.cpp:699), the coupled draft round's mapped staging");
}
void coupled_draft_sample(float*, int, const int32_t*, const int32_t*, int, const SamplerParams*, int32_t*, int, int,
                          const int32_t*, void*, int32_t*, float*, void*) {
    refuse_unreachable("coupled_draft_sample", "--spec 4 --mtp (mtp.cpp:635), the coupled drafter's sampler");
}
void add_streams_broadcast(const float*, const float*, float*, int64_t, int, int, void*) {
    refuse_unreachable("add_streams_broadcast", "--spec 4 --mtp (mtp.cpp:497), the drafter's embedding branch");
}
void fused_gr_read_multi(const FusedGrArgs*, int, float*, void*, unsigned long long*, int) {
    refuse_unreachable("fused_gr_read_multi", "--spec 4 --mtp (mtp.cpp:510/571/620) AND the backend answering fused_gr_supported TRUE");
}
void moe_grouped_s2(const unsigned long long*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t,
                    int64_t, const uint8_t*, const float*, void*, float*, void*) {
    refuse_unreachable("moe_grouped_s2", "--spec 4 --mtp (mtp.cpp:582) OR --expert-cache-remote N / --peer-device (remote_experts.cpp:314)");
}
void moe_group_resident(const int32_t*, int, int, const uint8_t*, int64_t, unsigned long long*, int32_t*, int32_t*,
                        int32_t*, int32_t*, void*) {
    refuse_unreachable("moe_group_resident", "--spec 4 --mtp (mtp.cpp:579), the resident-plan group build");
}
void row_top_prob(const float*, int, int, const int32_t*, float*, void*) {
    refuse_unreachable("row_top_prob", "--spec 4 --mtp (mtp.cpp:643), the drafter's probability readout");
}
void map_ids(int32_t*, const int32_t*, int, void*) {
    refuse_unreachable("map_ids", "--spec 4 --mtp (mtp.cpp:644), the subset-index map");
}
void window_ids(int32_t*, int, int, int32_t*, int64_t, void*) {
    refuse_unreachable("window_ids", "--spec 4 --mtp (mtp.cpp:551), the drafter's sliding window");
}
void mtp_select(const float*, int64_t, const int32_t*, const int32_t*, float*, int32_t*, int32_t*, int, void*,
                const float*, float*) {
    refuse_unreachable("mtp_select", "--spec 4 --mtp (mtp.cpp:710/719/738), the draft chain's next input");
}
void kv_ring_restore(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t, int64_t, const QsaShapes&, void*) {
    refuse_unreachable("kv_ring_restore", "--spec 4 --mtp (mtp.cpp:748) AND --kv-resident N>0 (kv_mode==2)");
}
void embedding_gather_dev(const uint8_t*, const float*, const float*, const int32_t*, int, int64_t, int, int, int,
                          uint64_t, uint64_t, float*, void*) {
    refuse_unreachable("embedding_gather_dev", "--spec 4 --mtp (mtp.cpp:485), the drafter's device-id gather");
}
void qsa_decode_attn_batch(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                           float*, float*, int64_t, void*) {
    refuse_unreachable("qsa_decode_attn_batch", "--spec 4 --mtp (mtp.cpp:552) OR the P6 verifier (verify.cpp), neither selected");
}
KvStreamCounters kv_stream_counters(const KvStreamMap&) {
    refuse_unreachable("kv_stream_counters", "--kv-resident N>0 (g_kv_resident default 0, layer.cpp:515)");
}

// ---- CLASS C: the A/B arm defaulting OFF --------------------------------------------------------------------
// `moe_hit_grouped_s2_cpu_order` (expert_source.cpp:2329): default FALSE (`generate.cpp:376`), set only by
// `--expert-cache-cpu-order` (`:1477`).  The DEFAULT is the PORTED `moe_hit_grouped_s2`.
void moe_hit_grouped_s2_cpu_order(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*,
                                  void*, float*, void*, const float*, float*) {
    refuse_unreachable("moe_hit_grouped_s2_cpu_order", "--expert-cache-cpu-order (d.hit_cpu_order default false, generate.cpp:376/1477); the default is the PORTED moe_hit_grouped_s2");
}

// ---- CLASS D: the P6 VERIFIER (src/core/verify.cpp), which cannot init under the contract ---------------------
// Same conjunction as the drafter (`layer_verify_compatible`, layer.cpp:476-491).  `verify.cpp` is not on the
// decode path.
void broadcast_streams(const float*, float*, int64_t, int, int, void*) {
    refuse_unreachable("broadcast_streams", "the P6 verifier (verify.cpp); Verifier::init refuses (layer.cpp:476-491)");
}
void copy_i32_from_mapped_unless(int32_t*, const int32_t*, long long, const uint32_t*, uint32_t, void*) {
    refuse_unreachable("copy_i32_from_mapped_unless", "the P6 verifier (verify.cpp:624); Verifier::init refuses");
}
void copy_indexed(float*, const float*, int64_t, const int32_t*, int64_t, void*) {
    refuse_unreachable("copy_indexed", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void copy_or_zero_from_mapped(float*, const float*, long long, const uint32_t*, uint32_t, void*) {
    refuse_unreachable("copy_or_zero_from_mapped", "the P6 verifier (verify.cpp:630) OR the remote-expert opt (remote_expert_opt.cu:124)");
}
void copy_rows_from_mapped(float*, const float*, int64_t, int64_t, const int32_t*, const int32_t*, void*) {
    refuse_unreachable("copy_rows_from_mapped", "the P6 verifier (verify.cpp) and the device-plan expert source (expert_source.cpp:2131); not the decode path");
}
void fetch_blobs(const unsigned long long*, const int32_t*, uint8_t*, int64_t, int, void*) {
    refuse_unreachable("fetch_blobs", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void fused_gr_check() {
    refuse_unreachable("fused_gr_check", "the P6 verifier (verify.cpp:311); Verifier::init refuses");
}
void gdn_ab_multi(const float*, const uint16_t*, const uint16_t*, const float*, const float*, float*, float*, int, int,
                  int, void*) {
    refuse_unreachable("gdn_ab_multi", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void gdn_conv_commit(float*, const float*, int, const int32_t*, void*) {
    refuse_unreachable("gdn_conv_commit", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void gdn_conv_l2_multi(const float*, const float*, const float*, float*, int, int, float, int, void*, int) {
    refuse_unreachable("gdn_conv_l2_multi", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void gdn_step_norm_multi(float*, const float*, int, const float*, const float*, const float*, const float*, float,
                         float*, int, int, int, const int32_t*, void*, int) {
    refuse_unreachable("gdn_step_norm_multi", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void gpu_stamp(unsigned long long*, int, void*) {
    refuse_unreachable("gpu_stamp", "the P6 verifier (verify.cpp), its stage profiler; Verifier::init refuses");
}
void native_moe_combine_multi(const float*, const float*, const float*, float*, int64_t, int64_t, int, void*) {
    refuse_unreachable("native_moe_combine_multi", "the P6 verifier (verify.cpp:1081); Verifier::init refuses");
}
void native_router_top10_multi(const float*, int32_t*, float*, int, void*) {
    refuse_unreachable("native_router_top10_multi", "the P6 verifier (verify.cpp:916); Verifier::init refuses");
}
// `native_expert_grouped` is DEFINED (not refused) in vulkan/src/kernels/native_expert_grouped_vk.cpp: the grouped
// IQ-expert launcher, over the port's byte-offset group table.  See that file's header note.
NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    // TRANSCRIBED from the reference `src/kernels/cuda/iq_kernels.cu:1869` - this is host arithmetic over the
    // SAME row table the port's `*_mmvq` shaders take their stride from (`iq_row_bytes`), so a blob's internal
    // offsets here cannot disagree with the kernels that read it.  It is a layout description, not a dispatch:
    // nothing about the LOGIC changed from the refusal it replaces, only that the answer is now the real one.
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = strata::kernels::iq_row_bytes(gu_type, n_embd);
    L.d_row = strata::kernels::iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}
size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    // TRANSCRIBED from `src/kernels/cuda/iq_kernels.cu:1879`: three fp32 buffers (gate, up, swiglu h) plus the
    // q8_1 image of h, each 256-byte aligned.  `block_q8_1` is 36 B (a 4-B fp16 d+s union and 32 int8 qs).
    if (cap <= 0 || n_ff <= 0) return 0;
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) +
           (((size_t) cap * (size_t) (n_ff / 32) * 36u + 255) & ~(size_t) 255);
}
void ple_block_projected(const float*, const float*, const float*, const float*, const PleWeights&, PleOut&, void*, void*) {
    refuse_unreachable("ple_block_projected", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void rebase_ptrs(unsigned long long*, const int32_t*, uint8_t*, int64_t, void*) {
    refuse_unreachable("rebase_ptrs", "the P6 verifier (verify.cpp); Verifier::init refuses");
}
void resident_plan(const int32_t*, int, int, const int32_t*, int, const uint8_t*, const unsigned long long*, long long,
                   int32_t*, long long, uint32_t*, uint32_t, void*) {
    refuse_unreachable("resident_plan", "the P6 verifier (verify.cpp) - the all-resident device plan; Verifier::init refuses");
}
void shared_expert_multi(int, const float*, const uint16_t*, const NativeSharedWeights&, const uint16_t*, float*,
                         float*, float*, float*, int64_t, int64_t, void*) {
    refuse_unreachable("shared_expert_multi", "the P6 verifier (verify.cpp:980); Verifier::init refuses");
}
void wait_flag_ge(const uint32_t*, uint32_t, void*) {
    refuse_unreachable("wait_flag_ge", "the P6 verifier's split window (verify.cpp); a translating spin is forbidden here and the verifier cannot init");
}
void wait_flag_ge_or(const uint32_t*, uint32_t, const uint32_t*, void*) {
    refuse_unreachable("wait_flag_ge_or", "the P6 verifier's split window (verify.cpp); see wait_flag_ge");
}

// ---- the `kernels_cpu` half, deliberately OUT of the Vulkan build --------------------------------------------
// The name-only pattern `strata::kernels::cpu::bf16_rows_dot_multi{,_avx1}` (kq_avx2.cpp / kq_avx1.cpp, EXCLUDED
// from `strata_vulkan_kernels_cpu`, vulkan/CMakeLists.txt:145) is the routing-aware PREFETCH `RouterLookahead::run`
// (expert_source.cpp:1228/1231), started only on the FILE tier with `STRATA_LOOKAHEAD != 0`
// (generate.cpp:3676) - off on the packed-image launch, and "an estimate only" where it does run.
namespace cpu {
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) {
    refuse_unreachable("cpu::bf16_rows_dot_multi", "the FILE-tier RouterLookahead prefetch (expert_source.cpp:1228/1231, STRATA_LOOKAHEAD!=0, generate.cpp:3676); off on the packed-image launch");
}
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    refuse_unreachable("cpu::bf16_rows_dot_multi_avx1", "the AVX-only arm of the same FILE-tier prefetch (expert_source.cpp:1231)");
}
}  // namespace cpu

// ---- A CAPABILITY, ANSWERED NOT REFUSED: `native_expert_supported` ------------------------------------------
// `bool ... noexcept` (iq_kernels.hpp:51).  The engine's load path (generate.cpp:2007) asks it, BEFORE anything
// is allocated, whether the GROUPED native-expert kernel can compute a given (gu_type, d_type) at this geometry;
// a false answer refuses the pack by layer.  The reference (`src/kernels/cuda/iq_kernels.cu:1863`) answers from
// its own kernel inventory and four geometry constraints.  THIS backend answers from ITS OWN inventory, and the
// two grouped shaders cover the real pack's four gate/up formats and two down formats:
//
//   * `native_gu_any.spv`   - grouped gate/up, ggml {18 IQ3_XXS, 21 IQ3_S, 22 IQ2_S, 23 IQ4_XS} (one generic shader)
//   * `native_down_any.spv` - grouped down,    ggml {20 IQ4_NL, 42 Q2_0} (one generic shader)
//   * AND THE LAUNCHER `native_expert_grouped` IS WIRED (native_expert_grouped_vk.cpp).
//
// `coder-iq1_m`'s seven (gu,d) pairs are all inside that product, so this answers TRUE for every one of its 48
// layers - and FALSE for a type outside the two shaders, which is the capability telling the engine which branch
// its own code may take (a Vulkan backend has no `--native` dispatch table to fall back on).
static bool grouped_gu_shader(int gu_type) noexcept {
    return gu_type == 18 || gu_type == 21 || gu_type == 22 || gu_type == 23;   // native_gu_any.spv
}
static bool grouped_down_shader(int d_type) noexcept {
    return d_type == 20 || d_type == 42;                                        // native_down_any.spv
}
static constexpr bool kGroupedLauncher = true;   // native_expert_grouped_vk.cpp

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    if (!kGroupedLauncher) return false;
    if (!grouped_gu_shader(gu_type) || !grouped_down_shader(d_type)) return false;
    // The port's shader geometry: gate/up is a 256-value block format, down is 32-value (IQ4_NL) or 64-value
    // (Q2_0); the activation is q8_1, so (n_ff * n_embd) % 256 == 0 as the reference also requires.
    if (n_embd % 256 != 0) return false;
    if (d_type == 20 && n_ff % 32 != 0) return false;
    if (d_type == 42 && n_ff % 64 != 0) return false;
    if ((n_ff * n_embd) % 256 != 0) return false;
    return true;
}

// Exposed so the gate can pin the inventory itself (a two-sided check: the types with a built shader, and a type
// without one) rather than only the composed answer, which today is false whatever the inventory says.
bool native_expert_grouped_shaders(int gu_type, int d_type) noexcept {
    return grouped_gu_shader(gu_type) && grouped_down_shader(d_type);
}
bool native_expert_grouped_launcher() noexcept { return kGroupedLauncher; }

}  // namespace strata::kernels
