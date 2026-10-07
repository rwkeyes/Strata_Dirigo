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

// ONE refusal body: it names the symbol, the configuration that reaches it, and REFUSES.  `std::exit(2)` (not
// an exception): a reach here is a configuration the backend does not implement, and the port's rule is to
// refuse loudly rather than degrade.
//
// **THE TEXT CHANGED IN THIS BATCH, AND THE CHANGE IS THE POINT.**  It used to read "NOT PORTED and NOT REACHED
// by the shipped configuration", and that claim was FALSE OF THE CODE: `Verifier::init` now SUCCEEDS
// (`verify.cpp:336`'s `fused_gr_supported` disjunct is true because the fused read is ported), so the P6
// verify window's body EXECUTES and reaches its symbols in turn - and the window is a native pack's ONLY decode
// path (`generate.cpp:7578-7579`).  **A refusal that says "not reached" while being reached is a defect in the
// INSTRUMENT.**  So the body now states the symbol and the chain, and the CHAIN STRING says which it is:
// `REACHED BY THE SHIPPED CONFIGURATION (<site>)` or `NOT reached (deciding condition: ...)`.  No row here is a
// capability; every one is a HOLE.
[[noreturn]] static void refuse_not_ported(const char* sym, const char* chain) {
    std::fprintf(stderr,
                 "strata::kernels::%s: NOT PORTED on the Vulkan backend - REFUSING.\n"
                 "  Reached by: %s\n"
                 "  This definition exists so the engine LINKS; it is a LOUD REFUSAL, never a silent\n"
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
    refuse_not_ported("native_flash_attn_short_step",
                       "--native-flash-attn-short (layer.cpp:92 default false; generate.cpp:2290; NOT set by --native)");
}

// `qsa_attend_step` (layer.cpp:1002): the final `else` of the `native_flash_attn_short` test (:989), inside the
// `else` of the fast-attention test (:978).  Reached only when NOT(g_fast_attn && !native_flash_attn_short &&
// dump==nullptr) AND !native_flash_attn_short: i.e. `--no-fast-attn` (g_fast_attn default true, layer.cpp:42),
// OR `native_flash_attn_short` true, OR a non-null dump.  Shipped: fast attn on, nfas off, dump null -> the
// ported `qsa_decode_attn_step` runs.
void qsa_attend_step(const float*, const uint16_t*, const uint16_t*, const int32_t*, int64_t, const QsaShapes&,
                     float*, float*, void*) {
    refuse_not_ported("qsa_attend_step",
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
    refuse_not_ported("qsa_index_step",
                       "--no-fast-select (g_fast_select default true, layer.cpp:42; generate.cpp:2282) - the `else` "
                       "of `if (g_fast_select)` (layer.cpp:968); the selected branch is the ported qsa_block_scores");
}
void topk_512_step(const float*, const QsaShapes&, int64_t, const int32_t*, int32_t*, void*) {
    refuse_not_ported("topk_512_step",
                       "--no-fast-select (g_fast_select default true, layer.cpp:42; generate.cpp:2282) - the `else` "
                       "of `if (g_fast_select)` (layer.cpp:968); the selected branch is the ported qsa_block_topk");
}

// `native_qsa_indexer_append` (layer.cpp:945) is now PORTED (vulkan/src/kernels/qsa_vk.cpp, shader
// pf_indexer_native.spv) and the backend ANSWERS `native_qsa_indexer_enabled()` TRUE (native_caps_vk.cpp), so
// that branch is the shipped one.  Nothing is refusen here for it any more.

// ---- the fused hyper-connection read -------------------------------------------------------------------------
// **BOTH SYMBOLS ARE NOW DEFINED (this batch), NOT REFUSED.**  `fused_gr_read` / `fused_gr_read_multi` live in
// `vulkan/src/kernels/fused_gr_vk.cpp` over four new shaders (`fused_gr_rs/down/mix/inject`), and the backend
// now ANSWERS `fused_gr_supported` with the engine's own geometry predicate (ple_vk.cpp) - so the branch
// `fused = g_fused_gr && fused_gr_supported(...)` (layer.cpp:1188) IS the selected one, on the decode path AND
// in the P6 verify window (verify.cpp:693/:1142), which is a native pack's only decode path.  The two rows move
// `refused -> kernel` in PORT-MAP.tsv.  The refusals that used to sit here said the branch was unreachable
// because the backend answered FALSE; that was the correct answer while the kernels were missing, and it is the
// wrong answer now that they exist.  Nothing is refused here any more.

// ---- the KV streaming resident tier -------------------------------------------------------------------------
// `--kv-resident N` (generate.cpp:1319 -> qsa_set_kv_resident, layer.cpp:554) sets a nonzero resident-cell
// count; the residency PLAN (layer.cpp:528) then sets `p.mode` (1 = streamed, 2 = ring) ONLY when the
// resident count is positive and the ring is smaller than the page count.  `g_kv_resident` DEFAULTS 0
// (layer.cpp:515, generate.cpp:311), so `qsa_residency_plan` returns `p.mode == 0` and the state init takes the
// IDENTITY page table (layer.cpp:702-705).  `kv_mode == 0` everywhere, so all three are unreachable.
void kv_stream_reset(const KvStreamMap&, void*) {
    refuse_not_ported("kv_stream_reset",
                       "--kv-resident N>0 (g_kv_resident default 0, layer.cpp:515; generate.cpp:1319/1773) making "
                       "qsa_residency_plan set p.mode==1 (layer.cpp:538), so layer.cpp:707/:738 (kv_mode==1) run");
}
void kv_ring_table(int32_t*, int64_t, int64_t, void*) {
    refuse_not_ported("kv_ring_table",
                       "--kv-resident N>0 with a ring smaller than the page count (layer.cpp:535 sets p.mode==2); "
                       "layer.cpp:709 is the `else` of the mode test, with g_kv_resident default 0 (layer.cpp:515)");
}
void kv_stream_resolve(const KvStreamMap&, const QsaAttnPools&, const KvHostPools&, int, const int32_t*,
                       const int32_t*, int64_t, int64_t, const QsaShapes&, void*) {
    refuse_not_ported("kv_stream_resolve",
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
// selected branch is a `--spec 2 --prefill 256` run (step4/run.sh), whose speculator is the engine's P6 verifier,
// NOT the `--mtp` drafter this class names: a native pack REFUSES `--spec 0`/`--spec 1` (generate.cpp:2167,
// "it needs --spec T (T >= 2)"), so `--spec 0` is not a runnable branch at all.
size_t coupled_draft_scratch_bytes(int) {
    refuse_not_ported("coupled_draft_scratch_bytes", "--spec 4 --mtp (mtp.cpp), a drafter config the contract refuses at Verifier::init");
}
void coupled_draft_stage(const SamplerParams*, const int32_t*, SamplerParams*, int32_t*, int, void*) {
    refuse_not_ported("coupled_draft_stage", "--spec 4 --mtp (mtp.cpp:699), the coupled draft round's mapped staging");
}
void coupled_draft_sample(float*, int, const int32_t*, const int32_t*, int, const SamplerParams*, int32_t*, int, int,
                          const int32_t*, void*, int32_t*, float*, void*) {
    refuse_not_ported("coupled_draft_sample", "--spec 4 --mtp (mtp.cpp:635), the coupled drafter's sampler");
}
// `add_streams_broadcast` is now PORTED (`vulkan/src/kernels/verify_vk.cpp`, the same `bcast_streams.spv` as
// `broadcast_streams` with `mode = 1`).  It is the MTP DRAFTER's embedding branch (mtp.cpp:497); the port's
// shipped launch does not pass `--mtp`, but the symbol is real now, so a reader comparing the two names in the
// map finds both.
void moe_grouped_s2(const unsigned long long*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int64_t,
                    int64_t, const uint8_t*, const float*, void*, float*, void*) {
    refuse_not_ported("moe_grouped_s2", "--spec 4 --mtp (mtp.cpp:582) OR --expert-cache-remote N / --peer-device (remote_experts.cpp:314)");
}
void moe_group_resident(const int32_t*, int, int, const uint8_t*, int64_t, unsigned long long*, int32_t*, int32_t*,
                        int32_t*, int32_t*, void*) {
    refuse_not_ported("moe_group_resident", "--spec 4 --mtp (mtp.cpp:579), the resident-plan group build");
}
void row_top_prob(const float*, int, int, const int32_t*, float*, void*) {
    refuse_not_ported("row_top_prob", "--spec 4 --mtp (mtp.cpp:643), the drafter's probability readout");
}
void map_ids(int32_t*, const int32_t*, int, void*) {
    refuse_not_ported("map_ids", "--spec 4 --mtp (mtp.cpp:644), the subset-index map");
}
void window_ids(int32_t*, int, int, int32_t*, int64_t, void*) {
    refuse_not_ported("window_ids", "--spec 4 --mtp (mtp.cpp:551), the drafter's sliding window");
}
void mtp_select(const float*, int64_t, const int32_t*, const int32_t*, float*, int32_t*, int32_t*, int, void*,
                const float*, float*) {
    refuse_not_ported("mtp_select", "--spec 4 --mtp (mtp.cpp:710/719/738), the draft chain's next input");
}
void kv_ring_restore(const QsaAttnPools&, const KvHostPools&, int, int64_t, int64_t, int64_t, const QsaShapes&, void*) {
    refuse_not_ported("kv_ring_restore", "--spec 4 --mtp (mtp.cpp:748) AND --kv-resident N>0 (kv_mode==2)");
}
void embedding_gather_dev(const uint8_t*, const float*, const float*, const int32_t*, int, int64_t, int, int, int,
                          uint64_t, uint64_t, float*, void*) {
    refuse_not_ported("embedding_gather_dev", "--spec 4 --mtp (mtp.cpp:485), the drafter's device-id gather");
}
// `qsa_decode_attn_batch` (qsa_decode_attn.hpp) is now PORTED (qsa_vk.cpp: a per-query LOOP over the gated
// `qsa_decode_attn_step`), so the prompt fallback, the P6 verifier and the MTP drafter all reach a real body.

KvStreamCounters kv_stream_counters(const KvStreamMap&) {
    refuse_not_ported("kv_stream_counters", "--kv-resident N>0 (g_kv_resident default 0, layer.cpp:515)");
}

// ---- CLASS C: the A/B arm defaulting OFF --------------------------------------------------------------------
// `moe_hit_grouped_s2_cpu_order` (expert_source.cpp:2329): default FALSE (`generate.cpp:376`), set only by
// `--expert-cache-cpu-order` (`:1477`).  The DEFAULT is the PORTED `moe_hit_grouped_s2`.
void moe_hit_grouped_s2_cpu_order(const uint8_t*, const int32_t*, const int32_t*, int64_t, int64_t, const uint8_t*,
                                  void*, float*, void*, const float*, float*) {
    refuse_not_ported("moe_hit_grouped_s2_cpu_order", "--expert-cache-cpu-order (d.hit_cpu_order default false, generate.cpp:376/1477); the default is the PORTED moe_hit_grouped_s2");
}

// ---- THE P6 VERIFIER (src/core/verify.cpp) - THE WINDOW NOW RUNS, AND THESE ROWS MOVED WITH IT --------------
// **THE HEADING THAT USED TO SIT HERE SAID "which cannot init under the contract", AND THAT WAS FALSE OF THE
// CODE.**  `Verifier::init` SUCCEEDS (`verify.cpp:336`; the fused read is ported and `fused_gr_supported`
// answers the engine's own geometry predicate), so the window's body EXECUTES and calls each of its symbols in
// turn.  This batch PORTED nine of them (`vulkan/src/kernels/verify_vk.cpp`): `broadcast_streams`,
// `add_streams_broadcast`, `gdn_conv_l2_multi`, `gdn_ab_multi`, `gdn_step_norm_multi`, `gdn_conv_commit`,
// `native_router_top10_multi`, `native_moe_combine_multi`, `shared_expert_multi`.  What is left below is the
// rest of the window, and EACH CHAIN STRING NOW SAYS WHETHER THE SHIPPED CONFIGURATION REACHES IT - the old
// blanket "NOT REACHED" is gone (see the `refuse_not_ported` note above).
//
// GENUINELY NOT REACHED, with the deciding condition (still a loud refusal, still a hole):
//   * `copy_i32_from_mapped_unless`, `copy_or_zero_from_mapped`, `wait_flag_ge_or`
//     (`verify.cpp:1039-1063`) - inside `if (device_plan_)`, and `device_plan_` needs
//     `STRATA_VERIFY_DEVICE_PLAN` (verify.cpp:512-515).
//   * `fetch_blobs`/`rebase_ptrs` (`verify.cpp:1053/:1054`) - inside `if (sink_.pcie_mode == 2)`, and the PCIe
//     probe on this box reads 0.1 GB/s -> `pcie_frac 0.00`.
//   * `gpu_stamp` (`verify.cpp:564/:565`) - guarded by `prof_on_` / `trace_m_`, i.e. `STRATA_VERIFY_PROFILE` /
//     `STRATA_VERIFY_TRACE`, both unset.
//   * `ple_block_projected` (`verify.cpp:668`) - guarded by `ple_batch_kv`, which needs
//     `ple_native_bf16_enabled()` AND `ple_native_postops_enabled()`; the backend answers both FALSE.
//
// REACHED BY THE SHIPPED CONFIGURATION (and therefore a HOLE, not a note):
//   * `wait_flag_ge` (`verify.cpp:1042/:1049/:1066`) - the `else` of `if (all_resident_)` in `post`, the FIRST
//     symbol the window reaches once `pre` completes.  A translating spin is forbidden here (the port's
//     no-spinning rule), so the window's next stop is this symbol; its chain string says so.
//   * `resident_plan` (`verify.cpp:938`, the `if (all_resident_)` branch of the window's per-group plan) -
//     REACHED ONCE THE ALL-RESIDENT FIT CLOSES, and MEASURED REACHED on `vega` 2026-10-06 with
//     `--expert-cache 12288`.  **NOW PORTED** (`vulkan/src/kernels/verify_vk.cpp`, shader `resident_plan.spv`):
//     it was the LAST unported symbol between the all-resident window and the launch.  The refusal here was the
//     corrected text the 2026-10-06 batch shipped (the old text wrongly claimed `all_resident_` was
//     unreachable); it is gone because the symbol is now a real body.
//   * `copy_rows_from_mapped` (`verify.cpp:1071`) - the `dec_batch` arm of the CPU-share copy, with `dec_batch`
//     TRUE by default (`STRATA_DEC_BATCH` unset).
//   * `copy_indexed` (`verify.cpp:1311`, the commit graph) - reached whenever the PLE stage is ready, which it
//     is for a native pack (the PLE key is native too).
void copy_i32_from_mapped_unless(int32_t*, const int32_t*, long long, const uint32_t*, uint32_t, void*) {
    refuse_not_ported("copy_i32_from_mapped_unless",
                      "NOT reached - the P6 verifier's DEVICE-PLAN arm (verify.cpp:1040, `if (device_plan_)`; "
                      "STRATA_VERIFY_DEVICE_PLAN unset, verify.cpp:512-515)");
}
void copy_or_zero_from_mapped(float*, const float*, long long, const uint32_t*, uint32_t, void*) {
    refuse_not_ported("copy_or_zero_from_mapped",
                      "NOT reached - the P6 verifier's DEVICE-PLAN arm (verify.cpp:1063, `if (device_plan_)`) OR "
                      "the remote-expert opt (remote_expert_opt.cu:124)");
}
// `copy_rows_from_mapped` is NOW DEFINED (not refused) in `vulkan/src/kernels/elementwise_vk.cpp`
// (shader `copy_rows_from_mapped.spv`): the P6 verify window's `dec_batch` CPU-share copy (verify.cpp:1071),
// reached after the three handshake waits.  Its row was previously `refused`; the body is a real one, so the row
// moves `refused -> kernel` in PORT-MAP.tsv.
// `copy_indexed` is NOW DEFINED in `vulkan/src/kernels/verify_vk.cpp` (shader `copy_indexed.spv`): the commit
// graph's PLE-history copy (verify.cpp:1311/:1867), reached whenever the PLE stage is ready.  Same row move.
// `fetch_blobs` is NOW DEFINED (not refused) in `vulkan/src/kernels/verify_vk.cpp`: the P6 verify window's PCIe
// staging (`verify.cpp:1053`), ON PATH under the default `--pcie-mode auto` -> `set_pcie_mode(2)`.  It is a
// DEVICE-side gather over the pointer table (the `ptr_to_off.spv` technique: lo/hi uint32 words, the arena base
// subtracted in 64-bit arithmetic, the source bound as a 4 GiB arena window) so `*n` is re-read at every replay;
// at `pcie_frac 0.00` `*n == 0` for every group, which is the CUDA's own empty no-op.  The refusal that used to
// sit here claimed the symbol was NOT reached - FALSE of the code, as the run showed.
// `fused_gr_check` is DEFINED (not refused) in `vulkan/src/kernels/ple_vk.cpp`: it is a card CHARACTERISATION
// (no tensors), and on this backend its honest outcome is "the plain read runs here".  It was a refusal here
// until the shipped `--native` launch was measured to REACH it at `Verifier::init` (verify.cpp:311) - the
// "NOT REACHED by the shipped configuration" claim was false.  See that definition and DECODE-PATH-TRIAGE.md.
//
// `gdn_ab_multi`, `gdn_conv_commit`, `gdn_conv_l2_multi`, `gdn_step_norm_multi`, `native_moe_combine_multi`,
// `native_router_top10_multi` and `shared_expert_multi` were refusals here; **all seven are PORTED**
// (`vulkan/src/kernels/verify_vk.cpp`) and their rows moved `refused -> kernel`.  The two loops and the two
// shaders carry the per-token contract in their own header notes.
void gpu_stamp(unsigned long long*, int, void*) {
    refuse_not_ported("gpu_stamp",
                      "NOT reached - the P6 verifier's stage profiler (verify.cpp:564/:565), guarded by "
                      "`prof_on_` / `trace_m_` (`STRATA_VERIFY_PROFILE` / `STRATA_VERIFY_TRACE`, both unset)");
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
    refuse_not_ported("ple_block_projected",
                      "NOT reached - the PLE batch arm (verify.cpp:668), guarded by `ple_batch_kv` which needs "
                      "`ple_native_bf16_enabled()` AND `ple_native_postops_enabled()`; the backend answers both "
                      "FALSE (vulkan/src/kernels/ple_vk.cpp)");
}
// `rebase_ptrs` is NOW DEFINED (not refused) in `vulkan/src/kernels/verify_vk.cpp` (shader `rebase_ptrs.spv`):
// `ptr[k] = base + k*blob_bytes` for `k < *n`, the lo/hi write the CUDA does (`verify_kernels.cu:281`).  At
// `pcie_frac 0.00` `*n == 0`, so it writes nothing (the rebase is the IDENTITY); the shader carries the general
// case, not a host-side shortcut.
// `resident_plan` is NOW DEFINED (not refused) in `vulkan/src/kernels/verify_vk.cpp` (shader
// `resident_plan.spv`): the P6 verify window's per-group plan on the ALL-RESIDENT arm (verify.cpp:938), the
// group-by over the routed ids producing `ptr[grp] = cache_base + slot_off[slot]` and the per-entry
// (thread,token) map.  Its refusal here was the LAST symbol between the all-resident window and the launch;
// the row moves `refused -> kernel` in PORT-MAP.tsv.  The `else if (device_plan_)` arm (verify.cpp:943) passes
// `skip + grp` and a ring, which this definition also carries (the skip word is written when given).
void wait_flag_ge_or(const uint32_t*, uint32_t, const uint32_t*, void*) {
    refuse_not_ported("wait_flag_ge_or",
                      "NOT reached - the P6 verifier's DEVICE-PLAN arms (verify.cpp:1039/:1048/:1062, "
                      "`if (device_plan_)`); see wait_flag_ge for the reached sibling");
}

// ---- the `kernels_cpu` half, deliberately OUT of the Vulkan build --------------------------------------------
// The name-only pattern `strata::kernels::cpu::bf16_rows_dot_multi{,_avx1}` (kq_avx2.cpp / kq_avx1.cpp, EXCLUDED
// from `strata_vulkan_kernels_cpu`, vulkan/CMakeLists.txt:145) is the routing-aware PREFETCH `RouterLookahead::run`
// (expert_source.cpp:1228/1231), started only on the FILE tier with `STRATA_LOOKAHEAD != 0`
// (generate.cpp:3676) - off on the packed-image launch, and "an estimate only" where it does run.
namespace cpu {
void bf16_rows_dot_multi(const uint16_t*, int, int, const float*, int, float*) {
    refuse_not_ported("cpu::bf16_rows_dot_multi", "the FILE-tier RouterLookahead prefetch (expert_source.cpp:1228/1231, STRATA_LOOKAHEAD!=0, generate.cpp:3676); off on the packed-image launch");
}
void bf16_rows_dot_multi_avx1(const uint16_t*, int, int, const float*, int, float*) {
    refuse_not_ported("cpu::bf16_rows_dot_multi_avx1", "the AVX-only arm of the same FILE-tier prefetch (expert_source.cpp:1231)");
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
