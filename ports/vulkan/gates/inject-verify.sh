#!/usr/bin/env bash
# ports/vulkan/gates/inject-verify.sh - FALSIFY a gate case by injecting the WRONG rule into its shader.
#
# A case that cannot fail is not an oracle.  This script applies ONE named injection (the wrong rule), recompiles
# the shader from source, runs the gate on the default ICD, and REQUIRES the named case to FAIL - then restores
# the tree.  It never runs a stale binary: a missing anchor prints ANCHOR MISSED, a shader that will not compile
# prints DID NOT COMPILE (and the whole tree is restored), and restoring is on every exit path.
#
#   inject-verify.sh q4-gather-offset   kv_q4_gather.comp   drop the "-8" code offset
#                                       -> must FAIL  "kv_q4 round trip: append (rotated) -> gather ..."
#   inject-verify.sh q8-round-half-up   quantize_q8_0.comp  replace the ties-to-even rounding with half-toward-+inf
#                                       -> must FAIL  "quantize_q8_0 (ggml bytes)"
#   inject-verify.sh cvec-apply-drop-scale  cvec_apply.comp     drop the per-layer reflect factor s
#                                       -> must FAIL  "cvec_apply: project removes s(h.v)v ..."
#   inject-verify.sh gather-rows-identity   gather_rows.comp    ignore ids[r]: gather the row at the POSITION
#                                       -> must FAIL  "gather_rows: 16-byte-aligned rows ..."
#   inject-verify.sh scatter-rows-identity  scatter_rows_f32.comp  write dst row r instead of rows[r]
#                                       -> must FAIL  "scatter_rows_f32: permutation ..."
#   inject-verify.sh iq-dequant-iq1m-grid-high  common/iq_dequant.glsl  misplace the IQ1_M grid high bit
#                                       -> must FAIL  "iq_dequant_f32: IQ1_M"
#   inject-verify.sh iq-dequant-iq2xxs-grid-index  common/iq_dequant.glsl  read the neighbour lane's IQ2_XXS grid byte
#                                       -> must FAIL  "iq_dequant_f32: IQ2_XXS"
#   inject-verify.sh iq-dequant-iq2xs-grid-high    common/iq_dequant.glsl  drop the IQ2_XS 512-point grid's high bit
#                                       -> must FAIL  "iq_dequant_f32: IQ2_XS"
#   inject-verify.sh iq-embed-rows-identity     iq_embed_rows.comp  gather the row at the POSITION
#                                       -> must FAIL  "iq_embed_rows: ..."
#   inject-verify.sh native-q5k-aux-half        native_q5_k_f32.comp  drop the packed-scale half switch
#                                       -> must FAIL  "native_q5_k_f32"
#   inject-verify.sh moe-hit-select-residency   moe_hit_select.comp  write a non-resident hit too
#                                       -> must FAIL  "moe_hit_select"
#   inject-verify.sh moe-hit-grouped-s2-hit0-intermediate  s2expert_down.comp  every hit reads hit 0's intermediate
#                                       -> must FAIL  "moe_hit_grouped_s2"
#   inject-verify.sh moe-grouped-s2-entry-token  s2expert_gu_grouped.comp  every entry reads token 0's activation
#                                       -> must FAIL  "moe_grouped_s2"
#   inject-verify.sh moe-hit-add-accumulate      moe_hit_add.comp  `+=` -> `=` (the CPU's prior value dropped)
#                                       -> must FAIL  "moe_hit_add"
#   inject-verify.sh fwht-entry-wrong-view-offset  vulkan/src/kernels/fwht_vk.cpp  bind the source view one float
#                                       early -> must FAIL  "fwht256 entry point: engine wrapper == shader path"
#                                       (this is the ENGINE-side backend, a non-shader/non-harness source, so the
#                                       script rebuilds the gate - which links vulkan/src/device/ + the kernel TU)
#
#   inject-verify.sh gdn-conv-tap-order     gdn_conv_step.comp  reverse the tap order in the four-tap conv
#                                       -> must FAIL  "gdn_conv_step"
#   inject-verify.sh gdn-l2-norm-eps-on-mean  gdn_l2_norm.comp  put eps on the MEAN instead of the squared norm
#                                       -> must FAIL  "gdn_l2_norm"
#   inject-verify.sh gdn-beta-gate-drop-sigmoid  gdn_beta_gate.comp  drop the sigmoid (the raw projection)
#                                       -> must FAIL  "gdn_beta_gate"
#   inject-verify.sh gdn-step-head-pairing  gdn_step.comp  INTERLEAVE head pairing instead of MODULO (h/(h_v/h_k))
#                                       -> must FAIL  "gdn_step"
#   inject-verify.sh gdn-out-norm-eps-on-sum  gdn_out_norm.comp  put eps on the SUM (the gdn_l2_norm convention)
#                                       instead of the MEAN -> must FAIL  "gdn_out_norm"
#   inject-verify.sh qsa-gate-first-half   qsa_gate_apply_f32.comp  take the gate from the FIRST half of the
#                                       2*head_dim block -> must FAIL  "qsa_gate_apply_f32"
#   inject-verify.sh gr-write-drop-two-centring  gr_write.comp  drop the 2 that centres the write's gate on 1
#                                       -> must FAIL  "gr_write"
#   inject-verify.sh indexer-key-append-rotate-last  indexer_key_append.comp  rotate the pooled row at the
#                                       block's LAST cell -> must FAIL  "indexer_key_append"
#   inject-verify.sh gr-read-mean-vs-sum  gr_mean.comp  drop the `/ hc` (the SUM over streams, not the mean)
#                                       -> must FAIL  "gr_read"
#
#   (performance tier, class B - each replaces a legacy kernel)
#   inject-verify.sh native-rope-adjacent-pairing      native_rope_apply.comp  adjacent-pair convention
#                                       -> must FAIL  "native_rope_apply"
#   inject-verify.sh native-router-top10-tie-high-index  common/router_select.glsl  tie -> highest index
#                                       -> must FAIL  "native_router_top10"
#   inject-verify.sh native-moe-combine-drop-shared    native_moe_combine.comp  invert the shared add
#                                       -> must FAIL  "native_moe_combine"
#   inject-verify.sh native-qsa-rms-norm-eps-on-sum    native_qsa_rms_norm_weighted.comp  eps on SUM
#                                       -> must FAIL  "native_qsa_rms_norm_weighted"
#   inject-verify.sh native-qsa-gate-first-half    native_qsa_gate_apply.comp  take the gate from the FIRST half
#                                       of the 2*head_dim block -> must FAIL  "native_qsa_gate_apply"
#   inject-verify.sh native-caps-qsa-false  vulkan/src/kernels/native_caps_vk.cpp  answer the QSA flag FALSE
#                                       while every gated shader exists (this batch ported the LAST gated symbol,
#                                       `native_qsa_gate_apply`, so the flag now answers TRUE; the pre-batch
#                                       injection that answered it true was the truth then and is retired)
#                                       -> must FAIL  "native capabilities: qsa flag"
#
#   (performance tier, class B, batch 2 - the native GDN / DeltaNet mixer)
#   inject-verify.sh native-caps-gdn-false  vulkan/src/kernels/native_caps_vk.cpp  answer the GDN flag FALSE
#                                       while every gated shader exists (batch 4 flipped it TRUE; the pre-batch-4
#                                       injection that answered it true is now the truth and is retired)
#                                       -> must FAIL  "native capabilities: gdn flag"
#   inject-verify.sh native-gdn-conv-silu-drop-silu  native_gdn_conv_silu.comp  drop the fused SiLU
#                                       -> must FAIL  "native_gdn_conv_silu"
#   inject-verify.sh native-gdn-l2-norm-drop-folded-scale  native_gdn_l2_norm.comp  drop the folded 1/sqrt(S)
#                                       -> must FAIL  "native_gdn_l2_norm"
#   inject-verify.sh native-gdn-beta-gate-sign-flip  native_gdn_beta_gate.comp  flip the sigmoid's exponent sign
#                                       -> must FAIL  "native_gdn_beta_gate"
#
#   (performance tier, class B, batch 3 - the remaining three native GDN / DeltaNet mixer kernels)
#   inject-verify.sh native-gdn-gate-drop-ssm-a  native_gdn_gate.comp  drop the ssm_a factor of
#                                       `softplus(alpha+dt) * ssm_a` -> must FAIL  "native_gdn_gate"
#   inject-verify.sh native-gdn-out-norm-silu-instead-of-sigmoid  native_gdn_out_norm.comp  SiLU instead of
#                                       sigmoid (the gdn_parity.cpp s4 trap) -> must FAIL  "native_gdn_out_norm"
#   inject-verify.sh native-gdn-step-drop-readout-scale  native_gdn_step.comp  drop the folded 1/sqrt(S)
#                                       readout scale the native kernel fuses -> must FAIL  "native_gdn_step"
#
#   (performance tier, class B, batch 4 - the three fused GDN paths; the flag now answers TRUE)
#   inject-verify.sh fused-gdn-conv-l2-drop-norm  fused_gdn_conv_l2.comp  drop the per-head L2 scale
#                                       -> must FAIL  "fused_gdn_conv_l2"
#   inject-verify.sh fused-gdn-ab-swap-bf16-halves  fused_gdn_ab.comp  swap the BF16 pair halves
#                                       -> must FAIL  "fused_gdn_ab"
#   inject-verify.sh fused-gdn-step-norm-silu-not-sigmoid  fused_gdn_step_norm.comp  SiLU instead of sigmoid
#                                       -> must FAIL  "fused_gdn_step_norm"
#
#   (the BF16-projection pair: `bf16_gemv` / `bf16_gemv_split`, ONE shared shader, the DEFAULT side of
#    `native_bf16_projections`)
#   inject-verify.sh bf16-gemv-swap-halves  bf16_gemv.comp  swap the BF16 pair halves
#                                       -> must FAIL  "bf16_gemv"
#   inject-verify.sh bf16-gemv-row-base     bf16_gemv.comp  index the weight row with the OUTPUT stride
#                                       -> must FAIL  "bf16_gemv"
#
#   (THIS BATCH: the two S-family split GEMV entry points and the SHARED EXPERT; plus the fused_gr_supported
#    reachability fix.  The two split GEMVs drive ONE shader, so one engine-side and one shader-side
#    falsification; the shared expert's falsification is the documented SILU-on-up trap; the fused_gr_supported
#    one re-claims the unported fused read that the shipped --native launch would then reach - a HOLE.)
#   inject-verify.sh s-gemv-q8k-flag-flip   vulkan/src/kernels/matvec_vk.cpp  hardwire the push constant's `q8k`
#                                       to 0 (read a Q8_K buffer as Q8_0) -> must FAIL  "s_gemv_q8k_split entry"
#   inject-verify.sh s-gemv-q8-split-wrong-act-block  s_gemv_q8_split.comp  use the Q8_0 block STRIDE for the
#                                       Q8_K image -> must FAIL  "s_gemv_q8k_split entry"
#   inject-verify.sh shared-expert-silu-on-up  vulkan/src/kernels/shared_expert_vk.cpp  put the SiLU on `up`
#                                       instead of the GATE -> must FAIL  "shared_expert entry"
#   (THE FUSED HYPER-CONNECTION READ - the P6 verify window's per-layer GR read, a native pack's ONLY decode
#    path.  `fused_gr_read` / `fused_gr_read_multi` are now WIRED (vulkan/src/kernels/fused_gr_vk.cpp) over four
#    shaders, and `fused_gr_supported` answers the engine's own geometry predicate TRUE.  Six falsifications,
#    each of them a rule of the fused read: the capability going FALSE while the kernels exist, the fold dropped,
#    the mean read as a sum, the bf16 pair halves swapped, the inject's `rs` dropped, and the multi reading
#    token 0's arguments for every token.)
#   inject-verify.sh fused-gr-supported-false  vulkan/src/kernels/ple_vk.cpp  answer the capability FALSE while
#                                       the four fused_gr shaders exist (the exact state that refused a native
#                                       pack's only decode path at verify.cpp:336) -> must FAIL
#                                       "fused_gr_supported entry"
#   inject-verify.sh fused-gr-rs-drop-fold   fused_gr_rs.comp  force `ap` false (the previous half's write no
#                                       longer folded, R_out no longer written) -> must FAIL "fused_gr_read entry:
#                                       wrapper vs the engine's own rule"
#   inject-verify.sh fused-gr-mix-sum-not-mean  fused_gr_mix.comp  drop the `/ hc` (a SUM over streams) -> must
#                                       FAIL "fused_gr_read entry: wrapper vs the engine's own rule"
#   inject-verify.sh fused-gr-down-swap-halves  fused_gr_down.comp  swap the BF16 pair halves of the down dot
#                                       -> must FAIL "fused_gr_read entry: wrapper vs the engine's own rule"
#   inject-verify.sh fused-gr-inject-drop-rs  fused_gr_inject.comp  drop the `rs[c]` factor of the inject's
#                                       activation -> must FAIL "fused_gr_read entry: wrapper vs the engine's own
#                                       rule"
#   inject-verify.sh fused-gr-multi-token0-args  vulkan/src/kernels/fused_gr_vk.cpp  read token 0's arguments
#                                       for every token of the window -> must FAIL "fused_gr_read_multi entry:
#                                       3 tokens == 3 x fused_gr_read, BITWISE"
#
#   (the earlier `fused-gr-supported-true` injection is RETIRED: it answered the CUDA geometry rule TRUE while
#    no fused_gr shader existed, which was the lie THEN; the rule IS the answer now, so the mirror lie
#    `fused-gr-supported-false` above is the one that falsifies.)
#
#   (the DEFAULT QSA decode attention: the KV pools read through the PAGE TABLE, `qsa_decode_attn_step`)
#   inject-verify.sh qsa-decode-attn-drop-kv-head  qsa_decode_attn.comp  drop the KV head term from the pool
#                                       row index -> must FAIL  "qsa_decode_attn (page_size=4"
#
#   (the SAMPLER family - the default SPLIT path, its f32 sibling, the `sample_tokens` choice, the coupled drafter)
#   inject-verify.sh sampler-split-merge-drop-parts  common/sampler_select.glsl  drain the running list before
#                                       the partition's -> must FAIL  "sampler_split:"
#   inject-verify.sh sampler-select-penalty-drop  common/sampler_select.glsl  zero the HOISTED penalty's count
#                                       (the per-partition penalty cache added when the split's per-round window
#                                       rescan was removed: 348.95 ms -> 13.53 ms on the Arc) -> must FAIL
#                                       "sampler_split: the repeat penalty"
#   inject-verify.sh sampler-kernel-f32-top-p-boundary  common/sampler_tail.glsl  drop the >= top_p boundary
#                                       -> must FAIL  "sampler_kernel_f32: one survivor (top_p cut of one)"
#   inject-verify.sh sample-tokens-choice-temp0-to-sampled  harness/vk_gate.cpp  temp 0 no longer routes to
#                                       the argmax -> must FAIL  "sample_tokens: temperature 0"
#   inject-verify.sh sample-tokens-entry-temp0-to-sampled  vulkan/src/kernels/sampler_vk.cpp  the ENGINE WRAPPER
#                                       stops routing temp 0 to the argmax -> must FAIL  "sample_tokens entry
#                                       (temperature 0)"
#   inject-verify.sh coupled-draft-counter-off-by-one  coupled_sample.comp  drop the +1 of
#                                       `coupled_draft_counter` -> must FAIL  "coupled_draft: the counter"
#   inject-verify.sh coupled-draft-window-start    coupled_penalize.comp  drop the draft index j from the
#                                       window start -> must FAIL  "coupled_draft: the window"
#
#   (THE DOORBELL RING AT REPLAY - the engine's own requirement that the ring re-reads its host copy and
#    increments, so a replayed block rings again rather than replaying a stale literal)
#   inject-verify.sh doorbell-ring-captured-literal  vulkan/src/kernels/doorbell_vk.cpp  raise the ring on the
#                                       HOST at capture (the port's old defect) -> must FAIL  "doorbell ring:
#                                       capture records, does not run"
#   inject-verify.sh doorbell-ring-shader-literal  ring_inc.comp  store the push-constant instead of the
#                                       re-read increment -> must FAIL  "doorbell ring: direct increments the ring"
#   inject-verify.sh doorbell-wait-records-node  vulkan/src/kernels/doorbell_vk.cpp  record a device node under
#                                       capture where the wait must submit nothing (a waiting kernel in a
#                                       captured block) -> must FAIL  "doorbell_wait: records NOTHING under capture"
#
# Usage: inject-verify.sh <name> [icd.json]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"       # ports/vulkan
SH="$ROOT/shaders"
BUILD="$ROOT/harness/build"
GATE="$BUILD/vk_gate"
TREE="$(cd "$ROOT/../.." && pwd)"                 # the engine tree (for the real headers)
[ -x "$GATE" ] || { echo "no gate binary at $GATE - run run_gate.sh once first"; exit 4; }

# An injection can live in a SHADER (glslc) or in the HARNESS (a case's own rule, e.g. sample_tokens' dispatch
# choice).  Both are real falsifications, so both must rebuild the artifact they change - an injection that does
# not rebuild runs the stale binary and reports a clean result.
rebuild_harness() {
  # THE ENGINE-SIDE BACKEND IS PART OF THE HARNESS: a case can drive the engine's own entry points through
  # vulkan/src/, so a change to any of those sources must rebuild the gate.  THE SOURCE LIST IS DERIVED
  # (gates/harness_sources.sh), never hand-kept - see that file for why a literal list went stale three batches
  # running and silently weakened every injection.  The guard there fails loudly if a vulkan/src/**/*.cpp is
  # missing from the list it hands the compiler.
  source "$ROOT/gates/harness_sources.sh"
  build_harness
}
# A source that must be rebuilt into the GATE (the harness proper, or the engine-side backend the harness links).
# ANY file under the backend tree counts, at ANY depth: the list is globbed recursively, so the match here is
# recursive too (a nested vulkan/src/<a>/<b>/c.cpp that skipped this would not trigger a rebuild).
is_harness_src() {
  case "$1" in
    "$ROOT"/harness/*.cpp|"$ROOT"/harness/*.hpp) return 0 ;;
    "$TREE"/vulkan/src/*) return 0 ;;
    *) return 1 ;;
  esac
}

name="${1:-}"; icd="${2:-}"
case "$name" in
  q4-gather-offset)
    file="$SH/kv_q4_gather.comp"; spv="kv_q4_gather"
    old=$'KS_.y[dst + j] = uint16_t(f16_from_f32(float(kc - 8) * kd));'
    new=$'KS_.y[dst + j] = uint16_t(f16_from_f32(float(kc) * kd));   // INJECTION: the -8 offset dropped'
    want="FAIL  kv_q4 round trip" ;;
  q8-round-half-up)
    file="$SH/quantize_q8_0.comp"; spv="quantize_q8_0"
    old=$'        double q = fl + ((u - fl > 0.5) ? 1.0 : ((u - fl < 0.5) ? 0.0 : odd));'
    new=$'        double q = fl + ((u - fl >= 0.5) ? 1.0 : 0.0);   // INJECTION: round half toward +inf'
    want="FAIL  quantize_q8_0 (ggml bytes)" ;;
  cvec-apply-drop-scale)
    file="$SH/cvec_apply.comp"; spv="cvec_apply"
    old=$'    if (steer && pc.mode == 0) dot = wg_sum(dot) * SL.v[pc.layer];'
    new=$'    if (steer && pc.mode == 0) dot = wg_sum(dot);   // INJECTION: the per-layer scale s dropped'
    want="FAIL  cvec_apply: project removes" ;;
  gather-rows-identity)
    file="$SH/gather_rows.comp"; spv="gather_rows"
    old=$'    DST.b[i] = SRC.b[uint(IDS.v[r]) * pc.row_bytes + o];'
    new=$'    DST.b[i] = SRC.b[r * pc.row_bytes + o];   // INJECTION: ids[r] ignored - the row at the position'
    want="FAIL  gather_rows: 16-byte-aligned rows" ;;
  scatter-rows-identity)
    file="$SH/scatter_rows_f32.comp"; spv="scatter_rows_f32"
    old=$'    const uint dbase = uint(ROWS.v[r]) * pc.width;'
    new=$'    const uint dbase = r * pc.width;   // INJECTION: the destination row indirection dropped'
    want="FAIL  scatter_rows_f32: permutation" ;;
  iq-dequant-iq1m-grid-high)
    # The rule lives in a shared INCLUDE, so the compile target is the shader that includes it (below).
    file="$SH/common/iq_dequant.glsl"; spv="iq_dequant_f32"; comp="$SH/iq_dequant_f32.comp"
    old=$'        const uint gidx = iq_b(bb + 4u * ib + il) | (((qh >> (4u * (il % 2u))) & 7u) << 8u);'
    new=$'        const uint gidx = iq_b(bb + 4u * ib + il) | (((qh >> (4u * (il % 2u))) & 7u) << 7u);   // INJECTION: IQ1_M grid high bit misplaced'
    want="FAIL  iq_dequant_f32: IQ1_M" ;;
  iq-dequant-iq2xxs-grid-index)
    # IQ2_XXS's rule is `aux8[il]`: each lane reads ITS OWN byte of the 4-byte grid-index field.  Reading the
    # neighbour lane's byte lands on a different 8-byte grid point, which is the plausible wrong layout.
    file="$SH/common/iq_dequant.glsl"; spv="iq_dequant_f32"; comp="$SH/iq_dequant_f32.comp"
    old=$'        const uint gidx = iq_b(q2 + il);'
    new=$'        const uint gidx = iq_b(q2 + il + 1u);   // INJECTION: the IQ2_XXS grid index read off by one lane'
    want="FAIL  iq_dequant_f32: IQ2_XXS" ;;
  iq-dequant-iq2xs-grid-high)
    # IQ2_XS's grid index is 9 bits (`q2[il] & 511`) because the grid has 512 points.  Masking 8 bits indexes the
    # first half of the grid and silently alias half the points.
    file="$SH/common/iq_dequant.glsl"; spv="iq_dequant_f32"; comp="$SH/iq_dequant_f32.comp"
    old=$'        const uint gidx = w & 511u;'
    new=$'        const uint gidx = w & 255u;   // INJECTION: the IQ2_XS 512-point grid high bit dropped'
    want="FAIL  iq_dequant_f32: IQ2_XS" ;;
  iq-embed-rows-identity)
    file="$SH/iq_embed_rows.comp"; spv="iq_embed_rows"
    old=$'    const uint row = uint(uint64_t(tok) * uint64_t(pc.row_bytes));'
    new=$'    const uint row = uint(uint64_t(gl_WorkGroupID.y) * uint64_t(pc.row_bytes));   // INJECTION: row index = position, not the token'
    want="FAIL  iq_embed_rows: " ;;
  native-q5k-aux-half)
    # The kernel's distinguishing rule: the 12-byte `scales` packs SIX 6-bit scales and SIX 6-bit mins and the
    # `qh` shift `>> bq8_offset` selects the low/high group.  Dropping the shift reads one half's fields for
    # every group.  The dot now lives in the SHARED include (common/k_dots.glsl) that BOTH native_q5_k_f32.comp
    # and the generic native_k_mmvq.comp include, so the injection targets the include (its including shader is
    # the compile target) and both paths move together.  `vh0` is Q5_K-only, so the anchor is unique.
    file="$SH/common/k_dots.glsl"; spv="native_q5_k_f32"; comp="$SH/native_q5_k_f32.comp"
    old=$'    const int vh0 = q5_i32(qh) >> int(bq8_offset);'
    new=$'    const int vh0 = q5_i32(qh);   // INJECTION: the packed-scale half shift dropped'
    want="FAIL  native_q5_k_f32" ;;
  moe-hit-select-residency)
    # The selection's rule is "only a RESIDENT expert is a hit".  Dropping the test writes every routed entry,
    # with slot -1 for the non-resident ones and a count that includes them.
    file="$SH/moe_hit_select.comp"; spv="moe_hit_select"
    old=$'        if (s >= 0) {\n            slot.v[c] = s;\n            dst.v[c] = lane;\n            ++c;\n        }'
    new=$'        {   // INJECTION: a non-resident hit is written too\n            slot.v[c] = s;\n            dst.v[c] = lane;\n            ++c;\n        }'
    want="FAIL  moe_hit_select" ;;
  moe-hit-grouped-s2-hit0-intermediate)
    # The chain's wiring: `s2expert_down` reads hit h's own FF/32 q8_0 blocks at h*(FF/32)*34.  Reading hit 0's
    # for every hit is a stride error the individual down arm also catches, and the composed case must too.
    file="$SH/s2expert_down.comp"; spv="s2expert_down"
    old=$'    const uint x_off = h * (FF / 32u) * 34u;'
    new=$'    const uint x_off = 0u;   // INJECTION: every hit reads hit 0\'s intermediate'
    want="FAIL  moe_hit_grouped_s2" ;;
  moe-grouped-s2-entry-token)
    # The grouped gu's distinguishing rule is that EVERY ENTRY reads ITS OWN token's activation row (`ent_tok[e]`).
    # Making every entry read token 0's row is the plausible wrong rule the per-hit sibling's single activation
    # makes tempting - and it changes the UP rows, which the composed case pins against the blob.
    file="$SH/s2expert_gu_grouped.comp"; spv="s2expert_gu_grouped"
    old=$'            const uint tok = uint(ent_tok.v[e]);      // THIS entry\'s token, not the group\'s'
    new=$'            const uint tok = 0u;   // INJECTION: every entry reads token 0\'s activation'
    want="FAIL  moe_grouped_s2" ;;
  moe-hit-add-accumulate)
    # The accumulator's rule is `+=`: `parts` holds what the CPU left there.  `=` drops that prior value, and the
    # case gives every named row a NON-ZERO prior, so it is caught on the first row.
    file="$SH/moe_hit_add.comp"; spv="moe_hit_add"
    old=$'        parts.v[row + i] += hit_out.v[row + i];'
    new=$'        parts.v[row + i] = hit_out.v[row + i];   // INJECTION: the accumulate dropped'
    want="FAIL  moe_hit_add" ;;
  fwht-entry-wrong-view-offset)
    # THE ARENA'S failure mode, and the I1 case's named falsification.  The engine wrapper resolves a raw device
    # pointer to an arena VIEW and binds it; a wrong offset makes the shader read one thing while the transfer
    # wrote another.  NOTE: a UNIFORM shift (e.g. inside arena_resolve) does NOT bite - write, dispatch and read
    # all move together and cancel - which is exactly why the offset that matters is the DISPATCH's, so this
    # injection shifts the view the dispatch binds.
    file="$TREE/vulkan/src/kernels/fwht_vk.cpp"
    old=$'    s.ctx->dispatch(pipe, {&sv, &dv}, &pc, sizeof(pc), (uint32_t) n_rows);'
    new=$'    Buf s_src = sv; s_src.offset += 4u;   // INJECTION: the source view bound one float early\n    s.ctx->dispatch(pipe, {&s_src, &dv}, &pc, sizeof(pc), (uint32_t) n_rows);'
    want="FAIL  fwht256 entry point: engine wrapper == shader path" ;;
  sampler-split-merge-drop-parts)
    # The split's whole content is the MERGE: a row's list is the ordered union of its 4096-logit partitions'
    # lists.  Taking the running list first (instead of comparing the two heads) appends each partition after
    # the first to the tail of what is already there, so everything past the first partition is lost - which the
    # multi-partition arms (12288, 248320) must catch and a one-partition row cannot.
    file="$SH/common/sampler_select.glsl"; spv="sampler_split"; comp="$SH/sampler_split.comp"
    old=$'                else take_a = (sc_sel_lg[a] > sc_tmp_lg[c]) ||\n                              (sc_sel_lg[a] == sc_tmp_lg[c] && sc_sel_id[a] < sc_tmp_id[c]);'
    new=$'                else take_a = (a < ncur);   // INJECTION: the merge drains the running list first'
    want="FAIL  sampler_split:" ;;
  sampler-select-penalty-drop)
    # THE PENALTY HOIST (`common/sampler_select.glsl`).  `sampler_row_topk` now computes each partition
    # element's penalised logit ONCE, into the lane's registers (the engine's own `s[kSplitPerLane]` shape),
    # and runs the k rounds over the CACHED values - instead of re-reading the logit and re-scanning the
    # whole penalty window on every round.  Measured on the Arc at vocab 248320 / n_tokens 1 / k 64 / window
    # 64: 348.95 ms with the per-round rescan, 13.53 ms with the hoist.  Zeroing the count makes the wrong
    # rule the RAW logit; the split's penalty arm (id 100 hit 8x, `penalty_repeat = 4`) must move its head
    # from 4200 back to 100, so the arm is not decorative.
    file="$SH/common/sampler_select.glsl"; spv="sampler_split"; comp="$SH/sampler_split.comp"
    old=$'                s = sc_penalized(s, cnt, pc.penalty_repeat, pc.penalty_freq, pc.penalty_present);'
    new=$'                s = sc_penalized(s, 0, pc.penalty_repeat, pc.penalty_freq, pc.penalty_present);   // INJECTION: the hoisted penalty dropped (zero count)'
    want="FAIL  sampler_split: the repeat penalty" ;;
  coupled-draft-counter-off-by-one)
    # THE coupled rule: a draft at cell c is verified by a row drawn with counter c+1, so the drafter draws with
    # counter c+1 too.  Dropping the +1 makes it draw with c - a different (still valid) Philox stream and a worse
    # acceptance rate, which reads as "the model got a bit worse".  The equal-survivor arm observes the stream
    # exactly (the softmax cancels out of the walk), so it must move.
    file="$SH/coupled_sample.comp"; spv="coupled_sample"
    old=$'        const uint counter_lo = uint(STEP_.o[0]) + 1u;   // coupled_draft_counter(cell) = cell + 1'
    new=$'        const uint counter_lo = uint(STEP_.o[0]);   // INJECTION: the +1 of coupled_draft_counter dropped'
    want="FAIL  coupled_draft: the counter" ;;
  coupled-draft-window-start)
    # Draft j's penalty window is the ring's [cap + j - h, cap + j) - coupled_hist_start.  Dropping j reads the
    # STAGED BASE instead of the drafts, so the wrong subset indices are penalised (or none are).
    file="$SH/coupled_penalize.comp"; spv="coupled_penalize"
    old=$'    const int start = pc.cap + pc.j - h;                                          // coupled_hist_start'
    new=$'    const int start = pc.cap - h;   // INJECTION: the draft index j dropped from the window start'
    want="FAIL  coupled_draft: the window" ;;
  sample-tokens-choice-temp0-to-sampled)
    # The entry point's choice, and the one the task names: `temperature == 0` must route to the GREEDY kernel,
    # NOT to the sampled path's uniform draw.  Dropping the temperature test sends a temp-0 request to the sampler,
    # which draws uniformly over the shortlist - a different token for most seeds.  The case's temp-0 row runs the
    # chosen kernel, so the token moves (it also reads the pure choice table as WRONG).  This is a HARNESS source,
    # so the script rebuilds the gate rather than running a stale binary.
    file="$ROOT/harness/vk_gate.cpp"; spv="vk_gate"
    old=$'    if (greedy || temperature <= 0.0f) return P_GREEDY;'
    new=$'    if (greedy || temperature < 0.0f) return P_GREEDY;   // INJECTION: temp 0 no longer routes to the argmax'
    want="FAIL  sample_tokens: temperature 0" ;;
  sample-tokens-entry-temp0-to-sampled)
    # THE ENGINE WRAPPER's OWN routing rule (`vulkan/src/kernels/sampler_vk.cpp`): `sample_tokens` must send a
    # temperature-0 request to the ARGMAX shader, not to the sampled path's uniform draw (sampler.cu:1006).  The
    # harness injection above edits the CASE's own predicate, which the wrapper does not consult; THIS edits the
    # WRAPPER, so it is the arm that proves the SHIPPED rule.  Dropping the temperature test makes the wrapper
    # dispatch the sampled split for a temp-0 request - a different token for most seeds, which
    # `case_sample_tokens_entry`'s temperature-0 arm must observe.  ENGINE-side backend source, so the script
    # rebuilds the gate.
    file="$TREE/vulkan/src/kernels/sampler_vk.cpp"; spv="sampler_vk"
    old=$'    if (p.greedy || p.temperature <= 0.0f) {'
    new=$'    if (p.greedy) {   // INJECTION: temperature 0 no longer routes to the argmax'
    want="FAIL  sample_tokens entry (temperature 0)" ;;
  sampler-kernel-f32-top-p-boundary)
    # The portable f32 tail's top_p cut is `>=` (the SAMPLED chain's boundary, llama.cpp's).  Changing it to `>`
    # drops the boundary case: a cut of exactly top_p no longer closes the prefix, so a one-survivor shortlist
    # becomes two and the token depends on the draw.  The case's "one survivor (top_p cut of one)" arm is built to
    # a 0.50 boundary and must see it.
    file="$SH/common/sampler_tail.glsl"; spv="sampler_kernel_f32"; comp="$SH/sampler_kernel_f32.comp"
    old=$'            if (cum >= top_p) { cut = i + 1; break; }'
    new=$'            if (cum > top_p) { cut = i + 1; break; }   // INJECTION: the >= boundary dropped'
    want="FAIL  sampler_kernel_f32: one survivor (top_p cut of one)" ;;
  gdn-conv-tap-order)
    # `gdn_conv_step`'s rule reads tap i from `kW[c*d_conv+i]` (GGML-NATIVE, tap fastest).  Reversing the tap
    # order is the plausible wrong layout (the reference's (d_conv, C) row-major), and the fixture gives each tap
    # a DISTINCT magnitude so a reversal cannot hide in rounding noise.
    file="$SH/gdn_conv_step.comp"; spv="gdn_conv_step"
    old=$'    for (uint i = 0u; i + 1u < dc; ++i) acc += cs.v[sb + i] * w.v[wb + i];   // OLDEST state row is tap 0'
    new=$'    for (uint i = 0u; i + 1u < dc; ++i) acc += cs.v[sb + i] * w.v[wb + (dc - 1u - i)];   // INJECTION: tap order reversed'
    want="FAIL  gdn_conv_step" ;;
  gdn-l2-norm-eps-on-mean)
    # The kernel's distinguishing rule: `eps` is an ABSOLUTE floor on the SQUARED NORM (`sqrt(sum + eps)`).  The
    # rival reading puts the same eps on the MEAN and differs by sqrt(cols) = 11.3x at cols = 128 - which
    # gdn_parity.cpp §3 pins, and which the fixture's near-zero row also exercises in a ~1.5x regime.
    file="$SH/gdn_l2_norm.comp"; spv="gdn_l2_norm"
    old=$'    const float inv = inversesqrt(wg_sum(acc) + pc.eps);'
    new=$'    const float inv = inversesqrt(wg_sum(acc) / float(pc.cols) + pc.eps);   // INJECTION: eps on the MEAN'
    want="FAIL  gdn_l2_norm" ;;
  gdn-beta-gate-drop-sigmoid)
    # THE C1 bug this kernel exists to prevent: handing the recurrence the raw `ssm_beta @ cur` instead of the
    # fraction `gdn_step`'s contract demands (`d = (v - sk) * beta`).  The fixture spans the sigmoid's whole range
    # (large negative -> ~0, large positive -> ~1), so an identity reading moves every value.
    file="$SH/gdn_beta_gate.comp"; spv="gdn_beta_gate"
    old=$'    b.v[i] = 1.0f / (1.0f + exp(-b.v[i]));'
    new=$'    b.v[i] = b.v[i];   // INJECTION: the sigmoid dropped - the raw projection handed to the recurrence'
    want="FAIL  gdn_beta_gate" ;;
  gdn-step-head-pairing)
    # `gdn_step`'s rule is `src = h % h_k` (MODULO).  The INTERLEAVE reading `h / (h_v/h_k)` is the plausible
    # wrong rule: both produce a full-rank state of the right shape (gdn_parity.cpp section 1's first trap).  The
    # fixture promises this moves the output host-side; here it must move the DEVICE output too.
    file="$SH/gdn_step.comp"; spv="gdn_step"
    old=$'    const uint src = h % uint(pc.h_k);                   // MODULO head pairing (not h / (h_v/h_k))'
    new=$'    const uint src = h / (hv / uint(pc.h_k));   // INJECTION: INTERLEAVE head pairing'
    want="FAIL  gdn_step" ;;
  gdn-out-norm-eps-on-sum)
    # The kernel's distinguishing rule vs its neighbour `gdn_l2_norm`: here the eps sits on the MEAN
    # (`sum/S + eps`).  Taking the gdn_l2_norm convention drops the /S and scales every output by sqrt(S).
    file="$SH/gdn_out_norm.comp"; spv="gdn_out_norm"
    old=$'    const float inv = inversesqrt(wg_sum(acc) / float(S) + pc.eps);'
    new=$'    const float inv = inversesqrt(wg_sum(acc) + pc.eps);   // INJECTION: eps on the SUM, not the mean'
    want="FAIL  gdn_out_norm" ;;
  qsa-gate-first-half)
    # The gate is the SECOND half of each head's 2*head_dim block (qsa_parity.cpp PROPERTY 10 / its
    # `gate_second_half` rival).  Reading the FIRST half is the plausible wrong layout - and the fixture's first
    # half is built so its sigmoid differs from the second's.
    file="$SH/qsa_gate_apply_f32.comp"; spv="qsa_gate_apply_f32"
    old=$'    const float g = qfull.v[h * 2u * hd + hd + d];   // the SECOND half of the 2*head_dim block'
    new=$'    const float g = qfull.v[h * 2u * hd + d];   // INJECTION: the gate read from the FIRST half'
    want="FAIL  qsa_gate_apply_f32" ;;
  gr-write-drop-two-centring)
    # The `2 * sigmoid` is what CENTRES the write's gate on 1 (gr_parity.cpp item 6): a zero injection must give
    # w = 1 EXACTLY, i.e. a plain residual add.  Dropping the 2 is the obvious slip.  The fixture's non-zero
    # injections are sized ~ O(hc) so the sigmoid is RESPONSIVE (an inject of several hc saturates both readings
    # and the arm would be decorative), and the zero-inject PROPERTY arm moves too.
    file="$SH/gr_write.comp"; spv="gr_write"
    old=$'        const float w = 2.0f / (1.0f + exp(-inj.v[c] / float(pc.hc)));   // 2*sigmoid(inject/hc), centred on 1'
    new=$'        const float w = 1.0f / (1.0f + exp(-inj.v[c] / float(pc.hc)));   // INJECTION: the 2 dropped'
    want="FAIL  gr_write" ;;
  indexer-key-append-rotate-last)
    # The pooled row is rotated at the block's FIRST cell (qsa_parity.cpp PROPERTY 4).  Rotating at its LAST cell
    # keeps every shape and every magnitude; the fixture checks host-side that it moves the pooled rows, so this
    # one-line change must move the DEVICE output too.
    file="$SH/indexer_key_append.comp"; spv="indexer_key_append"
    old=$'        const uint toff = uint(pc.pos_base + int(b) * pc.r) * nhalf + d;'
    new=$'        const uint toff = uint(pc.pos_base + int(b) * pc.r + pc.r - 1) * nhalf + d;   // INJECTION: rotate at the LAST cell'
    want="FAIL  indexer_key_append" ;;
  gr-read-mean-vs-sum)
    # `gr_read`'s last stage is the MEAN over the streams (gr_parity.cpp item 4).  Dropping the `/ hc` leaves a
    # SUM - a factor of hc that reads as a scale problem rather than a structural one.  The case host-checks that
    # the sum reading moves `mixed` (measured ratio ~3.0 at the artifact), so the device output must move too.
    file="$SH/gr_mean.comp"; spv="gr_mean"
    old=$'    mixed.v[d] = m / float(pc.hc);'
    new=$'    mixed.v[d] = m;   // INJECTION: the /hc dropped - a SUM over the streams, not the mean'
    want="FAIL  gr_read" ;;
  native-qsa-rms-norm-eps-on-sum)
    # `native_qsa_rms_norm_weighted`'s rule divides the squared sum by `n_cols` (the MEAN) before the eps:
    # `rsqrt(sum/cols + eps)`.  Putting the eps on the SUM (the gdn_l2_norm convention) scales every output by
    # sqrt(cols).  The oracle is the rule's double transcription, so this bites on every shape.
    file="$SH/native_qsa_rms_norm_weighted.comp"; spv="native_qsa_rms_norm_weighted"
    old=$'    const float scale = inversesqrt(wg_sum(acc) / float(pc.cols) + pc.eps);'
    new=$'    const float scale = inversesqrt(wg_sum(acc) + pc.eps);   // INJECTION: eps on the SUM, not the mean'
    want="FAIL  native_qsa_rms_norm_weighted" ;;
  native-rope-adjacent-pairing)
    # The native rope's pairing is NEOX - `(pair, pair + n_rot/2)`.  Writing the results to the ADJACENT
    # (2*pair, 2*pair+1) slots is the plausible wrong convention, and rope_parity.cpp check 3 pins that it
    # produces correctly-SHAPED output with scrambled content, which a tolerance over all 256 dims would accept.
    file="$SH/native_rope_apply.comp"; spv="native_rope_apply"
    old=$'    OB.o[base + pair] = a * c - b * s;\n    OB.o[base + pair + nhalf] = a * s + b * c;'
    new=$'    OB.o[base + 2 * pair] = a * c - b * s;\n    OB.o[base + 2 * pair + 1] = a * s + b * c;   // INJECTION: the ADJACENT-pair convention'
    want="FAIL  native_rope_apply" ;;
  native-router-top10-tie-high-index)
    # The selection's rule is "the LOWEST index wins a tie" (router_top10_parity.cpp's stable descending argsort).
    # The scan's strict `>` keeps the lowest index on an exact tie; flipping it to `>=` keeps the HIGHEST.  The
    # all-equal arm makes every expert tie, so this is caught on that arm and on the 12-way tie arm.
    file="$SH/native_router_top10.comp"; spv="native_router_top10"
    old=$'        if (pe > bv) { bv = pe; bi = int(e); }   // STRICT: the lowest index wins a tie'
    new=$'        if (pe >= bv) { bv = pe; bi = int(e); }   // INJECTION: the tie rule flipped to the highest index'
    want="FAIL  native_router_top10" ;;
  native-moe-combine-drop-shared)
    # The native combine's rule is the shared expert's row added PLAIN (`if (shared) sum += shared[col]`).  The
    # inverted test drops it when it is present and adds it when it is absent - so BOTH arms move.
    file="$SH/native_moe_combine.comp"; spv="native_moe_combine"
    old=$'    if (pc.has_shared != 0) sum += sh.v[col];        // added PLAIN, exactly as the source adds it'
    new=$'    if (pc.has_shared == 0) sum += sh.v[col];        // INJECTION: the shared add inverted'
    want="FAIL  native_moe_combine" ;;
  native-caps-qsa-true)
    # RETIRED in this batch: this answered `native_qsa_enabled()` TRUE while `native_qsa_gate_apply` had no
    # shader.  Batch 5 ports that symbol, so TRUE is now the TRUTH and the injection no longer falsifies
    # anything; the mirror lie (answering FALSE while every gated shader exists) is `native-caps-qsa-false`.
    echo "RETIRED: native-caps-qsa-true - answering true is now the truth (see native-caps-qsa-false)" ; exit 2 ;;
  native-caps-qsa-false)
    # THE QSA FLAG'S own falsification, in the same form as batch 4's `native-caps-gdn-false`: this batch ported
    # the LAST symbol the flag gates (`native_qsa_gate_apply`), so both gated shaders exist and the flag must
    # answer TRUE.  The lie the invariant must catch is the flag answering FALSE while every gated shader exists.
    # This is the ENGINE-side backend, so the script rebuilds the gate (the capability TU is linked into it).
    file="$TREE/vulkan/src/kernels/native_caps_vk.cpp"
    old=$'bool native_qsa_enabled() { return true; }         // native_qsa_rms_norm_weighted + native_qsa_gate_apply, both built'
    new=$'bool native_qsa_enabled() { return false; }        // INJECTION: the QSA flag answered false while both gated shaders exist'
    want="FAIL  native capabilities: qsa flag" ;;
  native-qsa-gate-first-half)
    # The gate is the SECOND half of each head's 2*head_dim block (qsa_parity.cpp PROPERTY 10 / its
    # `gate_second_half` rival).  Reading the FIRST half is the plausible wrong layout - and the fixture's first
    # half is built so its sigmoid differs from the second's, which the case checks host-side before judging.
    file="$SH/native_qsa_gate_apply.comp"; spv="native_qsa_gate_apply"
    old=$'    const float raw = qfull.v[h * 2u * hd + hd + d];   // the SECOND half: [query, gate] per head'
    new=$'    const float raw = qfull.v[h * 2u * hd + d];   // INJECTION: the gate read from the FIRST half'
    want="FAIL  native_qsa_gate_apply" ;;
  qsa-decode-attn-drop-kv-head)
    # `qsa_decode_attn_step` is the DEFAULT decode attention (layer.cpp:980): it reads the KV POOLS through the
    # page table.  The pool row INTERLEAVES the KV head at the page level - `(page*kv_heads + kvh)*page_size +
    # (cell % page_size)`.  Dropping the `kvh` term makes BOTH KV heads read head 0's data: the whole GQA
    # grouping collapses and every head's output moves, so the shader-path arm against the engine's rule must FAIL.
    file="$SH/qsa_decode_attn.comp"; spv="qsa_decode_attn"
    old=$'        const int row = (page * pc.kv_heads + kvh) * pc.page_size + (cell % pc.page_size);'
    new=$'        const int row = page * pc.page_size + (cell % pc.page_size);   // INJECTION: the kv head term dropped'
    want="FAIL  qsa_decode_attn (page_size=4" ;;
  native-caps-gdn-false)
    # THE GDN FLAG'S own falsification, and it is batch 4's capability point: `native_gdn_enabled()` now answers
    # TRUE because ALL NINE gated symbols (the six native kernels and the three fused paths) have shaders.  The
    # mirror lie - the flag answering FALSE while every gated shader exists - is what the invariant must catch now
    # (before batch 4 the lie was `true` while a gated symbol was unported, which is what `native-caps-gdn-true`
    # checked; that injection is now the truth and is retired here).
    file="$TREE/vulkan/src/kernels/native_caps_vk.cpp"
    old=$'bool native_gdn_enabled() { return true; }         // all nine gated symbols have shaders (six native + three fused)'
    new=$'bool native_gdn_enabled() { return false; }        // INJECTION: the GDN flag answered false while every gated shader exists'
    want="FAIL  native capabilities: gdn flag" ;;
  native-gdn-conv-silu-drop-silu)
    # The native body's new content vs `gdn_conv_step` is the FUSED SiLU and the second output.  Writing the raw
    # sum into the SiLU output drops it; the case compares BOTH outputs against the native rule, so this bites on
    # every channel whose sum is not its own SiLU (which the case's margin check proves is most of them).
    file="$SH/native_gdn_conv_silu.comp"; spv="native_gdn_conv_silu"
    old=$'    silu.v[c] = sum / (1.0f + exp(-sum));                  // FP32 fast-math SiLU'
    new=$'    silu.v[c] = sum;   // INJECTION: the fused SiLU dropped - the raw sum written to the SiLU output'
    want="FAIL  native_gdn_conv_silu" ;;
  native-gdn-l2-norm-drop-folded-scale)
    # The native body applies TWO factors - `scale = rsqrtf(partial/S + eps/S)` and the folded
    # `scale_after = 1/sqrt(S)` - where the legacy kernel applies one and the layer adds the other in a separate
    # `scale_inplace`.  Dropping the second factor scales every output by sqrt(S) = 11.3.
    file="$SH/native_gdn_l2_norm.comp"; spv="native_gdn_l2_norm"
    old=$'        x.v[i] = (scale * x.v[i]) * pc.inv_sqrt_cols;    // the native body\'s two-factor scale'
    new=$'        x.v[i] = scale * x.v[i];   // INJECTION: the native body\'s folded 1/sqrt(S) dropped'
    want="FAIL  native_gdn_l2_norm" ;;
  native-gdn-beta-gate-sign-flip)
    # The rule is `sigmoid(x) = 1/(1+exp(-x))`.  Flipping the exponent's sign is the plausible slip and is a
    # different function; the fixture spans the sigmoid's whole range so every value moves.
    file="$SH/native_gdn_beta_gate.comp"; spv="native_gdn_beta_gate"
    old=$'    b.v[i] = 1.0f / (1.0f + exp(-b.v[i]));'
    new=$'    b.v[i] = 1.0f / (1.0f + exp(b.v[i]));   // INJECTION: the sigmoid exponent\'s sign flipped'
    want="FAIL  native_gdn_beta_gate" ;;
  native-gdn-gate-drop-ssm-a)
    # `native_gdn_gate`'s rule is `gate[i] = softplus(alpha[i] + dt[i]) * ssm_a[i]` - the second factor is the
    # SIGNED `ssm_a = -exp(A_log)`.  Writing just the softplus drops it; the fixture's ssm_a is negative and
    # of order 1, so the case checks host-side that the drop moves the output and it bites on every head.
    file="$SH/native_gdn_gate.comp"; spv="native_gdn_gate"
    old=$'    gate.v[i] = softplus * ssm_a.v[i];'
    new=$'    gate.v[i] = softplus;   // INJECTION: the ssm_a factor of softplus(alpha+dt)*ssm_a dropped'
    want="FAIL  native_gdn_gate" ;;
  native-gdn-out-norm-silu-instead-of-sigmoid)
    # The closing norm's gate is a SIGMOID, not SiLU (gdn_parity.cpp section 4 pins that the two are
    # distinguishable, and this artifact is the one that does NOT use qwen3.5's SiLU).  The fixture's z spans
    # +-20, so the SiLU reading moves the large-z third of every row by ~z at z = 20.
    file="$SH/native_gdn_out_norm.comp"; spv="native_gdn_out_norm"
    old=$'        yb.v[base + c] = weighted * (1.0f / (1.0f + exp(-zb.v[base + c])));   // sigmoid(z), NOT SiLU'
    new=$'        yb.v[base + c] = weighted * (zb.v[base + c] / (1.0f + exp(-zb.v[base + c])));   // INJECTION: SiLU instead of sigmoid'
    want="FAIL  native_gdn_out_norm" ;;
  native-gdn-step-drop-readout-scale)
    # The native body FUSES the `1/sqrt(S)` readout scale (`output = attn * scale`), which the legacy branch
    # applies in a separate `scale_inplace` launch.  Dropping it here makes this kernel match the legacy one
    # and scales every output by sqrt(128) = 11.3 - the native body's distinguishing fused step.
    file="$SH/native_gdn_step.comp"; spv="native_gdn_step"
    old=$'    ob.v[h * S + j] = attn * pc.scale;                  // the native body\'s folded 1/sqrt(S) readout scale'
    new=$'    ob.v[h * S + j] = attn;   // INJECTION: the folded 1/sqrt(S) readout scale dropped'
    want="FAIL  native_gdn_step" ;;
  fused-gdn-conv-l2-drop-norm)
    # `fused_gdn_conv_l2`'s fused content is the per-head L2 norm on top of the conv+SiLU: the kernel writes
    # `y * rsqrt(sum(y^2) + eps)` for the q/k heads.  Dropping the scale writes the plain SiLU for every head,
    # so the q/k arms move (the norm scales by ~1/sqrt(128) of the value there) and the v-untouched arm still
    # passes - the case compares the q/k heads against the fused rule.
    file="$SH/fused_gdn_conv_l2.comp"; spv="fused_gdn_conv_l2"
    old=$'        h.v[c] = norm_this_head ? (y * red[hg * 128u]) : y;'
    new=$'        h.v[c] = y;   // INJECTION: the per-head L2 scale of the fused conv+L2 dropped'
    want="FAIL  fused_gdn_conv_l2" ;;
  fused-gdn-ab-swap-bf16-halves)
    # The fused AB kernel reads each BF16 weight row as 32-bit PAIRS: the LOW half is element 2p, the HIGH half
    # 2p+1.  Swapping them is a plausible slip and is invisible on equal-magnitude data; the case's fixture
    # gives the two halves magnitudes 100x apart, so the swap moves every row by O(1).
    file="$SH/fused_gdn_ab.comp"; spv="fused_gdn_ab"
    old=$'        acc = fma(bf16_to_f32(w0 & 0xFFFFu), x.v[xb + 0u], acc);\n        acc = fma(bf16_to_f32(w0 >> 16u),    x.v[xb + 1u], acc);'
    new=$'        acc = fma(bf16_to_f32(w0 >> 16u),    x.v[xb + 0u], acc);   // INJECTION: the bf16 pair halves swapped\n        acc = fma(bf16_to_f32(w0 & 0xFFFFu), x.v[xb + 1u], acc);'
    want="FAIL  fused_gdn_ab" ;;
  fused-gdn-step-norm-silu-not-sigmoid)
    # The fused closing gate is `sigmoid(z)`, NOT SiLU (the gdn_parity.cpp section 4 trap: this artifact does not
    # use qwen3.5's SiLU).  The fixture's z spans +-20, so the SiLU reading moves the large-z third of every row
    # by ~z at z = 20 - and the case host-checks that the SiLU rival moves the output before judging.
    file="$SH/fused_gdn_step_norm.comp"; spv="fused_gdn_step_norm"
    old=$'        yb.v[head * S + j] = weighted * (1.0f / (1.0f + exp(-zb.v[head * S + j])));'
    new=$'        yb.v[head * S + j] = weighted * (zb.v[head * S + j] / (1.0f + exp(-zb.v[head * S + j])));   // INJECTION: SiLU instead of sigmoid'
    want="FAIL  fused_gdn_step_norm" ;;
  bf16-gemv-swap-halves)
    # The port reads a BF16 weight row as 32-bit PAIRS: element 2p in the LOW half, 2p+1 in the HIGH half.  The
    # swap is a plausible slip and is invisible on equal-magnitude data; row 0 of the fixture is the LAYOUT PROBE
    # (low halves ~1e3, high halves ~1e-3), so the swap moves every arm's row 0 by O(1) and this case must fail.
    file="$SH/bf16_gemv.comp"; spv="bf16_gemv"
    old=$'        acc += bf16_to_f32(pw & 0xFFFFu) * bf16_to_f32(px & 0xFFFFu);   // element 2p\n        acc += bf16_to_f32(pw >> 16u)    * bf16_to_f32(px >> 16u);      // element 2p+1'
    new=$'        acc += bf16_to_f32(pw >> 16u)    * bf16_to_f32(px & 0xFFFFu);   // INJECTION: pair halves swapped\n        acc += bf16_to_f32(pw & 0xFFFFu) * bf16_to_f32(px >> 16u);'
    want="FAIL  bf16_gemv n_in=2560 n_out=512" ;;
  bf16-gemv-row-base)
    # The OTHER layout trap: row o's weights start at `o * (n_in/2)` 32-bit words.  Reading at `o * n_out` (the
    # row stride of the OUTPUT, a plausible confusion) lands on a different row's weights; the arms are
    # NON-SQUARE on purpose so n_in/2 != n_out and the misread is O(1), not rounding.
    file="$SH/bf16_gemv.comp"; spv="bf16_gemv"
    old=$'    const uint wbase = o * npair;                    // row o of the weight is a contiguous run of npair words'
    new=$'    const uint wbase = o * uint(pc.n_out);   // INJECTION: the weight row base uses the OUTPUT stride'
    want="FAIL  bf16_gemv n_in=2560 n_out=512" ;;
  s-gemv-q8k-flag-flip)
    # `s_gemv_q8k_split` and `s_gemv_q8_0_split` drive ONE shader; the activation KIND is the push constant's
    # `q8k`.  Hardwiring it to 0 reads a Q8_K buffer (292 B / 256 elems) as Q8_0 (34 B / 32) - the scale sits at
    # a different offset and the block stride is wrong, so every Q8_K arm moves.  ENGINE-side backend source, so
    # the script rebuilds the gate (which links vulkan/src/kernels/matvec_vk.cpp).
    file="$TREE/vulkan/src/kernels/matvec_vk.cpp"
    old=$'        group_shift, form.has_offset ? 1 : 0, q8k ? 1 : 0};'
    new=$'        group_shift, form.has_offset ? 1 : 0, 0};   // INJECTION: the Q8_K activation kind hardwired to Q8_0'
    want="FAIL  s_gemv_q8k_split entry" ;;
  s-gemv-q8-split-wrong-act-block)
    # The activation BLOCK STRIDE the shader hoists out of its 16 loads: Q8_K is 292 B / 256 elems, Q8_0 is 34 B /
    # 32.  Using the Q8_0 stride for the Q8_K image reads every block's `d` from the wrong byte, which moves the
    # K-quant arms and nothing else - the shader-side falsification of the pair.
    file="$SH/s_gemv_q8_split.comp"; spv="s_gemv_q8_split"
    old=$'        const uint blk_bytes = (pc.q8k != 0) ? 292u : 34u;\n        const uint xb = (i / blk_elems) * blk_bytes;'
    new=$'        const uint blk_bytes = 34u;   // INJECTION: the Q8_K block stride read as Q8_0\n        const uint xb = (i / blk_elems) * blk_bytes;'
    want="FAIL  s_gemv_q8k_split entry" ;;
  shared-expert-silu-on-up)
    # The documented trap the source settles: SILU GOES ON THE GATE, not on `up`.  Swapping the operands keeps
    # every shape and produces a plausible number; the case's rival-margin arm proves this moves the reference.
    # ENGINE-side backend source -> the script rebuilds the gate.
    file="$TREE/vulkan/src/kernels/shared_expert_vk.cpp"
    old=$'    strata::vulkan::swiglu_f32(s, gate, up, gate, n_ff);'
    new=$'    strata::vulkan::swiglu_f32(s, up, gate, gate, n_ff);   // INJECTION: SILU on UP instead of GATE'
    want="FAIL  shared_expert entry" ;;
  fused-gr-supported-false)
    # THE CAPABILITY'S OWN falsification, in the same form as the other `native-caps-*-false` injections.  This
    # batch WIRES the fused hyper-connection read (`fused_gr_vk.cpp`, four shaders) and the backend's answer must
    # BE the engine's geometry predicate (fused_gr.cu:1164-1166).  The lie the invariant must catch now is the
    # answer going FALSE while the kernels exist - which is exactly the state that made `verify.cpp:336`'s first
    # disjunct refuse a native pack's only decode path.  (The pre-batch injection that answered the rule TRUE
    # while the kernels were MISSING was the lie THEN; it is the truth NOW and is retired here.)
    file="$TREE/vulkan/src/kernels/ple_vk.cpp"
    old=$'    return n_embd == 2560 && hc == 4 && hc_lr == 320;'
    new=$'    (void) n_embd; (void) hc; (void) hc_lr; return false;   // INJECTION: the fused kernels exist but the capability answers FALSE'
    want="FAIL  fused_gr_supported entry" ;;
  fused-gr-rs-drop-fold)
    # `fused_gr_rs` folds the previous half's write into R'.  Dropping the fold (ap forced false) makes the read
    # a plain normalise - and also stops `R_out` being written.  The case's rule arm uses an apply==true token,
    # whose R' differs from R by ~gw*bo, so every output moves.
    file="$SH/fused_gr_rs.comp"; spv="fused_gr_rs"
    old=$'    const bool ap = pc.apply != 0;'
    new=$'    const bool ap = false;   // INJECTION: the fold dropped'
    want="FAIL  fused_gr_read entry: wrapper vs the engine's own rule" ;;
  fused-gr-mix-sum-not-mean)
    # The fused gate+mean's last step is the MEAN over the streams.  Dropping the `/ hc` leaves a SUM - a factor
    # of hc that reads as a scale problem.  The case host-checks that the sum reading moves `mixed`.
    file="$SH/fused_gr_mix.comp"; spv="fused_gr_mix"
    old=$'    mx.v[d] = sum / float(pc.hc);              // the MEAN, not the sum'
    new=$'    mx.v[d] = sum;   // INJECTION: the /hc dropped - a SUM over the streams'
    want="FAIL  fused_gr_read entry: wrapper vs the engine's own rule" ;;
  fused-gr-down-swap-halves)
    # The port reads each BF16 weight row as 32-bit PAIRS: element 2p in the LOW half, 2p+1 in the HIGH half.
    # Swapping them is the plausible slip and is O(1) on independent random weights.
    file="$SH/fused_gr_down.comp"; spv="fused_gr_down"
    old=$'        acc += (rf.v[i0] * wn.v[i0] * rsb.v[c0]) * bf16_to_f32(packed & 0xFFFFu);\n        acc += (rf.v[i1] * wn.v[i1] * rsb.v[c0]) * bf16_to_f32(packed >> 16u);'
    new=$'        acc += (rf.v[i0] * wn.v[i0] * rsb.v[c0]) * bf16_to_f32(packed >> 16u);   // INJECTION: pair halves swapped\n        acc += (rf.v[i1] * wn.v[i1] * rsb.v[c0]) * bf16_to_f32(packed & 0xFFFFu);'
    want="FAIL  fused_gr_read entry: wrapper vs the engine's own rule" ;;
  fused-gr-inject-drop-rs)
    # The inject dot reads the SAME activation the up projection does: `R' * w_norm * rs`.  Dropping the `rs[c]`
    # factor reads the un-normalised residual - a pure scale error, invisible to any shape check.
    file="$SH/fused_gr_inject.comp"; spv="fused_gr_inject"
    old=$'        acc += (rf.v[i0] * wn.v[i0] * rsb.v[cc]) * bf16_to_f32(packed & 0xFFFFu);\n        acc += (rf.v[i1] * wn.v[i1] * rsb.v[cc]) * bf16_to_f32(packed >> 16u);'
    new=$'        acc += (rf.v[i0] * wn.v[i0]) * bf16_to_f32(packed & 0xFFFFu);   // INJECTION: rs dropped\n        acc += (rf.v[i1] * wn.v[i1]) * bf16_to_f32(packed >> 16u);'
    want="FAIL  fused_gr_read entry: wrapper vs the engine's own rule" ;;
  fused-gr-multi-token0-args)
    # `fused_gr_read_multi` iterates the window's tokens, each with ITS OWN R/bo/inj and its own outputs.  Reading
    # token 0's arguments for every token is exactly the bug a shared-weight kernel invites - and the case's
    # three-token bitwise arm (token 2 has different weights than token 0's apply/inj) must catch it.
    file="$TREE/vulkan/src/kernels/fused_gr_vk.cpp"
    old=$'    for (int t = 0; t < n_tok; ++t) strata::vulkan::fused_gr_read_one(*s, a[t]);'
    new=$'    for (int t = 0; t < n_tok; ++t) strata::vulkan::fused_gr_read_one(*s, a[0]);   // INJECTION: every token reads token 0 args'
    want="FAIL  fused_gr_read_multi entry: 3 tokens == 3 x fused_gr_read, BITWISE" ;;
  fused-gr-check-records-staged)
    # `fused_gr_check` is the hyper-connection read's CARD CHARACTERISATION (no tensors).  On this backend the
    # ported plain read is what runs (`fused_gr_supported()` false), so the honest recorded variant is the plain
    # one.  Recording a FUSED variant the backend has no shader for is the "card that does not exist" a wrong
    # probe would report - and the case must FAIL.
    file="$TREE/vulkan/src/kernels/ple_vk.cpp"
    old=$'    g_hc_variant[0].store(kHcPlain);'
    new=$'    g_hc_variant[0].store(kHcStaged);   // INJECTION: record a fused variant the backend does not run'
    want="FAIL  fused_gr_check: the check records the plain read" ;;
  copy-indexed-ignore-index)
    # `copy_indexed` selects the source ROW from DEVICE memory (`idx = *index`, then `src[idx*stride + i]`).
    # Reading the row at the POSITION (src[i]) is the rival the case's two-index arm pins.
    file="$SH/copy_indexed.comp"; spv="copy_indexed"
    old=$'    dst.v[i] = src.v[row * pc.stride + int(i)];'
    new=$'    dst.v[i] = src.v[int(i)];   // INJECTION: the device index ignored'
    want="FAIL  copy_indexed entry: device index 3 selects row 3" ;;
  copy-rows-never-zero)
    # `copy_rows_from_mapped` must ZERO the GPU's own rows and copy the rest from the mapped source.  Never
    # zeroing leaves the GPU's rows to be added twice by the later `moe_hit_add` - the `moved` arm pins it.
    file="$SH/copy_rows_from_mapped.comp"; spv="copy_rows_from_mapped"
    old=$'        if (hit) { dst.v[o] = 0.0; dst.v[o + 1] = 0.0; dst.v[o + 2] = 0.0; dst.v[o + 3] = 0.0; }'
    new=$'        if (false) { dst.v[o] = 0.0; dst.v[o + 1] = 0.0; dst.v[o + 2] = 0.0; dst.v[o + 3] = 0.0; }   // INJECTION: the hit rows are never zeroed'
    want="FAIL  copy_rows_from_mapped entry: hit rows -> 0" ;;
  verify-seam-no-boundary)
    # `wait_flag_ge` under capture records a HOST BOUNDARY; WITHOUT it the window is ONE submission and the ops
    # after the wait run at launch - the exact wrong-token shape the seam exists to prevent.  The case's CUT arm
    # (the sentinel read before the host raises the flag) must catch it.
    file="$TREE/vulkan/src/kernels/verify_vk.cpp"
    old=$'        s.ctx->capture_boundary(flag, value);'
    new=$'        (void) 0;   // INJECTION: the boundary dropped - the recording is not cut'
    want="FAIL  verify seam: the ops AFTER the boundary did NOT run" ;;
  graph-drop-last-node)
    # THE CUDA GRAPH API (this batch).  A capture RECORDS the dispatches a body issues; dropping the LAST one is
    # the plausible "off by one node" a hand-rolled recorder ships with.  The recorded step then replays 5 of 6
    # silu passes, so the replay cannot equal direct execution - and the node count moves too.
    file="$TREE/vulkan/src/device/vk_compute.cpp"
    old=$'    encode_dispatch(rec_cb_, pipe, bufs, push, push_bytes, groups, groups_y, /*chain_barrier=*/true,\n                    /*fresh_set=*/true);\n    ++recorded_;'
    new=$'    if (capture_ && recorded_ == 5) return;   // INJECTION: the capture drops its last node\n    encode_dispatch(rec_cb_, pipe, bufs, push, push_bytes, groups, groups_y, /*chain_barrier=*/true,\n                    /*fresh_set=*/true);\n    ++recorded_;'
    want="FAIL  cuda graph: replay == direct execution" ;;
  graph-replay-stale)
    # A replay that does NOT re-submit the recording hands back a STALE answer: the buffer keeps whatever the
    # host last wrote (here the new input), instead of the recorded dispatches' output.  That is the "graph
    # replays stale arguments" hazard, and it must move the live-input arm.
    file="$TREE/vulkan/src/compat/cuda_runtime.cpp"
    old=$'    exec->owner->submit_owned(exec->cb, exec->fence);   // the port\'s own submit + fence, the same path a replay uses'
    new=$'    static bool injected_first = false;   // INJECTION: only the first launch submits; later replays are stale\n    if (injected_first) { g_last = cudaSuccess; return cudaSuccess; }\n    injected_first = true;\n    exec->owner->submit_owned(exec->cb, exec->fence);'
    want="FAIL  cuda graph: host mutations between replays are SEEN" ;;
  graph-exec-destroy-leak)
    # A leaked instantiation: `cudaGraphExecDestroy` drops the recording instead of destroying it.  The arena is
    # untouched (the leak is command buffers + fences), so the leak is observed by the backend's own live-recording
    # counter, which must return to zero.
    file="$TREE/vulkan/src/compat/cuda_runtime.cpp"
    old=$'    if (exec->cb != VK_NULL_HANDLE && exec->owner != nullptr)\n        exec->owner->destroy_owned(exec->cb, exec->fence);\n    delete exec;'
    new=$'    // INJECTION: the instantiation is leaked (its recording is never destroyed)\n    delete exec;'
    want="FAIL  cuda graph: no instantiation left live" ;;
  doorbell-ring-captured-literal)
    # THE ENGINE'S OWN REQUIREMENT (layer.cpp:383-389): the ring must RE-READ its host copy, because "a captured
    # literal would ring the same number forever and the host would never see a change".  This injection restores
    # the capture-time HOST raise (the port's old defect): the ring moves during capture and the recorded block
    # never moves it again.  The case's "capture records, does not run" arm must see the ring already != 0.
    file="$TREE/vulkan/src/kernels/doorbell_vk.cpp"
    old=$'    s.ctx->dispatch(p, {&v}, &pc, (uint32_t) sizeof(pc), /*groups=*/1);'
    new=$'    if (s.ctx->capturing()) {   // INJECTION: the ring raised on the HOST at capture - a captured literal\n        uint32_t cur = 0;\n        s.ctx->read(v, &cur, sizeof(uint32_t), v.offset);\n        const uint32_t next = store ? value : cur + 1u;\n        s.ctx->write(v, &next, sizeof(uint32_t), v.offset);\n        return;\n    }\n    s.ctx->dispatch(p, {&v}, &pc, (uint32_t) sizeof(pc), /*groups=*/1);'
    want="FAIL  doorbell ring: capture records, does not run" ;;
  doorbell-ring-shader-literal)
    # THE SHADER side of the same lie: the increment becomes a STORE of the push-constant (a captured literal),
    # so the ring reads 0 forever.  This bites on the DIRECT arm first - the recorded arms below cannot pass if
    # the shader cannot count at all.
    file="$SH/ring_inc.comp"; spv="ring_inc"
    old=$'        ring.seq = (pc.store != 0u) ? pc.value : (ring.seq + 1u);'
    new=$'        ring.seq = pc.value;   // INJECTION: a captured literal, not the re-read increment'
    want="FAIL  doorbell ring: direct increments the ring" ;;
  doorbell-wait-records-node)
    # THE OTHER FORBIDDEN THING: a WAITING KERNEL inside a captured block.  The contract is that the host writes
    # the answer, THEN submits the consumer, so `doorbell_wait` must record NOTHING under capture.  This
    # injection makes it record a device node there; the case's "records NOTHING under capture" arm must fail
    # (a capture holding only the wait then produces a graph instead of refusing).
    file="$TREE/vulkan/src/kernels/doorbell_vk.cpp"
    old=$'    if (s->ctx->capturing()) return;                    // no node, no read, no refusal: the host answers later'
    new=$'    if (s->ctx->capturing()) { strata::vulkan::ring_raise(*s, const_cast<uint32_t*>(d_seq), false, 0, "doorbell_wait/INJECTION"); return; }   // INJECTION: a recorded device node where the wait must submit nothing'
    want="FAIL  doorbell_wait: records NOTHING under capture" ;;
  stream-null-not-default)
    # THE STREAM SEAM (generate.cpp:3893): a NULL handle is CUDA's default stream, and the engine passes it on the
    # DEFAULT single-token path.  This injection restores the old refusal, so the whole decode refuses at its
    # first op.  The case runs the null-handle call in a CHILD (a refusal exits 2), so this FAILS the verdict
    # rather than aborting the gate.
    file="$TREE/vulkan/src/device/vk_arena.cpp"
    old=$'    if (stream == nullptr) return default_stream();'
    new=$'    if (stream == nullptr) return nullptr;   // INJECTION: a null handle refused instead of resolving to the default stream'
    want="FAIL  null stream: cuda's nullptr IS the default stream" ;;
  native-k-q6-byte-sub)
    # Q6_K centres its 6-bit codes by `__vsubss4(v, 0x20202020)` - a PER-BYTE subtract.  A plain 32-bit
    # subtract is the plausible wrong reading (it borrows across a byte boundary).  Must FAIL the Q6_K arm.
    file="$SH/common/k_dots.glsl"; spv="native_k_mmvq"; comp="$SH/native_k_mmvq.comp"
    old=$'        uint viu = 0u;\n        for (uint b = 0u; b < 4u; ++b) {\n            const int bytev = int((vi0 >> (8u * b)) & 0xFFu) - 32;\n            viu |= (uint(bytev) & 0xFFu) << (8u * b);\n        }'
    new=$'        const uint viu = vi0 - 0x20202020u;   // INJECTION: a 32-bit subtract instead of the per-byte one'
    want="FAIL  native_k_mmvq (ty=14 Q6_K" ;;
  native-k-q4-ql-offset)
    # Q4_K's two `qs` words sit at `16*bq8_offset + 4*nib`.  Reading them at `2*nib` is the plausible
    # "step-4 like the IQ formats" mistake.  Must FAIL the Q4_K arm.
    file="$SH/common/k_dots.glsl"; spv="native_k_mmvq"; comp="$SH/native_k_mmvq.comp"
    old=$'    const uint ql = blk + 16u + 16u * bq8_offset + 4u * nib;   // bq4->qs (offset 16) + the source\'s expression'
    new=$'    const uint ql = blk + 16u + 16u * bq8_offset + 2u * nib;   // INJECTION: the Q4_K code nibble stride halved'
    want="FAIL  native_k_mmvq (ty=12 Q4_K" ;;
  native-mmvq-k-ty)
    # The generic K-quant shader takes the ggml type in a FIFTH push-constant field.  Sending a constant 12
    # (Q4_K) for every K-quant is the "one field, one value" mistake; the Q5_K/Q6_K arms must move.
    file="$TREE/vulkan/src/kernels/matvec_vk.cpp"
    old=$'            (int32_t) n_in, (int32_t) n_out, (int32_t) row_bytes, (int32_t) ncols, (int32_t) ggml_type};'
    new=$'            (int32_t) n_in, (int32_t) n_out, (int32_t) row_bytes, (int32_t) ncols, (int32_t) 12};   // INJECTION: the K-quant `ty` pinned to Q4_K'
    want="FAIL  native_mmvq entry (ty=13 Q5_K)" ;;
  native-expert-grouped-window)
    # THE LAUNCHER'S WINDOW LOOP.  The real pack's experts sit at 1.4 .. 24.8 GiB, so every expert past the
    # first 4 GiB is read through `win_id * WIN_BYTES`.  Running only window 0 leaves those groups unwritten -
    # a silently wrong expert.  Arm B (same expert through window 0 and window 1) must FAIL.
    file="$TREE/vulkan/src/kernels/native_expert_grouped_vk.cpp"
    old=$'    for (uint32_t w = 0; w < nwin; ++w) {'
    new=$'    for (uint32_t w = 0; w < nwin && w < 1; ++w) {   // INJECTION: only window 0 is dispatched'
    want="FAIL  native_expert_grouped: the SAME expert read through WINDOW 0 and WINDOW 1" ;;
  view-absolute-offset)
    # THE MEASURED `view()` DEFECT (2026-10-05).  `view(b, off)` used to SET `v.offset = off` instead of adding
    # the base buffer's offset, so every view of a bump-allocated buffer (non-zero arena offset) bound at the
    # ARENA BASE.  In `native_expert_grouped` that put the gate/up/h/hq scratch regions at the arena's first
    # ~8 KiB while the raw-pointer quantiser wrote to the caller's scratch - the expert's gate/up half never
    # reached the output.  This restores the defect; the launcher arm must FAIL.
    file="$TREE/vulkan/src/device/vk_compute.hpp"
    old=$'    v.offset = b.offset + off;'
    new=$'    v.offset = off;   // INJECTION: the view offset taken as arena-absolute (the measured defect)'
    want="FAIL  native_expert_grouped: the expert's GATE/UP half reaches the output" ;;
  # ==========================================================================================================
  # THIS BATCH: THE PREFILL PATH.  Each injection falsifies one wrapper's arithmetic on the REAL engine rule
  # (src/prefill/gemm.cu / kernels.cu), and each must FAIL the named `prefill ... entry` verdict.
  # ==========================================================================================================
  pf-gemm-fma-wrong-ldy)
    # The FMA GEMM's row stride: writing `row * pc.n` instead of `row * pc.ldy` smears the columns between n
    # and ldy - exactly the `ldy > N` arm the case runs (T=3 N=5 K=8 ldy=7).
    file="$SH/gemm_prefill_fma.comp"; spv="gemm_prefill_fma"
    old=$'    Y.y[row * pc.ldy + col] = acc;'
    new=$'    Y.y[row * pc.n + col] = acc;   // INJECTION: the row stride taken as n, not ldy'
    want="FAIL  prefill Gemm::f16 entry" ;;
  pf-gr-bcast-single-stream)
    # gr_broadcast writes the SAME e[t,d] into all HC streams; indexing e by the flat R index reads another
    # token's row for the streams past the first.
    file="$SH/pf_gr_bcast.comp"; spv="pf_gr_bcast"
    old=$'    rb.v[i] = eb.v[t * N + d];'
    new=$'    rb.v[i] = eb.v[i];   // INJECTION: the source indexed by the flat residual index'
    want="FAIL  prefill gr_broadcast entry" ;;
  pf-gr-norm-mean-hc)
    # The GR row scale is over ONE stream (n_embd values); dividing by HC*n_embd is the plausible wrong mean
    # and moves every scale by sqrt(HC) = 2x.
    file="$SH/pf_gr_norm.comp"; spv="pf_gr_norm"
    old=$'    const float rs = inversesqrt(wg_sum(ss) / float(N) + pc.eps);'
    new=$'    const float rs = inversesqrt(wg_sum(ss) / float(N * HC) + pc.eps);   // INJECTION: mean over the stack'
    want="FAIL  prefill gr_norm_rs entry" ;;
  pf-gdn-gates-drop-ssm-a)
    # `softplus(ab+dt) * ssm_a`: dropping ssm_a is the gdn_parity.cpp trap (the decay rate is data, not 1).
    file="$SH/pf_gdn_gates.comp"; spv="pf_gdn_gates"
    old=$'    gt.v[i] = (v > 20.0 ? v : log(1.0 + exp(v))) * sa.v[h];'
    new=$'    gt.v[i] = (v > 20.0 ? v : log(1.0 + exp(v)));   // INJECTION: ssm_a dropped'
    want="FAIL  prefill gdn_gates entry" ;;
  pf-gdn-l2-stride-cols)
    # The q/k rows live inside the chunk's C-wide rows: the stride is C, not S.  Reading at t*S + head*S
    # aliases a different token's channels and moves the L2.
    file="$SH/pf_gdn_l2.comp"; spv="pf_gdn_l2"
    old=$'    const uint base = t * C + head * S;'
    new=$'    const uint base = t * S + head * S;   // INJECTION: the row stride taken as S, not C'
    want="FAIL  prefill gdn_conv entry" ;;
  pf-swiglu16-drop-silu)
    # pf_swiglu16's rule is `hf_sat(silu(a) * u)` with silu(a) = a/(1+exp(-a)).  Dropping the silu leaves the raw
    # gate times up - every shape and magnitude "looks like" a SwiGLU output.  The case oracles against the host
    # rule, so it moves on both modes.
    file="$SH/pf_swiglu16.comp"; spv="pf_swiglu16"
    old=$'    o_b.v[i] = hf_sat(a / (1.0 + exp(-a)) * u);'
    new=$'    o_b.v[i] = hf_sat(a * u);   // INJECTION: the SiLU dropped'
    want="FAIL  prefill swiglu_pair" ;;
  pf-moe-combine-plain-shared)
    # The prompt combine SCALES the shared row by sigmoid(sg[t]); adding it PLAIN (the decode combine's rule) is
    # the plausible wrong reading the header warns about.  The oracle is the double rule, so it bites at any sg.
    file="$SH/pf_moe_combine.comp"; spv="pf_moe_combine"
    old=$'    bo_b.v[i] = s + sh_b.v[t * kN + d] * (1.0 / (1.0 + exp(-sg_b.v[t])));'
    new=$'    bo_b.v[i] = s + sh_b.v[t * kN + d];   // INJECTION: the shared row added PLAIN (no sigmoid)'
    want="FAIL  prefill moe_combine" ;;
  pf-split-q-wrong-stride)
    # q_full is [T,24,512]: the q half of head h starts at h*512.  Reading at h*256 (the OUTPUT stride) aliases
    # the previous head's gate half - correctly shaped, wrong content.
    file="$SH/pf_split_q.comp"; spv="pf_split_q"
    old=$'    q_b.v[i] = qf_b.v[t * 24 * 512 + h * 512 + d];'
    new=$'    q_b.v[i] = qf_b.v[t * 24 * 512 + h * 256 + d];   // INJECTION: the output stride used on the input'
    want="FAIL  prefill split_q" ;;
  pf-gate-attn-first-half)
    # The attention gate is the SECOND half of the 2*head_dim q_full block (the + 256).  Taking the FIRST half
    # (the q values) keeps every shape.
    file="$SH/pf_gate_attn.comp"; spv="pf_gate_attn"
    old=$'    const float g = qf_b.v[t * 24 * 512 + h * 512 + 256 + d];'
    new=$'    const float g = qf_b.v[t * 24 * 512 + h * 512 + d];   // INJECTION: the gate taken from the FIRST half'
    want="FAIL  prefill gate_attn" ;;
  pf-indexer-native-rotate-last)
    # The native append rotates the pooled key at the block's FIRST cell (`pos_base + R*b`); the LAST cell keeps
    # every shape and magnitude.  The case's host transcription rotates at the first cell, so this moves pool.
    file="$SH/pf_indexer_native.comp"; spv="pf_indexer_native"
    old=$'    const int rope_pos = (pos == 0) ? 0 : pc.pos_base + pc.r * b;'
    new=$'    const int rope_pos = (pos == 0) ? 0 : pc.pos_base + pc.r * b + pc.r - 1;   // INJECTION: rotate at the LAST cell'
    want="FAIL  prefill native_qsa_indexer_append" ;;
  pf-kv-append-step-off-by-one)
    # kv_f16_append reads the cell from `step`; +1 writes every token one cell late, which the row check catches.
    file="$SH/kv_f16_append.comp"; spv="kv_f16_append"
    old=$'    const int pos = STEP_.s[0];                      // kStepPos == 0 in qsa.hpp'"'"'s enum'
    new=$'    const int pos = STEP_.s[0] + 1;                  // INJECTION: the cell off by one'
    want="FAIL  prefill kv_append" ;;
  pf-gu-interleave-swap-roles)
    # The gate matrix goes to the EVEN rows (2r) and up to the odd (2r+1).  Swapping the roles keeps the shape.
    file="$SH/pf_gu_interleave_f16.comp"; spv="pf_gu_interleave_f16"
    old=$'    o_b.v[(2u * r) * ne + c]     = f16_from_f32_port(g_b.v[i]);\n    o_b.v[(2u * r + 1u) * ne + c] = f16_from_f32_port(u_b.v[i]);'
    new=$'    o_b.v[(2u * r) * ne + c]     = f16_from_f32_port(u_b.v[i]);   // INJECTION: gate/up roles swapped\n    o_b.v[(2u * r + 1u) * ne + c] = f16_from_f32_port(g_b.v[i]);'
    want="FAIL  pf_gu_interleave_f16" ;;
  verify-bcast-mode-ignored)
    # THIS BATCH: the verify window's embedding broadcast.  `mode` must SELECT the rule; making mode 1 emit the
    # plain broadcast leaves the add_streams arm with the mode-0 answer.
    file="$SH/bcast_streams.comp"; spv="bcast_streams"
    old=$'                                        : (a.v[tbase + i] + b.v[ebase + d]);  // h + e'
    new=$'                                        : emb;   // INJECTION: the add dropped (mode ignored)'
    want="FAIL  add_streams_broadcast entry" ;;
  verify-bcast-token-stride)
    # The R token stride: every token writing at token 0's base leaves the later tokens sentinelled.
    file="$SH/bcast_streams.comp"; spv="bcast_streams"
    old=$'    const uint tbase = t * total;'
    new=$'    const uint tbase = 0u;   // INJECTION: the R token stride dropped'
    want="FAIL  broadcast_streams entry" ;;
  verify-conv-tail-window)
    # `gdn_conv_commit`: the window is [hist | qkv_0 .. qkv_{n-1}]; dropping `n` reads the FIRST three entries.
    file="$SH/gdn_conv_tail.comp"; spv="gdn_conv_tail"
    old=$'        const int src = n + j;                           // index into [hist(3) | qkv...]'
    new=$'        const int src = j;   // INJECTION: the n offset dropped'
    want="FAIL  gdn_conv_commit entry" ;;
  verify-step-norm-ignore-nkeep)
    # `gdn_step_norm_multi`: the loop bound is DEVICE data; using n_tok commits every token of the window.
    file="$SH/gdn_step_norm_multi.comp"; spv="gdn_step_norm_multi"
    old=$'    const int lim = (pc.has_nkeep != 0) ? min(nk.v[0], pc.n_tok) : pc.n_tok;   // the DEVICE loop bound (uniform)'
    new=$'    const int lim = pc.n_tok;   // INJECTION: the DEVICE n_keep bound ignored'
    want="FAIL  gdn_step_norm_multi entry (commit half" ;;
  verify-step-norm-y-guard)
    # The `t_out_begin` guard: a split window must not write the earlier group's y rows.
    file="$SH/gdn_step_norm_multi.comp"; spv="gdn_step_norm_multi"
    old=$'        if (live && t >= pc.t_out_begin) {'
    new=$'        if (live) {   // INJECTION: the t_out_begin guard dropped'
    want="FAIL  gdn_step_norm_multi entry: \`t_out_begin\` suppresses" ;;
  verify-conv-l2-multi-writes-history)
    # THE ENGINE-SIDE PROPERTY: `gdn_conv_l2_multi` must not advance the caller's conv history (the verify half
    # may be REJECTED).  Sliding the caller's array keeps every OUTPUT right and breaks the contract.
    file="$TREE/vulkan/src/kernels/verify_vk.cpp"; spv=""
    old=$'        fused_gdn_conv_l2(work, qkv + row, conv_w, h + row, channels, qk_heads, eps, stream);'
    new=$'        fused_gdn_conv_l2(const_cast<float*>(history), qkv + row, conv_w, h + row, channels, qk_heads, eps, stream);   // INJECTION'
    want="FAIL  gdn_conv_l2_multi entry: the caller's HISTORY is NOT written" ;;
  verify-step-norm-state-in-place)
    # THE ENGINE-SIDE PROPERTY: the verify half must run on the working scratch, not in place.
    file="$TREE/vulkan/src/kernels/verify_vk.cpp"; spv=""
    old=$'    const int use_scratch = (n_keep == nullptr) ? 1 : 0;'
    new=$'    const int use_scratch = 0;   // INJECTION: the verify half runs in place'
    want="FAIL  gdn_step_norm_multi entry (verify half): the STATE is left untouched" ;;
  fetch-blobs-ignore-count)
    # `fetch_blobs` reads its blob count `*n` FROM DEVICE MEMORY; dropping the guard copies the CAPACITY, so the
    # arms past the real count move.  `case_blob_stage_entry` (A2) pins it with sentinel slots.
    file="$SH/fetch_blobs.comp"; spv="fetch_blobs"
    old=$'    if (g >= uint(n.v[0])) return;                       // the CUDA\'s `k < *n`'
    new=$'    // INJECTION: the device count is ignored (the capacity is copied)'
    want="FAIL  fetch_blobs entry: the DEVICE count caps the gather" ;;
  fetch-blobs-window-ignored)
    # The source is bound as a 4 GiB WINDOW and only the blobs whose window matches may be copied; dropping the
    # test copies a blob that lives in another window - the 4 GiB-index-limit class.
    file="$SH/fetch_blobs.comp"; spv="fetch_blobs"
    old=$'    if (w != uint64_t(pc.win_id)) return;                // another window owns this blob'
    new=$'    // INJECTION: the window test is dropped (every window copies every blob)'
    want="FAIL  fetch_blobs window: a blob in ANOTHER window is NOT copied" ;;
  rebase-ptrs-fixed-cap)
    # `rebase_ptrs` rewrites `ptr[k] = base + k*bytes` for `k < *n` (DEVICE data).  A FIXED cap of 8 rewrites
    # entries past `*n`, which the tail arm catches (the table carries sentinels past the count).
    file="$SH/rebase_ptrs.comp"; spv="rebase_ptrs"
    old=$'    if (k >= uint(n.v[0])) return;                                      // the CUDA\'s `if (k < *n)`'
    new=$'    if (k >= 8u) return;   // INJECTION: a fixed cap instead of the device count'
    want="FAIL  rebase_ptrs entry: entries >= *n are UNCHANGED" ;;
  copy-from-mapped-arena-binds-dst)
    # The `verify.cpp:678` PLE snapshot has an ARENA (device) source; this batch added the arena fallback.  This
    # injection binds the DESTINATION as the source - the copy becomes a no-op and the arm's oracle bites.
    file="$TREE/vulkan/src/kernels/elementwise_vk.cpp"; spv=""
    old=$'    if (!have_src) have_src = arena_resolve(s, src, (uint64_t) n * 4, sv);'
    new=$'    if (!have_src) { have_src = true; sv = dv; }   // INJECTION: the arena source binds the DESTINATION'
    want="FAIL  copy_from_mapped entry: a DEVICE (arena) source copies" ;;
  sample-tokens-mapped-out-wrong-view)
    # The verify window's sampler writes `out` into a MAPPED region (`verify.cpp:1170`, `m_out_`); this batch
    # accepts it.  This injection binds the LOGITS view as `out`, so the id never reaches the mapped buffer.
    file="$TREE/vulkan/src/kernels/sampler_vk.cpp"; spv=""
    old=$'    const bool ook = resolve_operand(out, (uint64_t) n_tokens * 4, ov);'
    new=$'    const bool ook = resolve_operand(out, (uint64_t) n_tokens * 4, ov);\n    if (ook) ov = lv;   // INJECTION: the mapped out binds the logits view'
    want="FAIL  sample_tokens entry (mapped out): the id lands" ;;
  *) echo "unknown injection '$name'"; exit 2 ;;
esac
COMPILE_TARGET="${comp:-$file}"   # an include cannot be compiled alone; its including shader is the target

# 1. the anchor must be there, or we would be "injecting" into a file that no longer says what we think.
if ! grep -qF -- "$old" "$file"; then
  echo "ANCHOR MISSED: $name - '$file' does not contain the anchored text; nothing changed"
  exit 2
fi

bak="$(mktemp)"
cp "$file" "$bak"
restore() {
  cp "$bak" "$file"; rm -f "$bak"
  if is_harness_src "$file"; then rebuild_harness 2>/dev/null; return; fi
  if [ "${file#"$SH/common/"}" = "$file" ]; then
    glslc --target-env=vulkan1.3 -fshader-stage=compute "$file" -o "$SH/$spv.spv" 2>/dev/null
  fi
  # if the changed file is a shared INCLUDE, every shader that includes it must be recompiled (the include itself
  # cannot be compiled: it has no #version)
  if [ "${file#"$SH/common/"}" != "$file" ]; then
    for c in "$SH"/*.comp; do
      grep -qF "$(basename "$file")" "$c" || continue
      glslc --target-env=vulkan1.3 -fshader-stage=compute "$c" -o "${c%.comp}.spv" 2>/dev/null
    done
  fi
}
trap restore EXIT

# 2. apply it, exactly once.
python3 - "$file" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
n = s.count(old)
if n != 1:
    sys.exit(f"ANCHOR COUNT {n} (expected 1)")
open(p, "w").write(s.replace(old, new))
PY
[ $? -eq 0 ] || { echo "ANCHOR MISSED: $name - anchor not unique"; exit 2; }

# 3. it must COMPILE - an injection that does not build is a stale binary wearing a new timestamp.
#    `$COMPILE_TARGET` is the shader that INCLUDES the changed file when the change is in common/ (an include has
#    no #version and cannot be compiled alone), and the changed file itself otherwise.  A harness source rebuilds
#    the harness instead.
if is_harness_src "$file"; then
  if ! rebuild_harness 2>"$BUILD/harness.builderr"; then
    echo "DID NOT COMPILE (harness): $name"; sed -n '1,12p' "$BUILD/harness.builderr"; exit 3
  fi
elif ! glslc --target-env=vulkan1.3 -fshader-stage=compute "$COMPILE_TARGET" -o "$SH/$spv.spv" 2>"$BUILD/$spv.glslerr"; then
  echo "DID NOT COMPILE: $name"; sed -n '1,12p' "$BUILD/$spv.glslerr"; exit 3
fi
if [ "${file#"$SH/common/"}" != "$file" ]; then
  for c in "$SH"/*.comp; do
    grep -qF "$(basename "$file")" "$c" || continue
    glslc --target-env=vulkan1.3 -fshader-stage=compute "$c" -o "${c%.comp}.spv" 2>/dev/null || { echo "DID NOT COMPILE (dependent): $c"; exit 3; }
  done
fi

# 4. run the gate and require the named case to FAIL.
if [ -n "$icd" ]; then out="$(VK_ICD_FILENAMES="$icd" "$GATE" --spv-dir "$SH" 2>&1)"; else out="$("$GATE" --spv-dir "$SH" 2>&1)"; fi
line="$(grep -F -- "$want" <<<"$out" | head -1)"
if [ -n "$line" ]; then
  echo "FALSIFIED ($name): $line"
  exit 0
else
  echo "NOT FALSIFIED ($name): expected a line starting '$want' and did not get one"
  grep -E '^(PASS|FAIL|SKIP) ' <<<"$out" | grep -iF "${want#FAIL  }" | head -3
  exit 1
fi
