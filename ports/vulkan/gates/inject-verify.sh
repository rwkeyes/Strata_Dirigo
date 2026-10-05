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
  g++ -std=c++20 -O2 -Wall -Wextra -Werror -I"$TREE/include" -o "$GATE" \
      "$ROOT/harness/vk_compute.cpp" "$ROOT/harness/vk_compat.cpp" "$ROOT/harness/vk_stack.cpp" \
      "$ROOT/harness/vk_gate.cpp" -lvulkan
}
is_harness_src() { case "$1" in "$ROOT"/harness/*.cpp|"$ROOT"/harness/*.hpp) return 0 ;; *) return 1 ;; esac; }

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
    # The kernel's distinguishing rule: the 12-byte `scales` packs SIX 6-bit scales and SIX 6-bit mins and `hi`
    # switches between groups 0..2 and 3..5.  Dropping it reads groups 0..2's fields for every group.
    file="$SH/native_q5_k_f32.comp"; spv="native_q5_k_f32"
    old=$'    const uint him = (j >= 2) ? 0xFFFFFFFFu : 0u;'
    new=$'    const uint him = 0u;   // INJECTION: the packed-scale half switch dropped'
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
  sampler-split-merge-drop-parts)
    # The split's whole content is the MERGE: a row's list is the ordered union of its 4096-logit partitions'
    # lists.  Taking the running list first (instead of comparing the two heads) appends each partition after
    # the first to the tail of what is already there, so everything past the first partition is lost - which the
    # multi-partition arms (12288, 248320) must catch and a one-partition row cannot.
    file="$SH/common/sampler_select.glsl"; spv="sampler_split"; comp="$SH/sampler_split.comp"
    old=$'                else take_a = (sc_sel_lg[a] > sc_tmp_lg[c]) ||\n                              (sc_sel_lg[a] == sc_tmp_lg[c] && sc_sel_id[a] < sc_tmp_id[c]);'
    new=$'                else take_a = (a < ncur);   // INJECTION: the merge drains the running list first'
    want="FAIL  sampler_split:" ;;
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
