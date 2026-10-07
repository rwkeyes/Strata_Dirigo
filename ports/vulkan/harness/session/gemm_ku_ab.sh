#!/usr/bin/env bash
# /home/bob/step4/gemm_ku_ab.sh -- the GEMM K-unroll A/B, interleaved INSIDE one launch, ids checked.
#
#   A (baseline): shaders/gemm_prefill_fma_small.spv = the shipped pre-unroll binary
#   B (KU=8)    : the same file with the compile-time K unroll (default `#define KU 8`)
#
# The variants differ ONLY in that one binary, and its sha is printed for every arm so the swap cannot be
# mistaken for noise.  The engine loads shaders at runtime, so no rebuild is involved.
set -u
LOGD=/home/bob/step4/logs
REPO=/home/bob/strata-vulkan-wt
SPVD=$REPO/ports/vulkan/shaders
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
BASE_ID=56a0b28d2de6
ROUNDS=${ROUNDS:-2}
T199="$(printf '1 %.0s' $(seq 1 199))"
ABDIR=/home/bob/step4/ab-spv
[ -f "$ABDIR/base.spv" ] && [ -f "$ABDIR/ku8.spv" ] || { echo "REFUSES: $ABDIR must hold base.spv and ku8.spv"; exit 2; }
echo "== gemm_ku_ab $(date -Is)  BIN=$(sha256sum "$BIN"|cut -c1-16)  rounds=$ROUNDS"
TSV=/home/bob/step4/gemm_ku_ab.tsv; : > "$TSV"

one() {  # one <name> <which>
  local name="$1" which="$2"
  local L="$LOGD/$name.log"
  cp -f "$ABDIR/$which.spv" "$SPVD/gemm_prefill_fma_small.spv"
  local spvsha; spvsha=$(sha256sum "$SPVD/gemm_prefill_fma_small.spv" | cut -c1-16)
  echo "[$(date '+%H:%M:%S')] ARM $name  ($which, spv=$spvsha)"
  env STRATA_VK_BIN="$BIN" STRATA_PREFILL_TIMING=1 timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" || true
  local id gu d rec tac
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  gu=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gemm gate\/up \([0-9][0-9]*\) .*/\1/p')
  d=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gemm down \([0-9][0-9]*\) .*/\1/p')
  rec=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gdn recurrence \([0-9][0-9]*\) .*/\1/p')
  tac=$(grep -m1 '^prefill ' "$L" | sed 's/.*tokens in \([0-9.]*\) ms.*-> *\([0-9.]*\) tok.*/\1|\2/')
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  if [ -n "$id" ] && [ "$id" != "$BASE_ID" ]; then echo "  ID_MOVED id=$id (baseline $BASE_ID)"; else echo "  id=$id"; fi
  printf '  gemm_gate_up=%s ms  gemm_down=%s ms  gdn_rec=%s ms  prefill=%s ms  %s tok/s\n' \
    "${gu:-NA}" "${d:-NA}" "${rec:-NA}" "$(echo "$tac" | cut -d'|' -f1)" "$(echo "$tac" | cut -d'|' -f2)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$which" "$id" "${gu:-NA}" "${d:-NA}" "$(echo "$tac" | cut -d'|' -f1)" "$(echo "$tac" | cut -d'|' -f2)" >> "$TSV"
}

for r in $(seq 1 "$ROUNDS"); do
  one "gk_A_$r" base
  one "gk_B_$r" ku8
done
cp -f "$ABDIR/ku8.spv" "$SPVD/gemm_prefill_fma_small.spv"   # leave the SHIPPED default (the win) in place
echo
printf '%-8s %-6s %-14s %-12s %-12s %-12s %s\n' arm variant id gemm_gate_up gemm_down prefill tok_s
awk -F'\t' '{printf "%-8s %-6s %-14s %-12s %-12s %-12s %s\n", $1, $2, $3, $4, $5, $6, $7}' "$TSV"
echo "== gemm_ku_ab DONE $(date -Is)  (shaders/ left with the KU=8 build)"
