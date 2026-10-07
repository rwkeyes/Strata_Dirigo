#!/usr/bin/env bash
# /home/bob/step4/nan_rowshape.sh -- SETTLE THE CAUSE OF AN ALL-ZEROS DECODE, by measuring the logits row instead of
# inferring it from the sampler's -inf branch.
#
#   The record claimed "an all-zeros output IS non-finite head logits" (read off `sampler_greedy.comp:106`).  But that
#   shader answers index 0 in TWO cases: nothing beats -inf, OR a FINITE row whose lowest index holds the maximum (a
#   tied/constant row).  nan_trivial_4.log (nan5) produced the all-zeros id with the armed detector naming NO stage and
#   the S5 arm not firing -- i.e. the sampler's 0 was the argmax of the logits it read, all finite.
#
#   ARM 1 (valid, the control): the row must be healthy - equal-to-max = 1, argmax = the token the sampler answered.
#   ARM 2 (trivial work): if the cause is a degenerate FINITE row, equal-to-max will be huge (a constant row) or the
#   argmax will be 0 with a spread present.  Either way it prints the row's min/max so the claim is measured.
set -u
BIN=${RS_BIN:-/home/bob/step4/bins/strata_vulkan.nan6}
T8="1 2 3 4 5 6 7 8"
LOGD=/home/bob/step4/logs
echo "== nan_rowshape $(date -Is)  BIN=$BIN sha=$(sha256sum "$BIN" | cut -c1-16)"
one() {  # one <name> <extra env>
  local name="$1" envs="$2" L="$LOGD/$1.log"
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$envs]"
  env STRATA_VK_BIN="$BIN" STRATA_DBG_NAN=1 $envs timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T8\"" || true
  grep -m1 'strata dbg NAN: head row0' "$L" | sed 's/^/  /' || echo "  (no row-shape line)"
  grep -m1 'FIRST NON-FINITE STAGE' "$L" | sed 's/^/  /' || echo "  (no stage named - every monitored buffer finite)"
  echo "  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)  $(grep -m1 '^decode ' "$L")"
  grep -m1 'ARM_DEGENERATE' "$L" | sed 's/^/  /' || true
}
one rs_valid_1 ""
one rs_trivial_1 "STRATA_VK_TRIVIAL_REC=1"
echo "== nan_rowshape DONE $(date -Is)"
