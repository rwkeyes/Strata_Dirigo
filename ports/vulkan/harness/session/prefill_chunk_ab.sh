#!/usr/bin/env bash
# /home/bob/step4/prefill_chunk_ab.sh -- the CHUNK recurrence A/B, same binary, ONE env var, interleaved.
#
#   A (the shipped chain)  : STRATA_PREFILL_TIMING=1                     -> T step dispatches + T norm dispatches
#   B (the chunk walk)     : STRATA_PREFILL_TIMING=1 STRATA_PF_GDN_REC_CHUNK=1 -> 1 step dispatch + T norm
#
# WHAT MUST HOLD FOR B TO BE QUOTABLE:
#   * the output id must be 56a0b28d2de6 (the port's 199-token baseline) on EVERY arm - the chunk walk claims
#     the SAME arithmetic, element for element and in the same order, so an id change REFUTES that claim and is
#     a finding, not a tolerance to widen.  The script reports it as ID_MOVED and keeps going so the phase
#     numbers are still readable.
#   * run.sh's own screen must not flag an arm degenerate/absent.
# The phase table is a WALL-CLOCK attribution (see prefill_phase.sh's note); read it as such.
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
BASE_ID=56a0b28d2de6
T199="$(printf '1 %.0s' $(seq 1 199))"
[ -x "$BIN" ] || { echo "CHUNK_AB REFUSES: $BIN is missing"; exit 2; }
echo "== prefill_chunk_ab $(date -Is)  BIN=$BIN sha=$(sha256sum "$BIN"|cut -c1-16)  n=2 per arm, interleaved"

one() {  # one <name> <env-prefix>
  local name="$1" envs="$2"
  local L="$LOGD/$name.log"
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$envs]"
  env STRATA_VK_BIN="$BIN" $envs timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" || true
  local id; id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  local tac recurrent total
  tac=$(grep -m1 '^prefill ' "$L")
  # NOTE: phase names contain spaces ("gdn recurrence", "host grouping"), so pair-based parsing yields NA.
  # Extract by NAME with sed, and take the total from the line's own GPU timeline.
  recurrent=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gdn recurrence \([0-9][0-9]*\) (.*/\1/p')
  total=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*GPU timeline \([0-9][0-9]*\) ms.*/\1/p')
  if [ "$id" != "$BASE_ID" ] && [ -n "$id" ]; then echo "  ID_MOVED id=$id (baseline $BASE_ID)"; else echo "  id=$id"; fi
  echo "  $tac"
  echo "  gdn_recurrence_ms=${recurrent:-NA}"
  grep -m1 'strata prefill timing' "$L" | sed 's/.*ms:/  phases:/'
  printf '%s\t%s\t%s\t%s\n' "$name" "$id" "${recurrent:-NA}" "${tac:-NA}" >> /home/bob/step4/prefill_chunk_ab.tsv
}

: > /home/bob/step4/prefill_chunk_ab.tsv
for r in 1 2; do
  one "pxch_A_$r" "STRATA_PREFILL_TIMING=1"
  one "pxch_B_$r" "STRATA_PREFILL_TIMING=1 STRATA_PF_GDN_REC_CHUNK=1"
done
echo "== raw (arm, id, gdn_recurrence_ms, line) =="
cat /home/bob/step4/prefill_chunk_ab.tsv
echo "== prefill_chunk_ab DONE $(date -Is)"
