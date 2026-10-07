#!/usr/bin/env bash
# /home/bob/step4/prefill_phase_chain.sh -- wait for nan_final.sh to release the card, then run the prefill phase
# diagnostic on the SAME build.  One writer, one GPU contender; refuses rather than racing.
set -u
OUT=/home/bob/step4/prefill_phase.out
: > "$OUT"
{
  echo "[$(date -Is)] chain: waiting for nan_final.sh to exit (budget 1500 s)"
  waited=0
  while pgrep -f 'nan_final.s[h]' >/dev/null; do
    [ "$waited" -ge 1500 ] && { echo "[$(date -Is)] ABORT: nan_final.sh still running after ${waited}s"; exit 3; }
    sleep 15; waited=$((waited + 15))
  done
  echo "[$(date -Is)] nan_final gone after ${waited}s. its table:"
  column -t /home/bob/step4/nan_final.tsv 2>/dev/null || true
  if pgrep -f 'step4/bins/strata_vulka[n]' >/dev/null; then
    echo "[$(date -Is)] ABORT: an engine is still alive (an orphan from nan_final):"
    pgrep -af 'step4/bins/strata_vulkan'; exit 4
  fi
  echo "[$(date -Is)] card free; starting the prefill phase diagnostic"
  REPS=2 PF_BIN=/home/bob/vkbuild-vulkan/vulkan/strata_vulkan bash /home/bob/step4/prefill_phase.sh
  echo "[$(date -Is)] CHAIN DONE rc=$?"
} >> "$OUT" 2>&1
