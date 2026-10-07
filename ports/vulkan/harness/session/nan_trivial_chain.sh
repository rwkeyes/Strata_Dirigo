#!/usr/bin/env bash
# /home/bob/step4/nan_trivial_chain.sh -- after the prefill phase diagnostic releases the card, name the stage that
# first goes non-finite on the TRIVIAL-work arm (nan5, controls proven).  Two arms, ~4 min total.  One writer.
set -u
OUT=/home/bob/step4/nan_trivial_stage.out
: > "$OUT"
{
  echo "[$(date -Is)] chain2: waiting for prefill_phase.sh AND its chain wrapper to exit (budget 1200 s)"
  waited=0
  while pgrep -f 'prefill_phase.s[h]|prefill_phase_chai[n]' >/dev/null; do
    [ "$waited" -ge 1200 ] && { echo "[$(date -Is)] ABORT: prefill_phase still running after ${waited}s"; exit 3; }
    sleep 15; waited=$((waited + 15))
  done
  echo "[$(date -Is)] prefill_phase gone after ${waited}s"
  if pgrep -f 'step4/bins/strata_vulka[n]' >/dev/null; then
    echo "[$(date -Is)] ABORT: an engine is still alive:"; pgrep -af 'step4/bins/strata_vulkan'; exit 4
  fi
  bash /home/bob/step4/nan_trivial_stage.sh
  echo "[$(date -Is)] CHAIN2 DONE rc=$?"
} >> "$OUT" 2>&1
