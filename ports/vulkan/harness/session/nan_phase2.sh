#!/usr/bin/env bash
# /home/bob/step4/nan_phase2.sh -- wait for nan_rate_sweep.sh to exit, then run nan_final.sh (nan5) on the freed card.
#   Refuses (with a diagnosis on stdout) rather than starting if the sweep is stuck or an engine orphans the lock.
set -u
OUT=/home/bob/step4/nan_phase2.out
: > "$OUT"
{
  echo "[$(date -Is)] phase2: waiting for nan_rate_sweep.sh to finish (budget 1500 s)"
  waited=0
  while pgrep -f 'bash nan_rate_sweep\.sh' >/dev/null; do
    if [ "$waited" -ge 1500 ]; then
      echo "[$(date -Is)] ABORT: nan_rate_sweep.sh still running after ${waited}s"
      pgrep -af 'bash nan_rate_sweep\.sh'; exit 3
    fi
    sleep 10; waited=$((waited + 10))
  done
  echo "[$(date -Is)] sweep gone after ${waited}s. tail:"; tail -4 /home/bob/step4/nan_rate_sweep.out
  # the ledger records an orphaned engine surviving a dead wrapper and holding /tmp/b70.lock: never kill it blind
  if pgrep -f 'bins/strata_vulka[n]' >/dev/null; then
    echo "[$(date -Is)] ABORT: an engine process is still alive (someone's arm, or an orphan):"
    pgrep -af 'bins/strata_vulkan'; exit 4
  fi
  if ! [ -x /home/bob/step4/bins/strata_vulkan.nan5 ]; then
    echo "[$(date -Is)] ABORT: bins/strata_vulkan.nan5 is missing"; exit 5
  fi
  echo "[$(date -Is)] card free; nan5 sha256=$(sha256sum /home/bob/step4/bins/strata_vulkan.nan5 | cut -c1-16)"
  echo "[$(date -Is)] starting nan_final.sh"
  bash /home/bob/step4/nan_final.sh
  rc=$?
  echo "[$(date -Is)] PHASE2 DONE rc=$rc"
  echo "--- nan_final.tsv ---"; cat /home/bob/step4/nan_final.tsv
} >> "$OUT" 2>&1
