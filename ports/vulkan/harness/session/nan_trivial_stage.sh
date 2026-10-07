#!/usr/bin/env bash
# /home/bob/step4/nan_trivial_stage.sh -- WHICH HEAD STAGE GOES NON-FINITE FIRST when the work is trivialised.
#   This is the one measurement the retired "machinery ceiling" question still needs: nan_trivial_3 (the
#   STRATA_VK_TRIVIAL_REC=1 arm that produces d01eee6a3948) ran on nan2, whose poison self-test never fired, so its
#   detector is unvalidated.  Run the SAME arm on nan5 (controls proven to fire) and read the stage.
#   Expected: a stage named; that stage is where trivialised work first produces a non-finite value.
set -u
LOGD=/home/bob/step4/logs
BIN=/home/bob/step4/bins/strata_vulkan.nan5
T8="1 2 3 4 5 6 7 8"
echo "== nan_trivial_stage $(date -Is)  BIN=$(sha256sum "$BIN" | cut -c1-16)"
for r in 1 2; do
  name="nan_triv5_$r"
  echo "[$(date '+%H:%M:%S')] ARM $name (STRATA_VK_TRIVIAL_REC=1, STRATA_DBG_NAN=1)"
  env STRATA_VK_BIN="$BIN" STRATA_VK_TRIVIAL_REC=1 STRATA_DBG_NAN=1 timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T8\"" || true
  L="$LOGD/$name.log"
  grep -m1 'instrument ARMED' "$L" | sed 's/^/  /' || true
  grep -m1 'FIRST NON-FINITE STAGE' "$L" | sed 's/^/  /' || echo "  (no stage named -- all monitored buffers FINITE)"
  echo "  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)  $(grep -m1 '^decode ' "$L")"
  grep -m1 ARM_DEGENERATE "$L" | sed 's/^/  /' || true
done
echo "== nan_trivial_stage DONE $(date -Is)"
