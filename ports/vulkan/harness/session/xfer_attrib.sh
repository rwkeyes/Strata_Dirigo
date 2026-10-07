#!/usr/bin/env bash
# /home/bob/step4/xfer_attrib.sh -- ATTRIBUTE THE 13,752 TRANSFER SUBMITS TO CODE.
#
# The dispatch-layer attribution (step4/prefill_only_dispstat.sh) said the prefill issues 14,318 submits, of which
# 13,752 are the TRANSFER arm (563 are live-batch flushes), and the transfer arm alone pays ~366 us per submit.
# That is the port's biggest own-layer number.  The port already carries the instrument for the next question:
# STRATA_VK_XFER_STAT records every upload/download with its CALL SITE (a backtrace, module-relative offset) and
# splits it into pre-capture vs decode-phase counts.  This run turns "13,752 transfers" into "these N call sites".
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
T199="$(printf '1 %.0s' $(seq 1 199))"
L="$LOGD/xf1.log"
echo "== xfer_attrib $(date -Is)  bin=$(sha256sum "$BIN" | cut -c1-16)"
env STRATA_VK_BIN="$BIN" STRATA_VK_XFER_STAT=1 STRATA_VK_DISP_STAT=1 STRATA_PREFILL_TIMING=1 \
    timeout 900 flock /tmp/b70.lock -c "bash /home/bob/step4/run.sh xf1 \"$T199\"" || true
echo
echo "== the run's own numbers =="
grep -m1 '^prefill ' "$L" | sed 's/^/  /'
grep -m1 'vk disp stat ms:' "$L" | sed 's/^/  /'
echo
echo "== transfers, split pre-capture vs decode =="
grep 'vk xfer stat' "$L" | sed 's/^/  /'
echo
echo "== the call sites, resolved against THIS binary =="
grep 'vk xfer site' "$L" | while IFS= read -r line; do
  n=$(echo "$line" | sed -n 's/.*n=\([0-9]*\).*/\1/p')
  key=$(echo "$line" | sed -E 's/^[^:]*: *n=[0-9]+ +bytes=[0-9]+ +dec_n=[0-9]+ +dec_bytes=[0-9]+ +//')
  printf '  n=%-7s ' "$n"
  # the key is a chain of module+offset frames; resolve each 0x... / hex offset with addr2line
  echo "$key" | grep -oE '0x[0-9a-fA-F]+|[0-9a-fA-F]{6,16}' | while read -r off; do
    sym=$(addr2line -f -C -e "$BIN" "$off" 2>/dev/null | paste -sd' ' -)
    [ -n "$sym" ] && printf '%s | ' "$sym"
  done
  echo
  echo "     raw: $key"
done
echo "== DONE $(date -Is)"
