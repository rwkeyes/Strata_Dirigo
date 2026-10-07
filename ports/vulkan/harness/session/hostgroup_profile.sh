#!/usr/bin/env bash
# /home/bob/step4/hostgroup_profile.sh -- SPLIT THE `host grouping` PHASE (1,524 ms, 22.9% of the prefill).
#
# WHY.  `host grouping` did NOT shrink when the GPU phases did (1,256 ms when the prefill was 9,127 ms; 1,524 ms
# when it is 6,650 ms), so it is host CPU work on the critical path - and the phase timer says only "chunk setup
# (PLE rows, the expert stream plan)".  The question this answers: is that time in the ENGINE's plan building, in
# the PORT's API (mapped-region resolution, per-expert weight lookups, arena bookkeeping), or in libc/libstdc++
# (allocation, string building)?  The fix is different in each case - and if it is engine code, the port does not
# fork it, which is itself an answer.
#
# METHOD.  `perf record` the engine for the prefill window only.  perf_event_paranoid=4 on this box, so it needs
# the passwordless sudo bob has.  The attach is late by ~2 s on purpose: the model load and the pack scan are not
# the phase under study, and perf's own startup must not land in the window either.
set -u
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
T199="$(printf '1 %.0s' $(seq 1 199))"
OUT=/home/bob/step4/hostgroup.prof
LOG=/home/bob/step4/logs/hgprof_1.log
: > "$OUT"
echo "== hostgroup_profile $(date -Is)"
echo "== launching the arm in the background; perf will attach once the prefill starts"
( env STRATA_VK_BIN="$BIN" STRATA_PREFILL_TIMING=1 timeout 900 flock /tmp/b70.lock -c \
    "bash /home/bob/step4/run.sh hgprof_1 \"$T199\"" >/dev/null 2>&1 ) &
LOOP=0
PID=""
while [ "$LOOP" -lt 60 ]; do
  PID=$(pgrep -f 'vkbuild-vulkan/vulkan/strata_vulka[n]' | head -1)
  [ -n "$PID" ] && break
  sleep 2; LOOP=$((LOOP+1))
done
[ -n "$PID" ] || { echo "  engine never started"; exit 2; }
echo "  engine pid=$PID; waiting for the prompt phase to begin"
sleep 8
echo "  perf record (sudo) for up to 30 s ..."
sudo -n timeout 30 perf record -F 1999 -g --call-graph dwarf -o "$OUT" -p "$PID" >/dev/null 2>&1 || \
  sudo -n timeout 30 perf record -F 999 -g -o "$OUT" -p "$PID" >/dev/null 2>&1 || echo "  perf record FAILED"
wait
echo
echo "== the arm's own numbers =="
grep -m1 '^prefill ' "$LOG" | sed 's/^/  /'
grep -m1 'strata prefill timing:' "$LOG" | sed -n 's/.*ms: //p' | tr ' ' '\n' | paste - - - 2>/dev/null | grep -iE 'host|grouping|embed' | sed 's/^/  /'
echo
echo "== the top host frames during the prompt (self time) =="
sudo -n perf report -i "$OUT" --stdio --no-children -g none --percent-limit 1 2>/dev/null | grep -vE '^\s*$|^#' | head -22
echo "== DONE $(date -Is)"
