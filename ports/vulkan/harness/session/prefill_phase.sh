#!/usr/bin/env bash
# /home/bob/step4/prefill_phase.sh -- WHERE THE PREFILL'S WALL CLOCK GOES (the prompt path's own phase timer).
#
#   STRATA_PREFILL_TIMING=1 makes the ENGINE print, once per prompt:
#     strata prefill timing: N tokens, GPU timeline X ms, wall Y ms, host staging Z ms: <phase> ms (%) ...
#     strata prefill timing: host: chunk setup .. ms, waiting for each chunk .. ms, after each chunk .. ms, PLE .. ms
#   On this port the event pair is the compat shim's HOST WALL CLOCK, and the engine charges each gap to the phase
#   that was WAITING, so the table is a wall-clock attribution - read it that way, and never as GPU time (that is
#   what STRATA_VK_KERNEL_TIME=1 is for, and its own cost must be stated).
#
#   Arms: the 199-token prompt (the arm results are REPORTED from), n=REPS, one config per invocation, every GPU
#   step under flock.  Baseline ids: 56a0b28d2de6.  A BOUND/degenerate arm is refused by run.sh's own screen.
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
REPS=${REPS:-2}
KERNEL_TIME=${KERNEL_TIME:-1}      # one extra arm at the end with STRATA_VK_KERNEL_TIME=1 (per-family GPU time)
T199="$(printf '1 %.0s' $(seq 1 199))"
[ "$(echo $T199 | wc -w)" -eq 199 ] || { echo "PREFILL_PHASE REFUSES: id list is $(echo $T199|wc -w), not 199"; exit 2; }
[ -x "$BIN" ] || { echo "PREFILL_PHASE REFUSES: $BIN is missing"; exit 2; }
echo "== prefill_phase $(date -Is)  BIN=$BIN sha=$(sha256sum "$BIN"|cut -c1-16)  REPS=$REPS"

one() {  # one <name> <env-prefix-string>
  local name="$1" envs="$2" L="$LOGD/$1.log"
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$envs]"
  env STRATA_VK_BIN="$BIN" $envs timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" || true
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  echo "  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)  $(grep -m1 '^prefill ' "$L")"
  grep -m1 'strata prefill timing' "$L" | sed 's/^/  /'
  grep -m1 'strata prefill timing: host:' "$L" | sed 's/^/  /' || true
  if grep -q 'vk kernel time' "$L"; then grep -m1 'vk kernel time' "$L" | sed 's/^/  /'; fi
  # rank the phases descending, from the raw line (a summary is not evidence)
  grep -m1 'strata prefill timing' "$L" | sed 's/.*ms://' | tr ' ' '\n' | grep -v '^$' | paste - - 2>/dev/null \
    | sort -k2 -nr | head -12 | sed 's/^/    /'
}

for r in $(seq 1 "$REPS"); do one "pfph_$r" "STRATA_PREFILL_TIMING=1"; done
if [ "$KERNEL_TIME" = 1 ]; then one "pfph_kt_1" "STRATA_PREFILL_TIMING=1 STRATA_VK_KERNEL_TIME=1"; fi
echo "== prefill_phase DONE $(date -Is)"
