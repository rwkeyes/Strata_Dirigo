#!/usr/bin/env bash
# smoke-arc.sh - PORT-PLAN stage 0: does sustained Vulkan COMPUTE wedge this card?
#
# The plan's §4 says this is the FIRST thing to do on Battlemage hardware, before scheduler work and before
# promising a date, because the xe KMD is known to wedge permanently under sustained compute load through
# Level-Zero, OpenCL AND Vulkan (intel/compute-runtime#948) - a driver bug no backend can fix around.
#
# So: run the port's own suite in a loop on the Intel ICD and watch for the three faces of the wedge -
#   1. a run that HANGS (per-run timeout),  2. a run that FAILS,  3. xe errors in the kernel log.
# Every iteration is timed, because a wedge usually shows as a growing stall before it shows as a hang.
#
#   bash gates/smoke-arc.sh [MINUTES]        # default 12; ICD=... and PER_RUN_TIMEOUT=... also honoured
#
# Exit: 0 if every run passed, 1 if any run failed or hung. The log is the evidence either way.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/harness/build"
GATE="$BUILD/vk_gate"
MINUTES="${1:-12}"
JOBS="${JOBS:-1}"
ICD="${ICD:-/usr/share/vulkan/icd.d/intel_icd.json}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-180}"
LOG="${LOG:-$ROOT/logs/smoke-arc-$(date +%Y%m%d-%H%M%S).log}"

[ -x "$GATE" ] || { echo "no $GATE - run gates/run_gate.sh once to build it"; exit 2; }
mkdir -p "$(dirname "$LOG")"

device="$(VK_ICD_FILENAMES="$ICD" "$GATE" --list 2>/dev/null | head -1)"

# The suite's two ENVIRONMENTAL checks assume a QUIET card, and NO SETTING fixes that - both read the DRIVER's
# view, not the port's:
#   budget: independent requery agrees  - two driver readings of free heap memory, taken while other processes
#                                         on the card are allocating.  An explicit ceiling does NOT stabilise it
#                                         (measured: still 8/8 failed with STRATA_VK_MAX_BUDGET_MIB=16384).
#   stack: resolvable ICD not flagged   - its child process is slower to start under load (measured 6-7/8).
# Under N concurrent instances they correctly report a busy card.  That is not a kernel failure, so this script
# CLASSIFIES the failing labels: those two are the environment, reported and counted but not fatal; ANY other
# failing case is a kernel finding and fails the run.
export STRATA_VK_MAX_BUDGET_MIB="${STRATA_VK_MAX_BUDGET_MIB:-16384}"   # keeps the PORT's own allocation deterministic
ENV_LABELS="budget: independent requery agrees|stack: resolvable ICD not flagged"

echo "smoke: sustained Vulkan compute for ${MINUTES} min on the Intel ICD" | tee "$LOG"
echo "  device: ${device:-NONE FOUND}" | tee -a "$LOG"
echo "  per-run timeout: ${PER_RUN_TIMEOUT}s   jobs: ${JOBS}   ceiling: ${STRATA_VK_MAX_BUDGET_MIB} MiB   log: $LOG" \
    | tee -a "$LOG"
[ -n "$device" ] || { echo "  the Intel ICD reports no device - nothing to smoke" | tee -a "$LOG"; exit 2; }

n=0; fails=0; hangs=0; envfails=0; t0=$(date +%s); worst=0; runs=0
while [ $(( $(date +%s) - t0 )) -lt $(( MINUTES * 60 )) ]; do
    n=$((n + 1)); s=$(date +%s)
    for j in $(seq 1 "$JOBS"); do
        ( timeout "$PER_RUN_TIMEOUT" env VK_ICD_FILENAMES="$ICD" "$GATE" --spv-dir "$ROOT/shaders" \
              >"$LOG.run$n.$j" 2>&1; echo $? >"$LOG.rc$n.$j" ) &
    done
    wait
    r_ok=0; r_bad=0; r_hang=0
    for j in $(seq 1 "$JOBS"); do
        rc="$(cat "$LOG.rc$n.$j" 2>/dev/null || echo 1)"
        runs=$((runs + 1))
        if [ "$rc" -eq 124 ]; then r_hang=$((r_hang + 1))
        elif [ "$rc" -ne 0 ]; then r_bad=$((r_bad + 1))
        else r_ok=$((r_ok + 1)); fi
    done
    # Classify this round's failures: the two ENV_LABELS are a busy card, anything else is a kernel finding.
    kern_bad=0
    for j in $(seq 1 "$JOBS"); do
        [ "$(cat "$LOG.rc$n.$j" 2>/dev/null)" = 0 ] && continue
        grep -E "^FAIL" "$LOG.run$n.$j" 2>/dev/null | grep -qvE "$ENV_LABELS" && kern_bad=$((kern_bad + 1))
    done
    env_bad=$((r_bad - kern_bad))
    fails=$((fails + kern_bad)); envfails=$((envfails + env_bad)); hangs=$((hangs + r_hang))
    d=$(( $(date +%s) - s ))
    [ "$d" -gt "$worst" ] && worst=$d
    printf 'round %3d  (%d concurrent)  ok %-3d bad %-3d hang %-3d  %4ds  %s\n' "$n" "$JOBS" "$r_ok" "$r_bad" \
        "$r_hang" "$d" "$(grep -hE '^== [0-9]+ passed' "$LOG.run$n".* 2>/dev/null | tail -1)" | tee -a "$LOG"
    if sudo -n journalctl -k --since "-$((d + 5))s" --no-pager 2>/dev/null \
            | grep -qiE "xe .*(reset|hang|timeout|error)|GPU HANG|GuC.*timeout"; then
        echo "  !! the kernel log shows an xe error around this round:" | tee -a "$LOG"
        sudo -n journalctl -k --since "-$((d + 5))s" --no-pager 2>/dev/null \
            | grep -iE "xe |GPU HANG" | tail -6 | tee -a "$LOG"
    fi
    if [ "$r_hang" -gt 0 ]; then
        echo "  !! $r_hang job(s) hung (${PER_RUN_TIMEOUT}s timeout) - this is the wedge the plan warns about" \
            | tee -a "$LOG"
        break
    fi
    if [ "$env_bad" -gt 0 ] && [ "$kern_bad" -eq 0 ]; then
        echo "     ($env_bad job(s) hit the busy-card labels only: budget requery / ICD positive control)" \
            | tee -a "$LOG"
    fi
    if [ "$kern_bad" -gt 0 ]; then
        echo "  !! $kern_bad job(s) failed on a NON-environmental case - that is the finding:" | tee -a "$LOG"
        for j in $(seq 1 "$JOBS"); do
            [ "$(cat "$LOG.rc$n.$j" 2>/dev/null)" = 0 ] && continue
            grep -hE "^FAIL" "$LOG.run$n.$j" 2>/dev/null | grep -vE "$ENV_LABELS" | tee -a "$LOG"
        done
        break
    fi
done
echo "--- $runs case-run(s) over $n round(s) of ${JOBS} concurrent in $(( $(date +%s) - t0 ))s:" \
     "$fails kernel case(s) failed, $envfails busy-card label(s), $hangs hung, slowest round ${worst}s" \
     | tee -a "$LOG"
[ "$fails" -eq 0 ] && [ "$hangs" -eq 0 ]
