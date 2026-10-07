#!/usr/bin/env bash
# /home/bob/step4/gate_run.sh <tag> -- run the port's ACCEPTANCE GATE once and stamp its log with the
# identity of the run: the gate BINARY's sha256, the commit, and the `git status --porcelain` state.
#
# WHY: the gate (ports/vulkan/gates/run_gate.sh) is the project's acceptance test -- the line
# `intel_icd == 965 passed, 0 failed, 0 skipped` is what every milestone rests on -- and it was the ONE
# artifact with no identity recorded, so a verdict could be quoted without saying which gate binary
# produced it.  The gate binary is REBUILT by run_gate.sh on EVERY run, so its sha is read AFTER the run
# (the binary that actually produced the verdict); the log is written as: identity header + gate body + end.
#
# The identity header gate_run.sh writes (checked by `arm_gate.py --check-gate-logs`):
#   == gate run <tag> | <ts>
#   == gate repo: <path>
#   == gate commit: <40-hex|none>
#   == gate status sha256: <64-hex of `git status --porcelain`>
#   == gate status lines: <N>
#   == gate bin path: <path>
#   == gate bin sha256: <64-hex|MISSING>
#   == gate runner: <path>
#   == gate status --porcelain:
#   <the actual porcelain lines, or "(clean)">
#   == gate body follows
#
# Env overrides (used by selftest_gate_identity.sh; defaults are the real tree):
#   GATE_REPO GATE_RUNNER GATE_BIN GATE_LOGD GATE_LOG GATE_LOCK GATE_TIMEOUT
set -u
R=${GATE_REPO:-/home/bob/strata-vulkan-wt}
TAG=${1:?usage: gate_run.sh <tag>}
LOGD=${GATE_LOGD:-/home/bob/step4/logs}
LOG=${GATE_LOG:-$LOGD/gate_$TAG.log}
RUNNER=${GATE_RUNNER:-$R/ports/vulkan/gates/run_gate.sh}
GATEBIN=${GATE_BIN:-$R/ports/vulkan/harness/build/vk_gate}
LOCK=${GATE_LOCK:-/tmp/b70.lock}
mkdir -p "$LOGD"
export R GATEBIN TAG

BODY="$LOG.body.$$"
echo "[$(date '+%H:%M:%S')] gate_run tag=$TAG runner=$RUNNER"
timeout "${GATE_TIMEOUT:-2400}" flock "$LOCK" -c "cd '$R' && bash '$RUNNER'" > "$BODY" 2>&1
rc=$?

commit=$(git -C "$R" rev-parse HEAD 2>/dev/null || echo none)
porc=$(git -C "$R" status --porcelain 2>/dev/null || true)
porc_sha=$(printf '%s' "$porc" | sha256sum | cut -d' ' -f1)
porc_n=$(printf '%s' "$porc" | grep -c . )
gsha=$(sha256sum "$GATEBIN" 2>/dev/null | cut -d' ' -f1)
[ -n "$gsha" ] || gsha=MISSING

{
  echo "== gate run $TAG | $(date -Is)"
  echo "== gate repo: $R"
  echo "== gate commit: $commit"
  echo "== gate status sha256: $porc_sha"
  echo "== gate status lines: $porc_n"
  echo "== gate bin path: $GATEBIN"
  echo "== gate bin sha256: $gsha"
  echo "== gate runner: $RUNNER"
  echo "== gate status --porcelain:"
  if [ -n "$porc" ]; then printf '%s\n' "$porc"; else echo "  (clean)"; fi
  echo "== gate body follows"
  cat "$BODY"
  echo "== gate end $TAG | $(date -Is) rc=$rc"
} > "$LOG"
rm -f "$BODY"

echo "GATE_RUN tag=$TAG rc=$rc gate_bin_sha256=$gsha commit=$commit status_sha256=$porc_sha log=$LOG"
exit "$rc"
