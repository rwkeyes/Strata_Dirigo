#!/usr/bin/env bash
# /home/bob/step4/task1_arms.sh -- Task 1.2 probes on the INSTRUMENTED ABLATION binary saved as
#  bins/strata_vulkan.base9cd885e (sha 714b760c...).  The NAME says "baseline"; the BYTES do not: this file
#  carries the TRIVIAL_REC_FAMILY instrument and is byte-identical to the instrumented ablation build.  It is
#  correct to RUN (the instrument is env-gated and inert when its variable is unset), but it is NOT a baseline,
#  and the label used to mislead the next reader.  A sha assertion below pins the CONTENT, so a future copy
#  that is not what its name claims fails loudly.
#  (a) the port's ONE existing second-queue overlap: STRATA_SH_STREAM 1 (fork shared expert -> sh_cs_) vs 0 (same queue).
#      Interleaved n=3; ids must stay 3aed108cceee.
#  (b) --spec-split clean (no pair flag): the reorder's own cost.
set -u
BIN=${TASK1_BIN:-/home/bob/step4/bins/strata_vulkan.base9cd885e}
BIN_SHA_WANT=${TASK1_BIN_SHA:-714b760c7d531f9a341ca3e20ede195933c75aff20eed32db1094d08f5d8222d}
T8="1 2 3 4 5 6 7 8"
LOGD=/home/bob/step4/logs
one(){  # one <name> <SH|-> <extra run arg or ->
  local name="$1"; local shv="$2"; local extra="$3"
  local envs=()
  [ "$shv" != "-" ] && envs+=("STRATA_SH_STREAM=$shv")
  local args=()
  [ "$extra" != "-" ] && args+=("$extra")
  echo "[$(date '+%H:%M:%S')] ARM $name env=[${envs[*]:-none}] args=[${args[*]:-none}]"
  env STRATA_VK_BIN="$BIN" "${envs[@]}" timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T8\" ${args[*]:-}" || true
  local L="$LOGD/$name.log"
  local id dec sync spec
  id=$(grep -m1 ARM_OUTPUT_ID "$L" 2>/dev/null | cut -d' ' -f2)
  dec=$(grep -m1 '^decode' "$L" 2>/dev/null | sed 's/^decode *//')
  sync=$(grep -m1 'verify window' "$L" 2>/dev/null | grep -oE 'sync [0-9.]+' | awk '{print $2}')
  spec=$(grep -m1 'speculation' "$L" 2>/dev/null | tr -s ' ' | sed 's/^ *//')
  printf '    RESULT %-12s id=%s sync=%s | %s | %s\n' "$name" "${id:-NONE}" "${sync:-NONE}" "${dec:-NONE}" "${spec:-NONE}"
}
# THE ASSERTION: the file named base9cd885e must be the INSTRUMENTED ABLATION build, not a baseline.
if ! bash /home/bob/step4/assert_sha.sh "task1 bin label (base9cd885e = instrumented ablation)" "$BIN" "$BIN_SHA_WANT"; then
  echo "TASK1_REFUSES reason=bin-sha-mismatch name=base9cd885e got=$(sha256sum "$BIN" 2>/dev/null | cut -d' ' -f1) want=$BIN_SHA_WANT" >&2
  echo "TASK1_REFUSES note=the file is NAMED like a baseline but its bytes must be the instrumented ablation build" >&2
  echo "TASK1_REFUSES note=(TRIVIAL_REC_FAMILY present); a copy that is not what its name claims must fail loudly" >&2
  exit 2
fi
echo "== task1_arms $(date -Is)  BIN=$BIN sha=$(sha256sum $BIN | cut -c1-16) (instrumented-ablation, asserted)"
if [ "${TASK1_CHECK_ONLY:-0}" = 1 ]; then
  echo "TASK1_ASSERT_ONLY the bin identity assertion PASSED; arms NOT run (TASK1_CHECK_ONLY=1)"
  exit 0
fi
for r in 1 2 3; do
  one "sh2q_${r}" 1 -
  one "sh1q_${r}" 0 -
done
one split_clean_1 - --spec-split
echo "== task1_arms DONE $(date -Is)"
