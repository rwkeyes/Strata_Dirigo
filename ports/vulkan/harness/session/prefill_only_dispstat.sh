#!/usr/bin/env bash
# /home/bob/step4/prefill_only_dispstat.sh -- ATTRIBUTE THE DISPATCH LAYER TO THE PREFILL ALONE.
#
# WHY.  `vk disp stat ms: total 44,348 = ... submit 16,558 + wait 27,625 ...` is the WHOLE RUN (prefill + 32 decoded
# tokens + the spec/verify windows).  The prefill's own share is what matters for a prefill performance question,
# and the port's submit cost is the biggest unexplained number in its records (~1.15 ms per submit against
# upstream SYCL's ~14.5 us PER DISPATCH).  `--max-new 1` gives a run that is essentially the prompt.
#
# Outputs: the prompt's own prefill line, the phase table, and the dispatch-layer split - side by side with the
# 32-token arm so the decode's contribution is visible by subtraction.
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
T199="$(printf '1 %.0s' $(seq 1 199))"
run_one() {  # run_one <name> <extra flags...>
  local name="$1"; shift
  local L="$LOGD/$name.log"
  echo "[$(date '+%H:%M:%S')] $name  extra=[$*]"
  env STRATA_VK_BIN="$BIN" STRATA_PREFILL_TIMING=1 STRATA_DECODE_TIMING=1 STRATA_VK_DISP_STAT=1 \
      timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\" $*" || true
  echo "  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)"
  grep -m1 '^prefill ' "$L" | sed 's/^/  /'
  grep -m1 '^decode' "$L" | sed 's/^/  /'
  grep -m1 'vk disp stat:' "$L" | sed 's/^/  /'
  grep -m1 'vk disp stat ms:' "$L" | sed 's/^/  /'
}

echo "== prefill_only_dispstat $(date -Is)  bin=$(sha256sum "$BIN"|cut -c1-16)"
run_one pfo_max1 --max-new 1              # the prompt, essentially alone
run_one pfo_max32                          # the same run the phase tables come from
echo "== the prefill's own dispatch-layer share = (max1 totals) minus a small decode residue"
echo "== DONE $(date -Is)"
