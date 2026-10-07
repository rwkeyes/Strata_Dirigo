#!/usr/bin/env bash
# /home/bob/step4/gemm_tiled_ab.sh -- with the KU=8 unroll in place, does the MATRIX-UNIT default still pay?
#
#   A (shipped): STRATA_VK_PREFILL_COOPMAT unset  -> T >= 8 goes to gemm_prefill_f16_m8.spv (the matrix units)
#   B (FMA)    : STRATA_VK_PREFILL_COOPMAT=0      -> the KU=8 FMA kernel takes every shape
#
# WHY THIS IS NOW ASKED.  The bench says the unrolled FMA kernel beats the cooperative-matrix kernel at EVERY
# measured shape once KU=8 is in (T=8: 0.1025 against 0.4017; T=199: 1.4690 against 3.3426; down T=199: 0.6445
# against 1.6094).  The shape rule that prefers the matrix units predates the unroll.  Isolated wins have fooled
# this port before (the tiled kernel is the fastest at large T in isolation and was a WASH end to end), so this is
# an engine A/B with the same guards: interleaved arms, one binary, ids checked in every arm.
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
BASE_ID=56a0b28d2de6
ROUNDS=${ROUNDS:-2}
T199="$(printf '1 %.0s' $(seq 1 199))"
echo "== gemm_tiled_ab $(date -Is)  BIN=$(sha256sum "$BIN"|cut -c1-16)  gemm_spv=$(sha256sum /home/bob/strata-vulkan-wt/ports/vulkan/shaders/gemm_prefill_fma_small.spv | cut -c1-16)  rounds=$ROUNDS"
TSV=/home/bob/step4/gemm_tiled_ab.tsv; : > "$TSV"

one() {  # one <name> <env-prefix>
  local name="$1" envs="$2"
  local L="$LOGD/$name.log"
  echo "[$(date '+%H:%M:%S')] ARM $name env=[$envs]"
  env STRATA_VK_BIN="$BIN" STRATA_PREFILL_TIMING=1 $envs timeout 900 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" || true
  local id gu d cm pre tac m8
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  gu=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gemm gate\/up \([0-9][0-9]*\) .*/\1/p')
  d=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*gemm down \([0-9][0-9]*\) .*/\1/p')
  pre=$(grep -m1 'strata prefill timing' "$L" | sed -n 's/.*GPU timeline \([0-9][0-9]*\) ms.*/\1/p')
  tac=$(grep -m1 '^prefill ' "$L" | sed 's/.*tokens in \([0-9.]*\) ms.*-> *\([0-9.]*\) tok.*/\1|\2/')
  # which kernels actually ran: the matrix-unit kernel and the untiled FMA in this arm's own histogram
  cm=$(grep -m1 'vk disp stat by shader' "$L" | grep -o 'gemm_prefill_f16_m8.spv [0-9]*' | awk '{print $2}')
  m8=$(grep -m1 'vk disp stat by shader' "$L" | grep -o 'gemm_prefill_fma_small.spv [0-9]*' | awk '{print $2}')
  grep -m1 ARM_DIDNOTRUN "$L" || true
  grep -m1 ARM_DEGENERATE "$L" || true
  if [ -n "$id" ] && [ "$id" != "$BASE_ID" ]; then echo "  ID_MOVED id=$id (baseline $BASE_ID)"; else echo "  id=$id"; fi
  printf '  cm_kernel=%s  fma_small=%s  gemm_gate_up=%s ms  gemm_down=%s ms  gpu_timeline=%s ms  prefill=%s ms  %s tok/s\n' \
    "${cm:-0}" "${m8:-0}" "${gu:-NA}" "${d:-NA}" "${pre:-NA}" "$(echo "$tac"|cut -d'|' -f1)" "$(echo "$tac"|cut -d'|' -f2)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$envs" "$id" "${gu:-NA}" "${d:-NA}" "${pre:-NA}" "$(echo "$tac"|cut -d'|' -f1)" "$(echo "$tac"|cut -d'|' -f2)" "${cm:-0}/${m8:-0}" >> "$TSV"
}

for r in $(seq 1 "$ROUNDS"); do
  one "gt_A_$r" ""                                        # shipped: matrix units for T>=8
  one "gt_B_$r" "STRATA_VK_PREFILL_TILED=1"             # the KU=8 FMA kernel everywhere
done
echo
printf '%-8s %-30s %-14s %-9s %-9s %-11s %-11s %s\n' arm env id gate_up down gpu_tl prefill tok_s
awk -F'\t' '{printf "%-8s %-30s %-14s %-9s %-9s %-11s %-11s %s\n", $1, ($2==""?"(shipped)":$2), $3, $4, $5, $6, $7, $8}' "$TSV"
echo "== gemm_tiled_ab DONE $(date -Is)"
