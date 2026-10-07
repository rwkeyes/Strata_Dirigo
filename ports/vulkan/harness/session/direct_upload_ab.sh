#!/usr/bin/env bash
# /home/bob/step4/direct_upload_ab.sh -- A/B THE STRATA_VK_DIRECT_UPLOAD PROTOTYPE (lever C).
#
# WHAT IS BEING TESTED.  With the flag, `alloc_device` hands out the MAPPABLE device-local type and `stage_upload`
# makes a host store instead of staging + copy + submit + fence-wait.  Measured basis: host store 5.64 GB/s against
# the staged path's 1.92 (ports/vulkan/tools/probe_mem.cpp), and the GPU reads a mappable allocation ~3% slower
# (ports/vulkan/tools/probe_gpuread.cpp, 499.71 vs 512.49 GB/s).  Applied to the 53.7 GB of cold-start uploads the
# prediction is roughly -18 s of a ~163 s startup for ~0.4% of the prefill.
#
# WHAT WOULD FALSIFY IT.  Three ways, all checked here:
#   1. THE WALL TIME DOES NOT MOVE.  Every arm is timed end to end (`date +%s` around the whole engine run): the
#      prediction is ~-18 s per flagged arm and nothing for the default.  If the flagged arms are not clearly
#      faster, the 5.64 GB/s figure does not reach the real workload and C is worthless.
#   2. THE TRANSFER ARM DOES NOT COLLAPSE.  `vk disp stat by arm:` reports `transfer <n>`; the default arm should
#      read 13,752 and the flagged arm should read ~0, because a host store submits nothing.  If it stays at 13,752
#      the direct path is not being taken (the destinations were not actually mappable) - a silent no-op.
#   3. THE ID MOVES.  `56a0b28d2de6` must hold in every arm.  A direct host store into coherent memory changes HOW
#      the bytes arrive, never WHICH bytes; an id move means a visibility/ordering bug and kills the prototype
#      regardless of the speed.
# `prefill`/`decode` are recorded too so the ~3% GPU-read cost of the mapping can be seen where it would land.
set -u
LOGD=/home/bob/step4/logs
BIN=${PF_BIN:-/home/bob/vkbuild-vulkan/vulkan/strata_vulkan}
T199="$(printf '1 %.0s' $(seq 1 199))"
TSV=/home/bob/step4/direct_upload_ab.tsv
: > "$TSV"
echo "== direct_upload_ab $(date -Is)  bin=$(sha256sum "$BIN" | cut -c1-16)  rounds=2"
printf '%-10s %-32s %-14s %8s %9s %8s %8s %6s %s\n' arm env id wall_s prefill_ms tok_s dec_tok_s transfer verdict | tee -a "$TSV"

one() {   # one <name> <env string>
  local name="$1" envs="$2"
  local L="$LOGD/$name.log"
  local t0 t1
  t0=$(date +%s)
  env $envs STRATA_VK_BIN="$BIN" STRATA_PREFILL_TIMING=1 STRATA_VK_DISP_STAT=1 STRATA_VK_XFER_STAT=1 \
      timeout 1800 flock /tmp/b70.lock -c \
      "bash /home/bob/step4/run.sh $name \"$T199\"" >/dev/null 2>&1 || true
  t1=$(date +%s)
  local wall=$((t1 - t0))
  local id pf toks dt transfer verdict
  id=$(grep -m1 ARM_OUTPUT_ID "$L" | cut -d' ' -f2)
  pf=$(grep -m1 '^prefill ' "$L" | sed -n 's/.*tokens in \([0-9.]*\) ms.*/\1/p')
  toks=$(grep -m1 '^prefill ' "$L" | sed -n 's/.*-> *\([0-9.]*\) tok.*/\1/p')
  dt=$(grep -m1 '^decode ' "$L" | sed -n 's/.*-> *\([0-9.]*\) tok.*/\1/p')
  transfer=$(grep -m1 'vk disp stat by arm' "$L" | sed -n 's/.*transfer \([0-9]*\).*/\1/p')
  verdict=""
  [ "$id" != "56a0b28d2de6" ] && verdict="ID-MOVED"
  printf '%-10s %-32s %-14s %8s %9s %8s %8s %6s %s\n' "$name" "${envs:-（none）}" "$id" "$wall" "$pf" "$toks" "$dt" "${transfer:-?}" "$verdict" | tee -a "$TSV"
}

for r in 1 2; do
  one "du_A$r" ""
  one "du_B$r" "STRATA_VK_DIRECT_UPLOAD=1"
done
echo
echo "== the upload path itself, per arm (submits vs bytes) =="
for a in du_A1 du_B1 du_A2 du_B2; do
  L="$LOGD/$a.log"
  [ -f "$L" ] || continue
  printf '  %-7s %s | %s\n' "$a" "$(grep -m1 'vk disp stat ms:' "$L" | sed 's/vk disp stat ms: //')" "$(grep -m1 'vk xfer stat:' "$L" | sed 's/vk xfer stat: //')"
done
echo "== DONE $(date -Is)"
